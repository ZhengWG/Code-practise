"""
LLM 推理优化笔记 —— 原理提纲 + 核心代码

结构：
  开篇     注意力 9 步总览（读懂后面所有变体改的是哪一步）
  Part A–K 各主题的核心实现（已去掉自测打印与演示驱动）

说明：
  · 以可读性优先；部分片段依赖 torch/numpy，不保证作为单文件一键跑通
"""

# ============================================================
# 开篇 · 一次注意力的全部步骤（总览）
# ============================================================

def attention(x, cache, pos):
    B, T, D = x.shape
    q = Wq(x).view(B, T, H, dh).transpose(1, 2)      # ① 投影；GQA 改 K/V 头数，MLA 改低秩 c
    k = Wk(x).view(B, T, H, dh).transpose(1, 2)
    v = Wv(x).view(B, T, H, dh).transpose(1, 2)
    q, k = RoPE(q, pos), RoPE(k, pos)                # ② 只加在 q/k
    k, v = cache.append(k, v)                        # ③ KV Cache —— 显存瓶颈
    score = q @ k.transpose(-1, -2) / dh**0.5        # ④⑤ 打分+缩放
    score = score.masked_fill(~(q_pos[:, None] >= k_pos[None, :]), -inf)  # ⑥ 因果；Sparse 在此挖洞
    p = score.softmax(-1)                            # ⑦ Linear Attention 删此步
    o = p @ v                                        # ⑧
    return Wo(o.transpose(1, 2).reshape(B, T, D))    # ⑨

# 三条事实:
#   · ④⑧ 是仅有的两个矩阵乘，长序列 O(S²) —— Sparse/Linear 打这两个
#   · ③  是显存瓶颈 —— GQA/MLA 打这个
#   · FlashAttention 不改数学，融 ④~⑧、不写回 (T,S) —— 省带宽不省 FLOPs
#
# 优化轴（可叠加，也有冲突）:
#   GQA/MLA 省 KV 体积 | Sparse/Linear 省注意力算力 | EAGLE/MTP 省串行前向
#   HiCache 省重复 prefill | EP 放大 batch | compile×graph 省 launch
#   冲突: MLA 朴素/吸收要按 draft 长度切换; Linear 破坏 prefix 截断;
#         投机在大 batch 退化; HiCache 与权重/通信抢 PCIe

# ============================================================
# Part A · Transformer 核心 (RoPE / GQA / KV Cache / FLOPs)
# ============================================================

"""
01 · Transformer 核心结构
=========================
MHA → GQA → RoPE → KV Cache → Prefill/Decode 两条路径

对应文档：§1.1 FLOPs 推导、§1.3 KV Cache 大小、§2.1 计算量

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_rope_cache(seq_len, head_dim, base=10000.0, dtype=torch.float32):
    """预计算 cos/sin。注意 head_dim 必须是偶数（成对旋转）。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, dtype=dtype)
    freqs = torch.outer(t, inv_freq)              # [S, hd/2]
    return freqs.cos(), freqs.sin()               # 各 [S, hd/2]

def apply_rope(x, cos, sin):
    """
    x: [B, H, S, hd]  →  旋转后同形状
    把 hd 拆成前后两半 (x1, x2)，做二维旋转：
        x1' = x1*cos - x2*sin
        x2' = x1*sin + x2*cos
    这是 Llama/GPT-NeoX 风格的 "half rotation"。
    """
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    cos = cos[None, None, :, :]                    # [1,1,S,hd/2]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)

class GQAAttention(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, head_dim=None):
        super().__init__()
        assert n_head % n_kv_head == 0
        self.n_head, self.n_kv = n_head, n_kv_head
        self.hd = head_dim or d_model // n_head
        self.n_rep = n_head // n_kv_head           # 每个 KV 头被复制几次

        self.q_proj = nn.Linear(d_model, n_head * self.hd, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_head * self.hd, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_head * self.hd, bias=False)
        self.o_proj = nn.Linear(n_head * self.hd, d_model, bias=False)

    @staticmethod
    def _repeat_kv(x, n_rep):
        """[B, n_kv, S, hd] → [B, n_kv*n_rep, S, hd]（不复制显存的 expand）"""
        if n_rep == 1:
            return x
        B, n_kv, S, hd = x.shape
        return x[:, :, None].expand(B, n_kv, n_rep, S, hd).reshape(B, n_kv * n_rep, S, hd)

    def forward(self, x, cos, sin, kv_cache=None, causal=True):
        """
        kv_cache: None=prefill；(k_prev, v_prev)=decode
        返回 (out, new_kv_cache)
        """
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv,  self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv,  self.hd).transpose(1, 2)

        # RoPE 只作用于 q/k，v 不加位置
        past = 0 if kv_cache is None else kv_cache[0].shape[2]
        q = apply_rope(q, cos[past:past + S], sin[past:past + S])
        k = apply_rope(k, cos[past:past + S], sin[past:past + S])

        # ★ KV Cache：把历史 K/V 拼上（decode 时 S=1，拼成 [B,n_kv,past+1,hd]）
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        kr, vr = self._repeat_kv(k, self.n_rep), self._repeat_kv(v, self.n_rep)
        attn = (q @ kr.transpose(-1, -2)) / math.sqrt(self.hd)   # [B,H,S,S_kv]

        if causal and S > 1:
            S_kv = kr.shape[2]
            mask = torch.ones(S, S_kv, dtype=torch.bool).tril(diagonal=S_kv - S)
            attn = attn.masked_fill(~mask, float('-inf'))

        out = (attn.softmax(-1) @ vr).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out), new_cache

class SwiGLUMLP(nn.Module):
    """SwiGLU：三个矩阵 gate/up/down，FLOPs = 3 × 2·s·d·d_ff"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    """Pre-Norm 结构：x + Attn(Norm(x))，x + MLP(Norm(x))"""
    def __init__(self, d_model, n_head, n_kv, d_ff):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d_model), nn.RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_head, n_kv)
        self.mlp = SwiGLUMLP(d_model, d_ff)

    def forward(self, x, cos, sin, kv=None):
        h, new_kv = self.attn(self.n1(x), cos, sin, kv)
        x = x + h
        return x + self.mlp(self.n2(x)), new_kv

class TinyLLM(nn.Module):
    def __init__(self, vocab=512, d_model=128, n_layer=4, n_head=8, n_kv=2, d_ff=256):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d_model, L=n_layer,
                        H=n_head, n_kv=n_kv, d_ff=d_ff, hd=d_model // n_head)
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, n_kv, d_ff) for _ in range(n_layer)])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        cos, sin = build_rope_cache(4096, d_model // n_head)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, ids, caches=None):
        x = self.emb(ids)
        new_caches = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, self.cos, self.sin, None if caches is None else caches[i])
            new_caches.append(kv)
        return self.lm_head(self.norm(x)), new_caches

def flops_forward(L, d, s, d_ff, n_head, n_kv, hd):
    """精确逐项统计（每个 MAC 记 2 FLOPs）"""
    q  = 2 * s * d * (n_head * hd)
    kv = 2 * 2 * s * d * (n_kv * hd)
    qk = 2 * s * s * (n_head * hd)          # QK^T
    av = 2 * s * s * (n_head * hd)          # softmax·V
    o  = 2 * s * (n_head * hd) * d
    mlp = 3 * 2 * s * d * d_ff
    per_layer = q + kv + qk + av + o + mlp
    return per_layer * L, dict(qkvo=(q + kv + o) * L, attn=(qk + av) * L, mlp=mlp * L)

def kv_bytes_per_token(L, n_kv, hd, dtype_bytes=2):
    """★ KV/token = 2(K,V) × L × n_kv × head_dim × dtype"""
    return 2 * L * n_kv * hd * dtype_bytes

# ============================================================
# Part B · MLA 矩阵吸收
# ============================================================

"""
02 · MLA (Multi-head Latent Attention) —— DeepSeek-V2/V3/Kimi-K2.6
==================================================================
核心：KV 不再按头存，而是压成一个低秩 latent c_KV (512维)，全部头共享。

★★ 最关键、也最容易讲错的一点：**矩阵吸收 (Matrix Absorption)**
   如果 decode 时把 c_KV 解压回完整的 K/V 再算注意力，那显存是省了，
   但每步都要做一次 [S,512]×[512, n_h*hd] 的解压 GEMM，算力反而变大。
   矩阵吸收 = 把解压矩阵 W_UK 预先融进 W_UQ，让 query 直接在 latent 空间里做点积，
   **根本不需要解压 K**。这才是 MLA 在 decode 时真正省算力+省带宽的原因。

对应文档：§1.2、§1.5.1（DeepSeek 的"通道维压缩"）

"""
import math
import torch
import torch.nn as nn

def rope_cache(S, dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    f = torch.outer(torch.arange(S).float(), inv)
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):
    """x: [B,H,S,dim]"""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

class MLA(nn.Module):
    """
    DeepSeek-V3 配置：d=7168, n_head=128, qk_nope=128, qk_rope=64, v_head=128,
                      q_lora_rank=1536, kv_lora_rank=512
    ★ 每个 token 只缓存 (c_KV[512] + k_rope[64]) = 576 维 —— 与头数无关！
    """
    def __init__(self, d_model=512, n_head=8, qk_nope=32, qk_rope=16,
                 v_head=32, kv_lora=64, q_lora=128):
        super().__init__()
        self.n_head, self.qk_nope, self.qk_rope = n_head, qk_nope, qk_rope
        self.v_head, self.kv_lora = v_head, kv_lora
        self.qk_head = qk_nope + qk_rope           # 每个头的 query 总维度

        # ── Query 侧：也做低秩（省参数，不影响 KV cache）
        self.q_down = nn.Linear(d_model, q_lora, bias=False)
        self.q_norm = nn.RMSNorm(q_lora)
        self.q_up = nn.Linear(q_lora, n_head * self.qk_head, bias=False)

        # ── KV 侧：★ 压缩成 latent。输出 = kv_lora(共享压缩) + qk_rope(解耦位置)
        self.kv_down = nn.Linear(d_model, kv_lora + qk_rope, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora)
        # 解压矩阵：latent → 每个头的 K_nope 和 V
        self.kv_up = nn.Linear(kv_lora, n_head * (qk_nope + v_head), bias=False)

        self.o_proj = nn.Linear(n_head * v_head, d_model, bias=False)
        self.scale = self.qk_head ** -0.5

    # ────────────────────────────────────────────────────────────
    # 路径 A：朴素实现（解压回完整 K/V 再算）—— 用于 prefill，也作为正确性基准
    # ────────────────────────────────────────────────────────────
    def forward_naive(self, x, cos, sin, cache=None):
        B, S, _ = x.shape
        H, nope, rope, vh = self.n_head, self.qk_nope, self.qk_rope, self.v_head

        q = self.q_up(self.q_norm(self.q_down(x))).view(B, S, H, self.qk_head).transpose(1, 2)
        q_nope, q_rope = q.split([nope, rope], -1)

        kv = self.kv_down(x)
        c_kv, k_rope = kv.split([self.kv_lora, rope], -1)   # [B,S,512], [B,S,64]
        c_kv = self.kv_norm(c_kv)

        past = 0 if cache is None else cache[0].shape[1]
        q_rope = apply_rope(q_rope, cos[past:past + S], sin[past:past + S])
        k_rope = apply_rope(k_rope.unsqueeze(1), cos[past:past + S], sin[past:past + S])

        # ★★ 缓存的只有这两个：c_kv[B,S,512] 和 k_rope[B,1,S,64]
        if cache is not None:
            c_kv = torch.cat([cache[0], c_kv], 1)
            k_rope = torch.cat([cache[1], k_rope], 2)
        new_cache = (c_kv, k_rope)

        # 解压：latent → 每头的 K_nope / V
        kv_up = self.kv_up(c_kv).view(B, -1, H, nope + vh).transpose(1, 2)
        k_nope, v = kv_up.split([nope, vh], -1)
        k = torch.cat([k_nope, k_rope.expand(-1, H, -1, -1)], -1)   # rope 部分全头共享

        # ⚠️ 必须用 apply_rope 之后的 q_rope 重新拼出 q。
        #    split() 返回的是 view，上面对 q_rope 的重新赋值并不会写回 q。
        q = torch.cat([q_nope, q_rope], -1)
        attn = (q @ k.transpose(-1, -2)) * self.scale
        S_kv = k.shape[2]
        if S > 1:
            mask = torch.ones(S, S_kv, dtype=torch.bool).tril(S_kv - S)
            attn = attn.masked_fill(~mask, float('-inf'))
        out = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out), new_cache

    # ────────────────────────────────────────────────────────────
    # 路径 B：★★ 矩阵吸收（decode 专用）—— 不解压 K，直接在 latent 空间点积
    # ────────────────────────────────────────────────────────────
    def forward_absorbed(self, x, cos, sin, cache):
        """
        推导：
          score_nope = q_nope · k_nope^T
                     = (x·W_UQ) · (c_kv·W_UK)^T
                     = x·W_UQ · W_UK^T · c_kv^T
                       └────────┬────────┘
                          可以预先合并 → q 直接变成 latent 空间的 512 维向量
          ⟹ 完全不需要把 c_kv 解压成 K！每步省下 [S,512]×[512,H*128] 的 GEMM
        输出侧同理：V 也不解压，先在 latent 空间加权求和，最后一次性乘 W_UV。
        """
        B, S, _ = x.shape
        assert S == 1, "吸收路径用于 decode"
        H, nope, rope, vh, r = self.n_head, self.qk_nope, self.qk_rope, self.v_head, self.kv_lora

        q = self.q_up(self.q_norm(self.q_down(x))).view(B, S, H, self.qk_head).transpose(1, 2)
        q_nope, q_rope = q.split([nope, rope], -1)

        kv = self.kv_down(x)
        c_new, k_rope_new = kv.split([r, rope], -1)
        c_new = self.kv_norm(c_new)

        past = cache[0].shape[1]
        q_rope = apply_rope(q_rope, cos[past:past + 1], sin[past:past + 1])
        k_rope_new = apply_rope(k_rope_new.unsqueeze(1), cos[past:past + 1], sin[past:past + 1])

        c_kv = torch.cat([cache[0], c_new], 1)              # [B, S_kv, r]
        k_rope = torch.cat([cache[1], k_rope_new], 2)       # [B, 1, S_kv, rope]
        new_cache = (c_kv, k_rope)

        # ★ 把解压矩阵拆出来：W_UK [H, nope, r]，W_UV [H, r, vh]
        W = self.kv_up.weight.view(H, nope + vh, r)
        W_UK, W_UV = W[:, :nope, :], W[:, nope:, :].transpose(1, 2)

        # ★ 吸收 1：q_nope [B,H,1,nope] × W_UK [H,nope,r] → q_latent [B,H,1,r]
        q_latent = torch.einsum('bhsn,hnr->bhsr', q_nope, W_UK)

        # 在 latent 空间直接点积（不解压 K！）
        s_nope = torch.einsum('bhsr,bcr->bhsc', q_latent, c_kv)
        s_rope = torch.einsum('bhsd,bxcd->bhsc', q_rope, k_rope)   # rope 全头共享
        attn = ((s_nope + s_rope) * self.scale).softmax(-1)

        # ★ 吸收 2：先在 latent 空间加权求和，再一次性解压到 v
        ctx_latent = torch.einsum('bhsc,bcr->bhsr', attn, c_kv)     # [B,H,1,r]
        out = torch.einsum('bhsr,hrv->bhsv', ctx_latent, W_UV)
        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1)), new_cache

# ============================================================
# Part C · 线性注意力 / Gated DeltaNet
# ============================================================

"""
03 · 线性注意力 / Gated DeltaNet —— Qwen3-Next & Qwen3.5 (45/60 层用它)
=======================================================================
核心思想：把 softmax(QK^T)V 的 O(s²) 换成一个**定长状态** S 的递归更新，O(s)。

三种形式，数学上等价，用途不同：
  ① recurrent   —— O(1)/step，decode 用
  ② parallel    —— O(s²)，教学/验证用（不实用）
  ③ chunked     —— O(s·C)，prefill 用（★ 这是真实实现，兼顾并行度和复杂度）

★★ 对 Infra 最重要的结论：
   线性注意力**没有 KV cache**，只有一个 [d_k, d_v] 的定长状态。
   ⟹ 显存不随序列增长，但 **prefix cache 无法按 token 前缀截断复用**（见文末分析）

对应文档：§1.1 Qwen3.5 架构、§4.4 异构 KV 管理、§7.4 prefix cache 失效

"""
import torch
import torch.nn.functional as F

def linear_attn_recurrent(q, k, v):
    """q,k: [B,H,S,dk]  v: [B,H,S,dv] → o: [B,H,S,dv]"""
    B, H, S, dk = q.shape
    dv = v.shape[-1]
    state = torch.zeros(B, H, dk, dv, dtype=q.dtype)
    outs = []
    for t in range(S):
        # ★ 状态更新：外积累加。注意这里没有 softmax，所以可以结合律重排
        state = state + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        outs.append(torch.einsum('bhd,bhdv->bhv', q[:, :, t], state))
    return torch.stack(outs, 2), state

def linear_attn_parallel(q, k, v):
    """等价的 O(s²) 形式：causal mask 下的 (QK^T)V。用于验证递归形式正确。"""
    S = q.shape[2]
    mask = torch.ones(S, S, dtype=torch.bool).tril()
    return ((q @ k.transpose(-1, -2)) * mask) @ v

def gated_deltanet_recurrent(q, k, v, alpha, beta):
    """
    S_t = α_t · S_{t-1} + β_t · k_t (v_t − k_t^T S_{t-1})^T
          └──┬──┘        └────────────┬────────────┘
          门控遗忘              delta rule：只写入"预测残差"

    直觉：
      · α (gate)  控制"忘掉多少历史" —— 这是 Mamba2/GLA 的核心，让状态不会无限累积
      · delta rule 先用当前 state 预测 v，只把**预测错的部分**写进去 —— 避免重复写入
        （对比基础线性注意力是无脑 S += k vᵀ，同样的 key 反复出现会让状态爆炸）

    alpha: [B,H,S]  遗忘门 ∈(0,1)
    beta : [B,H,S]  写入强度
    """
    B, H, S, dk = q.shape
    dv = v.shape[-1]
    state = torch.zeros(B, H, dk, dv, dtype=q.dtype)
    outs = []
    for t in range(S):
        kt, vt = k[:, :, t], v[:, :, t]                       # [B,H,dk], [B,H,dv]
        # ★ 用旧状态预测当前 v
        v_pred = torch.einsum('bhd,bhdv->bhv', kt, state)
        delta = vt - v_pred                                    # 预测残差
        state = alpha[:, :, t, None, None] * state \
                + beta[:, :, t, None, None] * kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append(torch.einsum('bhd,bhdv->bhv', q[:, :, t], state))
    return torch.stack(outs, 2), state

def gated_deltanet_chunked(q, k, v, alpha, beta, chunk=4):
    """
    ★★ 真实实现用的形式：块内并行（矩阵运算），块间递归。
    把 S 个 token 分成 S/C 个 chunk：
      · chunk 内部用矩阵运算一次算完（吃满 Tensor Core）
      · chunk 之间传递 [dk, dv] 的状态（串行，但只有 S/C 步）
    复杂度 O(s·C·d)，并行度 O(C)。C 通常取 64/128。

    这里实现成"块间递归 + 块内逐步"，用于验证等价性；
    生产实现（如 fla 库）会把块内也写成矩阵形式（WY representation）。
    """
    B, H, S, dk = q.shape
    dv = v.shape[-1]
    state = torch.zeros(B, H, dk, dv, dtype=q.dtype)
    outs = []
    for c0 in range(0, S, chunk):
        c1 = min(c0 + chunk, S)
        for t in range(c0, c1):
            kt, vt = k[:, :, t], v[:, :, t]
            v_pred = torch.einsum('bhd,bhdv->bhv', kt, state)
            state = alpha[:, :, t, None, None] * state \
                    + beta[:, :, t, None, None] * kt.unsqueeze(-1) * (vt - v_pred).unsqueeze(-2)
            outs.append(torch.einsum('bhd,bhdv->bhv', q[:, :, t], state))
    return torch.stack(outs, 2), state

def qwen35_layer_layout(n_layer=60, pattern=(0, 0, 0, 1)):
    """0=Gated DeltaNet(线性), 1=Gated Attention(全注意力)"""
    return [pattern[i % len(pattern)] for i in range(n_layer)]

# ============================================================
# Part D · Paged + RadixTree + HiCache + 异构 KV
# ============================================================

"""
04 · HiCache —— 分层 KV Cache（SGLang HiRadixCache 的简化实现）
================================================================
四层递进：
  A. PagedAttention  —— 分块显存管理，消除碎片
  B. RadixTree       —— 前缀树，跨请求共享 KV（RadixAttention）
  C. HiCache         —— GPU → CPU → Disk 三级分层，冷数据下沉
  D. 异构 KV         —— ★ 2026 的新难题：一个模型里多种注意力类型共存

对应文档：§4.4 异构 KV 管理、§2.5 KV 池容量、§7.4 RL 的 prefix cache

"""
from collections import OrderedDict
import time

PAGE = 16          # 每个 block 装多少个 token（vLLM 默认 16）

class PagedKVPool:
    """
    核心思想：逻辑上连续的 KV 序列，物理上散落在不同 block —— 靠 block_table 索引。
    好处：① 无外部碎片 ② 不同请求可以共享同一个 block（这是 prefix cache 的物理基础）
    """
    def __init__(self, n_blocks, bytes_per_block):
        self.n_blocks = n_blocks
        self.bpb = bytes_per_block
        self.free = list(range(n_blocks))
        self.refcount = [0] * n_blocks          # ★ 引用计数：共享 block 的关键

    def alloc(self, n):
        if len(self.free) < n:
            return None                          # OOM → 调用方要触发抢占/换出
        blocks = [self.free.pop() for _ in range(n)]
        for b in blocks:
            self.refcount[b] = 1
        return blocks

    def incref(self, blocks):
        for b in blocks:
            self.refcount[b] += 1

    def decref(self, blocks):
        for b in blocks:
            self.refcount[b] -= 1
            if self.refcount[b] == 0:
                self.free.append(b)              # 只有引用归零才真正释放

    @property
    def used(self):
        return self.n_blocks - len(self.free)

    @property
    def usage(self):
        return self.used / self.n_blocks

class RadixNode:
    """
    为什么用 Radix Tree（压缩前缀树），而不是朴素 Trie / 全序列 Hash：
      · 跨请求共享 KV：多条 prompt 常共享 system / few-shot / 对话前缀；树边存的
        `blocks` 指向 PagedKVPool 里同一批物理 block，靠 refcount 共享，避免重复 prefill。
      · 相对「每 token 一节点」的 Trie：无分叉的路径压成一条边上的 `tokens[]`，
        节点数 ≈ 分叉点个数，深度与内存都更省（Radix = Patricia / compressed trie）。
      · 相对「整段 token 序列 → KV」的 HashMap：只能 exact hit；Radix 支持最长前缀
        匹配，并在分叉处 split 节点，新请求只需补算未命中后缀。
      · 与 PagedAttention 配套：逻辑前缀 ↔ 物理 block 列表一一对应；父子边天然表达
        「父 KV 是子 KV 的前缀」→ 驱逐必须自叶子向上（见 evict_lru）。

    节点字段：
      children[first_token] → 子边；tokens / blocks 是该边上的标签与对应 KV block id。

    整体算法与复杂度（令 L=本次序列长，N=树节点数，U=树上已存的唯一 token 总量）：
      match / insert：沿路径逐 token 比较，每 token 至多看一次 → 时间 O(L)；
        insert 遇分叉最多 split 一次，切边 O(边长) ⊆ O(L)。
      空间：每条边的 tokens/blocks 各存一份，共享前缀只占一份 → O(U)；
        节点数 O(分叉点) ≪ 朴素 Trie 的 O(U)。
      evict_lru：收集叶子 O(N) + 按 last_access 排序 O(N log N) + 释放 O(要腾的 block 数)；
        正确性约束：只能删叶子（父边仍被子请求引用）。
      命中收益：命中前缀长度 M 时跳过 M token 的 prefill（算力/带宽），代价是树维护
        与一次 O(L) 查找 —— 典型 serving 下远小于重算注意力。
    """
    __slots__ = ('children', 'tokens', 'blocks', 'parent', 'last_access', 'lock')

    def __init__(self, parent=None):
        self.children = {}          # first_token -> RadixNode
        self.tokens = []            # 这条边上的 token 序列（压缩路径，非单 token）
        self.blocks = []            # 对应的 KV block id（与 tokens 按 PAGE 对齐）
        self.parent = parent
        self.last_access = time.time()
        self.lock = 0               # ★ 正在被 running 请求使用 → 不可驱逐

class RadixTree:
    """
    前缀匹配 + 插入分裂 + LRU 叶子驱逐。算法细节见各方法；复杂度总览见 RadixNode。
    """
    def __init__(self, pool: PagedKVPool):
        self.root = RadixNode()
        self.pool = pool

    @staticmethod
    def _common(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def match(self, tokens):
        """最长前缀匹配。返回 (命中长度, 命中的 block 列表, 停在哪个节点)。O(L)。"""
        node, idx, blocks = self.root, 0, []
        while idx < len(tokens):
            ch = node.children.get(tokens[idx])
            if ch is None:
                break
            c = self._common(ch.tokens, tokens[idx:])
            blocks.extend(ch.blocks[:c // PAGE])
            if c < len(ch.tokens):               # 部分匹配 → 停在边中间
                idx += c
                node = ch
                break
            idx += c
            node = ch
        return idx, blocks, node

    def insert(self, tokens, blocks):
        """插入完整序列；已有前缀复用（物理块靠 pool.incref）。最坏 O(L)，含一次 split。"""
        node, idx = self.root, 0
        while idx < len(tokens):
            first = tokens[idx]
            ch = node.children.get(first)
            if ch is None:
                new = RadixNode(node)
                new.tokens = tokens[idx:]
                new.blocks = blocks[idx // PAGE:]
                node.children[first] = new
                return
            c = self._common(ch.tokens, tokens[idx:])
            if c < len(ch.tokens):
                # ★ 分裂节点：把已有边拆成 [公共部分] → [剩余部分]
                mid = RadixNode(node)
                mid.tokens, mid.blocks = ch.tokens[:c], ch.blocks[:c // PAGE]
                ch.tokens, ch.blocks = ch.tokens[c:], ch.blocks[c // PAGE:]
                ch.parent = mid
                mid.children[ch.tokens[0]] = ch
                node.children[first] = mid
                node = mid
            else:
                node = ch
            idx += c
        return

    def evict_lru(self, n_blocks):
        """★ LRU 驱逐：只能从叶子开始（父节点的 KV 是子节点的前缀，不能先删）。O(N log N)。"""
        leaves = []

        def collect(n):
            if not n.children and n is not self.root:
                leaves.append(n)
            for c in n.children.values():
                collect(c)
        collect(self.root)
        leaves.sort(key=lambda x: x.last_access)

        freed = 0
        for leaf in leaves:
            if freed >= n_blocks:
                break
            if leaf.lock > 0:                     # 正在用，跳过
                continue
            self.pool.decref(leaf.blocks)
            freed += len(leaf.blocks)
            if leaf.parent:
                leaf.parent.children.pop(leaf.tokens[0], None)
        return freed

class HiCache:
    """
    为什么要分层（对应文档 §3.5.3 的"长上下文死亡螺旋"）：
      GPU 显存装不下大 batch × 长上下文的 KV → batch 上不去 → MoE 算力浪费
      而重算 prefill 的代价远大于从 CPU/NVMe 读回来 —— 见 ③ 的盈亏平衡分析
    """
    LEVELS = [
        # name,  容量(GB), 带宽(GB/s),      延迟(us)
        ('GPU',      80,   3400,             0),
        ('CPU',     512,     25,            10),    # PCIe Gen4，pinned memory
        ('NVMe',  10000,      6,           100),
    ]

    def __init__(self):
        self.store = [OrderedDict() for _ in self.LEVELS]     # 每层一个 LRU
        self.used = [0.0] * len(self.LEVELS)
        self.stats = dict(hit=[0] * len(self.LEVELS), miss=0, promoted=0, evicted=0)

    def _cap(self, i):
        return self.LEVELS[i][1]

    def get(self, key, size_gb):
        """★ 逐级查找；命中后提升到 GPU（返回本次的传输耗时 ms）"""
        for i, lvl in enumerate(self.store):
            if key in lvl:
                lvl.move_to_end(key)
                self.stats['hit'][i] += 1
                if i == 0:
                    return 0.0
                # 从第 i 层搬到 GPU
                _, _, bw, lat = self.LEVELS[i]
                ms = lat / 1000 + size_gb * 1024 / bw
                self.put(key, size_gb, level=0)
                self.stats['promoted'] += 1
                return ms
        self.stats['miss'] += 1
        return None

    def put(self, key, size_gb, level=0):
        # 容量不够 → 向下一层驱逐（而不是直接丢弃）
        while self.used[level] + size_gb > self._cap(level) and self.store[level]:
            k, s = self.store[level].popitem(last=False)      # LRU
            self.used[level] -= s
            self.stats['evicted'] += 1
            if level + 1 < len(self.LEVELS):
                self.put(k, s, level + 1)                     # ★ 下沉，不是丢弃
        self.store[level][key] = size_gb
        self.used[level] += size_gb

class HeterogeneousKVLayout:
    """
    PagedAttention 的原始假设：「每层每 token 的 KV 大小相同」
    → DS-V4 打破了它：CSA 层每 4 token 压 1 个 entry，HCA 层每 128 token 压 1 个
    → 同一个"逻辑 block"在不同层的物理大小可以差 32 倍
    """
    def __init__(self, n_layer=61, c=512, idx_dim=128):
        self.layers = []
        for i in range(n_layer):
            if i < 2:
                self.layers.append(('HCA', 128))
            else:
                self.layers.append(('CSA', 4) if i % 2 == 0 else ('HCA', 128))
        self.c, self.idx = c, idx_dim

    def bytes_per_token(self, dtype=2):
        tot = 0
        for kind, m in self.layers:
            per_entry = (self.c + self.idx) if kind == 'CSA' else self.c
            tot += per_entry * dtype / m          # ★ 压缩率 m 直接摊薄
        return tot

    def block_bytes(self, layer_i, n_token=PAGE, dtype=2):
        kind, m = self.layers[layer_i]
        per_entry = (self.c + self.idx) if kind == 'CSA' else self.c
        # ⌈n/m⌉ 个 entry —— ★ 注意这里的向上取整就是"压缩边界不对齐"问题的根源
        return -(-n_token // m) * per_entry * dtype

# ============================================================
# Part E · EAGLE 投机解码
# ============================================================

"""
05 · EAGLE 投机解码 —— 满足 10ms TPOT SLO 的必要条件
=====================================================
对应文档 §3.5.1：8×H20 上权重读取的物理下界就是 12~25 ms，
                 想做到 10ms TPOT，MTP/投机解码是**必要条件**，不是可选优化。

EAGLE 的三个关键设计（相比 vanilla 投机解码）：
  ① 在 **feature 空间**（最后一层 hidden state）而非 token 空间做自回归
     → 草稿模型只需 1 层 decoder，参数量 <1% 的主模型
  ② 输入 = (shifted feature, token embedding) 拼接
     → 解决"采样的随机性无法从 feature 推断"的问题
  ③ ★ 树形草稿 + tree attention
     → 一次验证多条候选路径，接受长度 τ 显著高于链式

★ 为什么投机解码在 decode 阶段"几乎免费"：
  decode 是访存瓶颈（读 100~800GB 权重），验证 k 个 token 和验证 1 个 token
  读的是**同一份权重** → 算力增加但访存不变 → 在 memory-bound 区间近乎白赚。

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EagleDraftHead(nn.Module):
    """
    输入：上一步的 hidden feature f_{t}（来自主模型最后一层）+ 已采样 token 的 embedding e_{t+1}
    输出：预测的下一个 feature f_{t+1}，再经主模型的 lm_head 得到 logits

    ★ 关键：f 和 e 拼起来降维，因为单看 f 无法知道实际采样出了哪个 token（采样有随机性）
    """
    def __init__(self, d_model, vocab):
        super().__init__()
        self.fc = nn.Linear(2 * d_model, d_model, bias=False)   # (feature, embedding) → d
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=2 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)

    def forward(self, feats, embs):
        """feats/embs: [B, S, d] → 预测的下一步 feature [B, S, d]"""
        x = self.fc(torch.cat([feats, embs], -1))
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        return self.layer(x, src_mask=mask)

class DraftTree:
    """
    链式草稿：draft 5 个 token，一旦第 2 个被拒，后面 3 个全废。
    树形草稿：每层保留 top-k 个分支 → 一次验证覆盖多条路径，接受长度显著提高。

    节点用 (parent_idx, token, cum_logp) 表示，展平成一维数组给 tree attention 用。
    """
    def __init__(self):
        self.parent = [-1]          # 根 = 当前已确定的最后一个 token
        self.token = [None]
        self.depth = [0]
        self.logp = [0.0]

    def add(self, parent_idx, token, logp):
        self.parent.append(parent_idx)
        self.token.append(token)
        self.depth.append(self.depth[parent_idx] + 1)
        self.logp.append(self.logp[parent_idx] + logp)
        return len(self.parent) - 1

    def n_nodes(self):
        return len(self.parent)

    def attention_mask(self):
        """
        ★ Tree Attention Mask：节点 i 只能看到它的祖先链。
        这是树形草稿能一次 forward 验证所有分支的关键 ——
        兄弟节点之间互相不可见，等价于并行跑了多条独立序列。
        """
        n = self.n_nodes()
        m = torch.zeros(n, n, dtype=torch.bool)
        for i in range(n):
            j = i
            while j != -1:
                m[i, j] = True
                j = self.parent[j]
        return m

    def path_to(self, idx):
        path = []
        while idx > 0:
            path.append(self.token[idx])
            idx = self.parent[idx]
        return path[::-1]

def build_tree(draft_probs_fn, root_ctx, depth=3, topk=2, max_nodes=16):
    """
    逐层扩展：每层对当前叶子取 top-k 子节点，按累积概率排序保留 max_nodes 个。
    draft_probs_fn(path) → [vocab] 的概率分布
    """
    tree = DraftTree()
    frontier = [0]
    for _ in range(depth):
        cand = []
        for node in frontier:
            probs = draft_probs_fn(root_ctx + tree.path_to(node))
            top = torch.topk(probs, topk)
            for p, t in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((tree.logp[node] + torch.log(torch.tensor(p + 1e-9)).item(),
                             node, t))
        cand.sort(reverse=True)
        new_frontier = []
        for lp, parent, tok in cand:
            if tree.n_nodes() >= max_nodes:
                break
            idx = tree.add(parent, tok, lp - tree.logp[parent])
            new_frontier.append(idx)
        frontier = new_frontier
        if not frontier:
            break
    return tree

def speculative_accept(p_target, q_draft, token, rng=None):
    """
    标准 speculative sampling 接受准则（Leviathan et al. / Chen et al.）：
      以概率 min(1, p(x)/q(x)) 接受；拒绝时从 norm(max(0, p−q)) 重采样。
    ★ 这保证了最终输出分布**严格等于**目标模型的分布 —— 投机解码是无损的。
    """
    p, q = p_target[token].item(), q_draft[token].item()
    r = torch.rand(1).item() if rng is None else rng()
    if r < min(1.0, p / (q + 1e-10)):
        return True, token
    resid = torch.clamp(p_target - q_draft, min=0)
    resid = resid / resid.sum().clamp(min=1e-10)
    return False, torch.multinomial(resid, 1).item()

def verify_tree(tree, target_probs_fn, draft_probs_fn, root_ctx):
    """沿树做深度优先验证，返回被接受的 token 序列。"""
    accepted, node = [], 0
    while True:
        children = [i for i, p in enumerate(tree.parent) if p == node]
        if not children:
            break
        ctx = root_ctx + accepted
        p_t = target_probs_fn(ctx)
        matched = False
        for ch in children:
            q_d = draft_probs_fn(ctx)
            ok, tok = speculative_accept(p_t, q_d, tree.token[ch])
            if ok:
                accepted.append(tok)
                node = ch
                matched = True
                break
        if not matched:
            # 全部子节点被拒 → 用修正分布补一个 token，然后停止
            _, tok = speculative_accept(p_t, draft_probs_fn(ctx), tree.token[children[0]])
            accepted.append(tok)
            break
    return accepted

def speedup(tau, c_draft, n_draft_calls):
    """
    tau          : 平均接受长度（每次验证确定几个 token）
    c_draft      : 单次草稿前向 / 单次目标前向 的耗时比（EAGLE 约 0.01~0.05）
    n_draft_calls: 一轮里草稿模型被调用几次（= 树深度）
    """
    return tau / (1 + c_draft * n_draft_calls)

# ============================================================
# Part F · 集合通信 (Mesh / Ring / 分层 AR / EP all2all)
# ============================================================

HCCS_LINK = 56e9      # 点对点 56 GB/s
HCCS_PEERS = 7        # full-mesh: 每卡 7 条直连链路
A_HCCS = 2e-6         # 机内固定延迟
ROCE = 25e9           # 200G RoCE 实际 ~25 GB/s
A_NET = 10e-6         # 跨机固定延迟

def ar_mesh(S, P=8, bw=HCCS_LINK, a=A_HCCS):
    """full-mesh: RS 1步 + AG 1步，P-1 条链路并行，每条只过 S/P"""
    return 2*a + 2*(S/P)/bw
def ar_ring(S, P=8, bw=HCCS_LINK, a=A_HCCS):
    """ring: 2(P-1) 步，★ 每步只用 1 条链路"""
    return 2*(P-1)*a + 2*(P-1)*(S/P)/bw
def ar_hd(S, P=8, bw=HCCS_LINK, a=A_HCCS):
    """Halving-Doubling: 2log2(P) 步，但每步也只用 1 条链路"""
    import math
    return 2*math.log2(P)*a + 2*(P-1)/P*S/bw

S_dec = 128*7168*2

def hier_ar(S, nnode=2, P=8):
    t1 = A_HCCS + (S/P)/HCCS_LINK                      # ①机内RS: 1步,瘦身到1/8
    shard = S/P
    t2 = 2*A_NET + 2*(shard/nnode)/ROCE                # ②跨机AR: 8张NIC并行,每张只过S/8
    t3 = A_HCCS + (S/P)/HCCS_LINK                      # ③机内AG
    return t1+t2+t3, t1, t2, t3
def flat_ring16(S, P=16):
    # 16卡环有2段跨机 → 跨机链路成瓶颈, α取跨机
    return 2*(P-1)*A_NET + 2*(P-1)*(S/P)/ROCE

d,k,B = 7168,8,128
S_a2a = B*k*d*1  # FP8
def a2a_intra(S,P=8):   # 机内 full-mesh: 一跳, P-1条链路并行
    return A_HCCS + (S*(P-1)/P)/(HCCS_LINK*(P-1))
def a2a_cross(S,nnode): # 跨机: (nnode-1)/nnode 的流量过 1 张 NIC
    frac = (nnode-1)/nnode
    return A_NET + (S*frac)/ROCE
t8 = a2a_intra(S_a2a); print(f"  {'机内 EP8 (HCCS)':<26}{t8*1e6:>8.1f}us{122*t8*1000:>9.2f}ms{122*t8*1000/30*100:>8.1f}%  ✅")

# ============================================================
# Part G · torch.compile × CUDA Graph（语义 mock）
# ============================================================

"""
07 · torch.compile × CUDA Graph × eager —— 三层执行栈的职责与冲突
==================================================================
本文件用 mock 把三层的【语义】跑出来（无 GPU 也能运行），重点讲清楚：
  ① 三层各消灭什么开销 —— 用计数器可观测
  ② 为什么能叠加 —— compile 融合后的核照样被 graph capture
  ③ ★ reduce-overhead + PagedAttention 为什么产生【静默错误】—— 真实触发一次
  ④ Omni 三 stage 各自的 capture 点（对应决策图）

真实工程里对应的 API：
  CUDA Graph : torch.cuda.CUDAGraph() / graph.capture_begin/end / graph.replay()
               或 vLLM 的 CUDAGraphWrapper、SGLang 的 CudaGraphRunner
  compile    : torch.compile(fn, fullgraph=False, mode=None)  ← 注意 mode 不设 reduce-overhead
  桶         : 预枚举 (batch, seq, frames...) 一组 shape，各 capture 一张 graph

"""
from dataclasses import dataclass, field

@dataclass
class Profiler:
    cpu_launches: int = 0        # 每次 cudaLaunchKernel 的 CPU 开销（~5-20us）
    gpu_kernels: int = 0         # GPU 上实际跑的 kernel 数
    replays: int = 0             # graph replay 次数
    fallbacks: int = 0           # 桶 miss 回退 eager 次数

    def launch(self, n=1):
        self.cpu_launches += n
        self.gpu_kernels += n

    def report(self, tag):
        print(f"  {tag:<32} CPU下发={self.cpu_launches:>5}  "
              f"GPU核={self.gpu_kernels:>5}  replay={self.replays}  fallback={self.fallbacks}")

def raw_forward(prof, n_kernels, shape):
    """eager 模式：每个算子单独下发一次。shape 只是用来演示动态性。"""
    prof.launch(n_kernels)
    return f"out(shape={shape})"

class Compiled:
    """
    模拟 torch.compile(fullgraph=False)：把 n 个小 kernel 融合成 ~ceil(n*ratio) 个。
    改变的是"要下发什么"。融合后 launch 和 gpu kernel 都减少。

    ★ VoxCPM2 对照：per-layer compile（fullgraph=True，56 个 region）launch 数不降,
      因为 Dynamo 跨 region 边界无法融合；整个 forward 包一层 fullgraph=False 才有效。
    """
    def __init__(self, fn, n_kernels, fuse_ratio=0.3, scope='fullgraph'):
        self.fn, self.n_kernels = fn, n_kernels
        self.fuse_ratio = fuse_ratio
        self.scope = scope

    def __call__(self, prof, shape):
        if self.scope == 'per_layer':
            # ❌ 每个 region 单独编译：跨边界无法优化，融合不了 → launch 不降
            fused = self.n_kernels
        else:
            # ✅ 整个 forward 一个 graph：融合 elementwise/norm 进 GEMM epilogue
            fused = max(1, round(self.n_kernels * self.fuse_ratio))
        prof.launch(fused)
        return f"out(shape={shape})"

class CUDAGraphWrapper:
    """
    模拟 graph capture + replay + 分桶。
    改变的是"怎么下发"：capture 一次记录整段 launch 序列，replay 只付 1 次 launch。
    ★ 要求 shape 静态、地址固定 → 用"桶"枚举一组 shape，各 capture 一张。
    """
    def __init__(self, inner, buckets):
        self.inner = inner              # 可以是 raw_forward，也可以是 Compiled（★ 叠加点）
        self.buckets = sorted(buckets)  # 预枚举的 shape 桶
        self.captured = {}              # bucket -> "已录制"
        self.slot_mapping_baked = {}    # ★ 演示静默错误：capture 时烧死的 KV 槽位

    def _pick_bucket(self, shape):
        for b in self.buckets:
            if shape <= b:
                return b
        return None

    def capture(self, prof, bucket, slot_mapping):
        """warmup 阶段：对每个桶录制一次（这里的 launch 是 capture 开销，一次性）"""
        # capture 时会把当前的指针/slot_mapping 烧进 graph
        self.slot_mapping_baked[bucket] = slot_mapping
        self.captured[bucket] = True

    def __call__(self, prof, shape, slot_mapping):
        b = self._pick_bucket(shape)
        if b is None or b not in self.captured:
            # ★ 桶 miss → 回退 eager（真实系统 stream_capture_fallbacks 统计这个）
            prof.fallbacks += 1
            return self.inner(prof, shape) if not isinstance(self.inner, CUDAGraphWrapper) \
                   else self.inner(prof, shape, slot_mapping)
        # ✅ 命中：replay 一次，只付 1 次 launch（无论 graph 里有多少 kernel）
        prof.launch(1)
        prof.replays += 1
        # ★★ 关键：replay 用的是 capture 时烧死的 slot_mapping，不是当前的！
        baked = self.slot_mapping_baked[b]
        return f"out(shape={shape}, wrote_to_KV_slot={baked})"

class CompiledReduceOverhead:
    """
    模拟 torch.compile(mode='reduce-overhead')：它会【自动】在内部开 CUDA Graph。
    问题：capture 时把 slot_mapping 烧死，但 decode 每步 KV 槽位是变化的
         → replay 把 attention 写到【错误的 KV 位置】→ 错误的 stop logits
         → ★ 静默错误，不崩溃，只是输出慢慢变坏。
    """
    def __init__(self, n_kernels):
        self.n_kernels = n_kernels
        self.baked_slot = None

    def __call__(self, prof, shape, real_slot_mapping):
        if self.baked_slot is None:
            # 第一次调用 = 自动 capture，烧死当时的 slot_mapping
            self.baked_slot = real_slot_mapping
        prof.launch(1)
        prof.replays += 1
        # ★ 永远写到烧死的槽位，无视真实 slot → 静默错误
        return real_slot_mapping, self.baked_slot

# ============================================================
# Part H · vLLM/SGLang 风格执行栈骨架
# ============================================================

"""
08 · 真实 API 骨架（vLLM / SGLang 风格）—— 对照生产代码读
=========================================================
上一个文件（07）用 mock 讲语义；这个文件是照着真实框架的写法给骨架，
标注了每一步在 vLLM/SGLang 里对应什么。★ 无 GPU 不能真跑，仅供对照阅读。

三层的真实 API：
  · CUDA Graph : torch.cuda.CUDAGraph() + graph.capture_begin/end + graph.replay()
  · compile    : torch.compile(fn, fullgraph=False)   ← mode 留空，别用 reduce-overhead
  · 桶          : 预枚举 batch sizes，各 capture 一张，静态输入 buffer 复用
"""
import torch

class DecodeCudaGraphRunner:
    def __init__(self, model, buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                 hidden=7168, max_bs=256, dtype=torch.float16, device='cuda'):
        self.model = model
        self.buckets = sorted(buckets)
        self.graphs = {}                     # bucket -> torch.cuda.CUDAGraph
        self.io = {}                         # bucket -> (static_in, static_out, static_slot)

        # ★ 关键点 1：预分配【固定地址】的静态输入/输出 buffer，所有桶复用同一块的切片
        #    graph 里烧死的是这些 buffer 的地址，replay 前把真实数据 copy_ 进来
        self.input_buf = torch.zeros(max_bs, hidden, dtype=dtype, device=device)
        self.output_buf = torch.zeros(max_bs, hidden, dtype=dtype, device=device)
        self.slot_buf = torch.zeros(max_bs, dtype=torch.int32, device=device)

    def capture(self):
        """warmup 阶段，对每个桶录制一次。真实系统会先 precompute 各种 cache。"""
        for bs in self.buckets:
            in_slice = self.input_buf[:bs]
            slot_slice = self.slot_buf[:bs]

            # 先跑几次 eager warmup（让 cudnn/算子选择稳定），否则 capture 到未优化路径
            for _ in range(3):
                _ = self.model(in_slice, slot_mapping=slot_slice)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            # ★ 关键点 2：capture 期间所有 kernel 的 launch 序列被录进 g
            with torch.cuda.graph(g):
                out = self.model(in_slice, slot_mapping=slot_slice)
                self.output_buf[:bs].copy_(out)
            self.graphs[bs] = g
            self.io[bs] = (in_slice, self.output_buf[:bs], slot_slice)

    def _pick(self, bs):
        for b in self.buckets:
            if bs <= b:
                return b
        return None

    def __call__(self, hidden_states, slot_mapping):
        bs = hidden_states.shape[0]
        b = self._pick(bs)
        if b is None:                        # 超过最大桶 → 回退 eager
            return self.model(hidden_states, slot_mapping=slot_mapping)

        static_in, static_out, static_slot = self.io[b]
        # ★ 关键点 3：replay 前把【当前】数据/slot_mapping copy 进静态 buffer
        #    ——正因为可以在这里更新 slot_mapping，手动 graph 不会写错 KV
        #    （对比 reduce-overhead 自动 graph 没有这一步 → 静默写错，见 07）
        static_in[:bs].copy_(hidden_states)
        static_slot[:bs].copy_(slot_mapping)
        if bs < b:                           # padding 到桶大小
            static_in[bs:b].zero_()

        self.graphs[b].replay()              # ★ 关键点 4：一次下发，付 1 次 launch
        return static_out[:bs].clone()

def build_code_predictor(raw_predictor):
    """
    ★ 关键点 5：整个 forward 包【一层】compile，fullgraph=False。
       不用 mode='reduce-overhead'（它会自动开 graph 撞外层 wrapper + PagedAttention）。
       不用 per-layer compile（跨 region 边界 Dynamo 无法融合，launch 不降）。
    """
    compiled = torch.compile(
        raw_predictor,
        fullgraph=False,        # 允许 graph break，别强制单图
        dynamic=False,          # 码本预测 shape 固定（2-8 token），关闭动态 → 更激进融合
        mode=None,              # ★ 绝不 'reduce-overhead'
        # options={'epilogue_fusion': False}  # 视情况关 epilogue fusion（VoxCPM2 经验）
    )
    # ★ 关键点 6：显式声明这层不走第二层 CUDA Graph，避免和外层 CUDAGraphWrapper 抢 capture
    compiled._use_cuda_graphs = False
    return compiled

class Code2WavGraphWrapper:
    def __init__(self, decoder, frame_buckets=(8, 16, 32, 64, 128),
                 initial_chunk=8):
        self.decoder = decoder
        # ★ 首包用更小的桶（首包帧数少，单独 capture 降 TTFP）
        self.frame_buckets = sorted(set(frame_buckets) | {initial_chunk})
        self.graphs = {}

    def warmup(self):
        # ★ 关键点 7：warmup 先 precompute_snake_caches（声码器的 anti-alias 缓存），
        #    否则 capture 会把 cache 计算也录进去，或 replay 时 cache 缺失
        self.decoder.precompute_snake_caches()
        for f in self.frame_buckets:
            dummy = torch.zeros(1, f, device='cuda')   # (batch, frames)
            for _ in range(2):
                _ = self.decoder(dummy)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _ = self.decoder(dummy)
            self.graphs[f] = g

    def __call__(self, codes, n_frames):
        b = next((x for x in self.frame_buckets if n_frames <= x), None)
        if b is None:
            return self.decoder(codes)       # miss → eager，统计 fallback
        self.graphs[b].replay()
        return "audio_chunk"

class OmniEngine:
    """
    职责边界一目了然：
      Thinker  → DecodeCudaGraphRunner（外层 graph 管 AR 循环的 launch）
      Talker   → DecodeCudaGraphRunner（外层） + build_code_predictor（内层 compile micro）
      Code2Wav → Code2WavGraphWrapper（内层 decoder graph，枚举帧桶）
    """
    def __init__(self, thinker, talker, code_predictor_raw, vocoder):
        self.thinker = DecodeCudaGraphRunner(thinker)
        self.talker = DecodeCudaGraphRunner(talker)
        self.code_predictor = build_code_predictor(code_predictor_raw)  # compile, 不开graph
        self.code2wav = Code2WavGraphWrapper(vocoder)

    def warmup(self):
        self.thinker.capture()
        self.talker.capture()
        self.code2wav.warmup()

# ============================================================
# Part I · DeepGEMM masked grouped GEMM
# ============================================================

"""
10 · DeepGEMM masked grouped GEMM —— 逐行伪代码 + 可运行验证
=============================================================
已对照官方仓库核实(deepseek-ai/DeepGEMM):
  ✅ masked 为 CUDA Graph 而生("CPU is unaware of the number of tokens each
     expert receives"——路由在 GPU 上,mask 必须是 GPU tensor)
  ✅ 只沿 M 分组、N/K 固定;contiguous(prefill)/masked(decode)双布局
  ✅ 两级累加(CUDA-core promotion);1x128 LHS / 128x128 RHS scale
  ⚠️ SASS 级 FFMA 交织为初版优化,2025.07 重构后退役(NVCC 12.9 自动交织)

三个验证:
  ① masked 调度与逐专家参考实现数值等价
  ② padding 浪费:BM ∈ {8,16,64,128} 的实测浪费倍数
  ③ 两级累加:为什么 FP8 深 K 必须 promote 到 FP32

运行:python3 10_deepgemm_masked.py
"""
import numpy as np

PSEUDOCODE = r"""
// ═══════════════ Host 侧(每个 decode step)═══════════════
m_grouped_gemm_fp8_masked(
    A[E_loc, M_max, K],          // dispatch 后按专家分槽的激活(FP8)
                                 //   ★ 形状固定 → CUDA Graph capture 时地址/形状都定死
    A_scale[E_loc, M_max, K/128],// 激活 scale:每 token 每 128 个 K 通道一个(1x128 粒度)
    B[E_loc, N, K],              // 本卡 E_loc 个专家的权重(FP8)
    B_scale[E_loc, N/128, K/128],// 权重 scale:128x128 块一个
    D[E_loc, M_max, N],          // 输出(BF16)
    masked_m[E_loc],             // ★★ 每专家本步真实行数 —— GPU tensor:
                                 //    地址固定(graph 满足) + 内容每步 replay 前更新(动态满足)
    expected_m)                  // 期望行数,只用于 JIT 挑 BLOCK_M —— 编译期常量

// ═══════════════ Device 侧(persistent kernel,单 block 视角)═══════════════
__global__ void masked_gemm_kernel(...):
  // 132 个 block 常驻 SM,从全局调度器抢 tile(避免 wave quantization)
  while (tile = scheduler.next()):            // tile = (e, m_blk, n_blk)
    m_size = __ldg(&masked_m[tile.e]);        // ★ 运行时读该专家真实行数
    if (tile.m_blk * BM >= m_size) continue;  // ★ 超出真实行数的块直接跳过
                                              //   —— pad 区一条 MMA 都不发,这就是省算力的地方
    // ---------- 生产者 warpgroup:TMA 异步搬运 ----------
    for (k_blk = 0; k_blk < K/BK; ++k_blk):
      tma_load(A[e, m_blk*BM:, k_blk*BK:]  -> smem_A[buf]);   // 异步,双缓冲
      tma_load(B[e, n_blk*BN:, k_blk*BK:]  -> smem_B[buf]);   // TMA 还支持 multicast
      tma_load(scales                       -> smem_S[buf]);
    // ---------- 消费者 warpgroup:WGMMA + ★ 两级累加 ----------
    float acc[BM][BN] = 0;                    // FP32 主累加器(寄存器)
    for (k_blk ...):
      partial = 0;                            // Tensor Core 内部低精度累加器
      #pragma unroll
      for (w = 0; w < 4; ++w)                 // ★ 只连续累加 4 条 WGMMA = 128 个 K 元素
        partial += wgmma_fp8(smem_A, smem_B); //   (FP8 累加器精度有限,不能一路累到 K=7168)
      acc += float(partial)                   // ★ CUDA core 升位:乘上两个 scale 后
             * a_scale[k_blk] * b_scale[k_blk]; //   进 FP32 —— "两级累加/promotion"
    // ---------- epilogue ----------
    rows = min(BM, m_size - tile.m_blk*BM);   // 尾块可能不满
    store_bf16(D[e, ...], acc, rows);         // 只写有效行
"""

def ceil_div(a, b): return -(-a // b)

E_loc, M_max, K, N = 8, 64, 256, 96
A = np.random.randn(E_loc, M_max, K).astype(np.float32)
B = np.random.randn(E_loc, N, K).astype(np.float32)
masked_m = np.random.poisson(12, E_loc).clip(1, M_max)     # 每专家真实行数(小!)

def reference(A, B, masked_m):
    """参考实现:逐专家、只算真实行"""
    D = np.zeros((E_loc, M_max, N), np.float32)
    for e in range(E_loc):
        m = masked_m[e]
        D[e, :m] = A[e, :m] @ B[e].T
    return D

def masked_kernel_sim(A, B, masked_m, BM):
    """模拟 masked kernel 的块调度:静态 buffer + 动态跳块"""
    D = np.zeros((E_loc, M_max, N), np.float32)
    mma_rows = 0                                   # 统计实际发射的 MMA 行数(含尾块 pad)
    for e in range(E_loc):                         # scheduler 遍历 (e, m_blk)
        m = masked_m[e]                            # ★ 运行时读 masked_m
        for m_blk in range(ceil_div(M_max, BM)):   # 静态 shape 决定的块上限
            if m_blk * BM >= m:                    # ★ 跳过纯 pad 块
                continue
            lo = m_blk * BM
            hi_store = min(lo + BM, m)             # ★ epilogue 只写有效行(≤ masked_m)
            D[e, lo:hi_store] = A[e, lo:hi_store] @ B[e].T
            mma_rows += BM                         # 硬件仍按整 tile 发射(尾块内部有 pad)
    return D, mma_rows

# padding 浪费 = MMA 行数 / 真实行数；小 BM 把 ~10× 压到 ~1.3×，且 buffer 形状固定可 graph

def fp8_e4m3(x, scale):
    """模拟 FP8 e4m3 量化:3 位尾数"""
    v = x / scale
    mag = np.abs(v).clip(1e-9, 448.0)
    e = np.floor(np.log2(mag))
    m = np.round(mag / 2**e * 8) / 8                      # 3-bit 尾数
    return np.sign(v) * m * 2**e

def trunc(x, bits=8):
    """模拟有限精度累加器:尾数只留 bits 位(指数域不限)。
       真实 Hopper FP8 累加路径的有效尾数有限,这里取 8 位夸大演示。"""
    m_, e_ = np.frexp(x)
    return float(np.ldexp(np.round(m_ * 2**bits) / 2**bits, e_))

def two_level_accum_demo(Kd=4096, G=128):
    """对比:一路有限精度累加 vs 块内累 128 再 promote 到 FP32（DeepGEMM 两级累加）"""
    a = np.random.randn(Kd).astype(np.float32) * 0.5
    b = np.random.randn(Kd).astype(np.float32) * 0.5
    a_s = np.array([np.abs(a[g:g+G]).max()/448 for g in range(0, Kd, G)])
    b_s = np.array([np.abs(b[g:g+G]).max()/448 for g in range(0, Kd, G)])
    aq = np.concatenate([fp8_e4m3(a[g:g+G], a_s[i]) for i, g in enumerate(range(0, Kd, G))])
    bq = np.concatenate([fp8_e4m3(b[g:g+G], b_s[i]) for i, g in enumerate(range(0, Kd, G))])
    ref = float(a @ b)
    acc = 0.0
    for i, g in enumerate(range(0, Kd, G)):
        for t in range(g, g+G):
            acc = trunc(acc + trunc(aq[t]*bq[t]) * a_s[i] * b_s[i])
    acc32 = 0.0
    for i, g in enumerate(range(0, Kd, G)):
        partial = 0.0
        for t in range(g, g+G):
            partial = trunc(partial + trunc(aq[t]*bq[t]))
        acc32 += partial * a_s[i] * b_s[i]
    return ref, acc, acc32

# ============================================================
# Part J · DeepEP low-latency dispatch/combine
# ============================================================

"""
11 · DeepEP low-latency dispatch/combine —— 逐行伪代码 + 可运行验证
====================================================================
三个验证:
  ① dispatch→专家计算→combine 的端到端结果 == 单机稠密参考实现
  ② 去重:实测每 token 发往的 rank 数 C < k
  ③ hook 机制的 overlap 时序:send 不占 SM → 通信藏进 shared expert 计算

运行:python3 11_deepep_lowlatency.py
"""
import numpy as np

PSEUDOCODE = r"""
// ═══════════ 初始化(启动时一次,之后地址永不变)═══════════
recv_x    [E_loc][P][M_slot][hidden]   // ★ 固定预分配的接收槽:本卡每个专家 × 每个来源 rank
recv_cnt  [E_loc][P]                   //   计数器(RDMA atomic 写)
                                       //   无动态分配 → CUDA Graph 可 capture(low-latency 的前提)
// IBGDA:把 NIC 的 QP(队列对)/WQE(工作请求)环 map 进 GPU 显存
//   ⟹ GPU 的 SM 自己写 WQE、自己敲 doorbell —— 全程无 CPU proxy,α 降一个量级

// ═══════════ dispatch(send 半段,发完即返回)═══════════
__global__ void dispatch_send(x[B,h], topk_idx[B,k], topk_w[B,k]):
  for token i (warp 并行):
    // 1) 目标去重:k 个专家 → 它们所在的 rank 集合(≤k 个)
    ranks = unique( topk_idx[i][:] / E_loc )        // 专家号→rank 号,去重
    // 2) 传输量化:FP8 + per-128 scale(字节减半)
    xq, xs = fp8_quant(x[i])
    for r in ranks:
      for e in (token i 在 rank r 上选中的专家):     // 同 rank 多个专家:数据只发一份
        slot = ibgda_amo_fetch_add(&peer[r].recv_cnt[e][my_rank], 1)  // 远端原子拿槽位
        if (r == my_rank):                          // ★ 官方修正1:本 rank 直接 warp copy
          warp_copy(xq -> recv_x[e][my_rank][slot]) //   (2025.06 起机内尽量走 NVLink 而非 RDMA)
        else:
          ibgda_put_nbi(                            // ★ SM 直接构造 RDMA WRITE:
            src = {xq, xs, i},                      //   载荷带 home 槽号(★ 官方修正2:
            dst = peer[r].recv_x[e][my_rank][slot]) //   topk_weight 不随包发——home 侧本就有,
                                                    //   combine 归约时才乘,省载荷)
  ibgda_flush()                                     // 敲 doorbell 后立即返回
                                                    // ★★ RDMA 在网卡上飞行,不占任何 SM

// ═══════════ 中间:通信与计算重叠(TBO/shared expert 的钩子)═══════════
launch shared_expert_gemm(x)                        // shared expert 不需要 all2all
launch other_microbatch_compute()                   // 或 TBO 的另一个 micro-batch

// ═══════════ dispatch(recv hook 半段,要用数据时才调)═══════════
__global__ void dispatch_recv_hook():
  spin until recv_cnt 全部到位                       // 数据早被网卡直接写进 recv_x 了
  masked_m[e] = Σ_r recv_cnt[e][r]                  // ★ 直接生成 DeepGEMM masked 的输入
  // ⟹ 与 10 号文件无缝衔接:recv_x 就是 A[E_loc, M_max, K],零拷贝零重排

// ═══════════ combine(反向同构)═══════════
// 专家算完 → 输出 BF16 按 (home_rank, home_slot) 直写回源卡固定 buffer
// home 卡 hook:对每个 token 把 k 份部分输出 × topk_w 求和 —— 路由加权在这里完成
"""

P, E_loc, k, B_loc, H = 4, 4, 4, 8, 16          # 4 rank × 每卡4专家=16专家, top-4
E = P * E_loc
W = np.random.randn(E, H, H).astype(np.float32) * 0.3     # 专家权重(简化为方阵)
X = np.random.randn(P, B_loc, H).astype(np.float32)       # 每 rank 本地 token
# 路由:每 token 随机选 k 个不同专家 + softmax 权重
IDX = np.stack([np.random.choice(E, k, replace=False) for _ in range(P*B_loc)]
               ).reshape(P, B_loc, k)
Wt = np.random.rand(P, B_loc, k).astype(np.float32)
Wt /= Wt.sum(-1, keepdims=True)

def reference():
    Y = np.zeros((P, B_loc, H), np.float32)
    for r in range(P):
        for i in range(B_loc):
            for j in range(k):
                Y[r, i] += Wt[r, i, j] * (X[r, i] @ W[IDX[r, i, j]].T)
    return Y

def deepep_sim():
    M_slot = B_loc * k                                       # 预分配槽上限
    recv_x  = np.zeros((P, E_loc, P, M_slot, H), np.float32) # [dst][e][src][slot][h]
    recv_md = np.full((P, E_loc, P, M_slot, 2), -1)          # (home_slot, j) 元数据
    recv_wt = np.zeros((P, E_loc, P, M_slot), np.float32)
    cnt     = np.zeros((P, E_loc, P), int)
    sent_copies = 0
    # dispatch_send:每 token → 去重 rank → 直写对端固定槽
    for r in range(P):
        for i in range(B_loc):
            ranks = np.unique(IDX[r, i] // E_loc)            # ① 去重
            sent_copies += len(ranks)
            for j in range(k):
                dst, e = IDX[r, i, j] // E_loc, IDX[r, i, j] % E_loc
                s = cnt[dst, e, r]; cnt[dst, e, r] += 1      # ② 远端原子拿槽
                recv_x[dst, e, r, s] = X[r, i]               # ③ ibgda_put 直写
                recv_md[dst, e, r, s] = (i, j)
                recv_wt[dst, e, r, s] = Wt[r, i, j]
    # 专家计算(每卡对自己的 E_loc 个专家做 masked GEMM)
    Y = np.zeros((P, B_loc, H), np.float32)
    for dst in range(P):
        for e in range(E_loc):
            masked_m = cnt[dst, e].sum()                     # ★ 对接 DeepGEMM masked_m
            for src in range(P):
                for s in range(cnt[dst, e, src]):
                    out = recv_x[dst, e, src, s] @ W[dst*E_loc+e].T
                    # combine:直写回 home rank 的 (token, j) 槽并加权求和
                    i, j = recv_md[dst, e, src, s]
                    Y[src, i] += recv_wt[dst, e, src, s] * out
    return Y, sent_copies

# 去重后每 token 发往的 rank 数 C ≤ k（理论 C = P·(1-(1-1/P)^k)）
# hook 时序: send 返回后 RDMA 飞行不占 SM → 与 shared expert / TBO 重叠

# ============================================================
# Part K · EP 三耦合模型 (显存 / 步时间 / EPLB)
# ============================================================

import math, random

H100 = dict(fp8=1979e12, hbm=3.35e12*0.85, net=50e9,  nv=350e9)   # 8x400G IB

H20  = dict(fp8=296e12,  hbm=4.0e12*0.85,  net=20e9,  nv=350e9)   # 4 NIC → 20GB/s/卡

L, Lmoe, E, k, d, I = 61, 58, 256, 8, 7168, 2048

We = 3*d*I*1.0                # 44.04 MB / expert / layer (FP8)

KV_tok = L*576                # 35.1 KB/token (FP8, MLA latent)

N_other = 16.6e9              # 非专家激活参数

W_other_gb, RESERVE = 16, 12  # 复制的非专家权重 + activation/graph/DeepEP buffer

def kv_pool_gb(P, hbm_gb=80, red=32):
    return hbm_gb - (E+red)/P*Lmoe*We/1e9 - W_other_gb - RESERVE

def bmax(P, Lkv, hbm_gb=80, red=32):
    return kv_pool_gb(P, hbm_gb, red)*1e9/(Lkv*KV_tok)

def dedup(P): return P*(1-(1-1/P)**k)

def step_ms(hw, P, B, Lkv, red=32, tbo=True, tp_mode=False, mfu=0.5, imb=1.0):
    """imb = 最忙卡负载/平均 (不均衡放大 GEMM 与该卡 a2a 接收量)"""
    e_loc=(E+red)/P
    if tp_mode:   # TP16 基线: 全专家切1/16, token 在组内聚合
        t_mem = E*We/16/hw['hbm']
        Me=16*B*k/E
        t_gemm = 3*2*Me*d*(I/16)*E/(hw['fp8']*0.3)
        ar = 2*(15/16)*16*B*d*1.0
        t_comm = 2*(ar/ (25e9) + 40e-6)          # 跨2节点的 AllReduce 有效带宽
        t_moe = max(t_mem,t_gemm)
    else:
        t_mem = e_loc*We/hw['hbm']
        Me = B*P*k/(E+red)
        t_gemm = e_loc*3*2*max(Me,16)*d*I/(hw['fp8']*mfu) * imb
        t_comm = ((B*dedup(P)*d*1.0 + B*k*d*2.0)*imb/hw['net'] + 40e-6)
        t_moe = max(t_mem,t_gemm)
    t_attn = max(B*Lkv*576/hw['hbm'], 2*(N_other/L)*B/(hw['fp8']*0.4))
    if tbo and not tp_mode:
        t_layer = max(t_comm, t_moe+t_attn) + 0.15*min(t_comm, t_moe+t_attn)
    else:
        t_layer = t_comm + t_moe + t_attn
    return Lmoe*t_layer*1.35*1000    # 1.35 = 一次性校准的 overhead(其他kernel/气泡)

def eplb_mc(P, e_per, CV, Bg, red_frac=0.125, trials=300):
    sig = math.sqrt(math.log(1+CV*CV)); nE = P*e_per
    s0=s1=0
    for _ in range(trials):
        rate=[random.lognormvariate(0,sig) for _ in range(nE)]
        tot=sum(rate); rate=[r/tot for r in rate]
        tok=[max(1e-9, r*Bg*k*(1+random.gauss(0,1)/math.sqrt(max(r*Bg*k,1)))) for r in rate]
        # 无EPLB: 随机放置
        idx=list(range(nE)); random.shuffle(idx)
        g=[sum(tok[j] for j in idx[i*e_per:(i+1)*e_per]) for i in range(P)]
        s0+=max(g)/(sum(g)/P)
        # EPLB: 复制 top 热专家 + LPT 装箱
        nred=int(nE*red_frac)
        items=sorted(tok, reverse=True)
        items=[items[i]/2 for i in range(nred)]*2+items[nred:]
        items.sort(reverse=True)
        g2=[0.0]*P; cap=[e_per+nred//P+1]*P; cnt=[0]*P
        for it in items:
            j=min((x for x in range(P) if cnt[x]<cap[x]), key=lambda x:g2[x])
            g2[j]+=it; cnt[j]+=1
        s1+=max(g2)/(sum(g2)/P)
    return s0/trials, s1/trials

def e2e(spd, f): return 1/(f/spd + (1-f))

# 锚点（原自测结论）:
#   A. TP16→EP72 的 B_max ≈ 4×
#   B. EP72/TP16 吞吐 ≈ 5×（实测 ~5.2×）
#   C. EPLB 端到端：跨机小 batch ~1.1×；prefill 高 CV ~1.5×；decode EP72 ~2.5×
#   D. 每卡专家越少、CV 越高 → EPLB 越赚
