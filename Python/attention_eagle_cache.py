"""
LLM 推理三大优化的核心原理（伪代码，重可读性，不保证可运行）

  Part 0  注意力的 9 个步骤 —— 后面所有变体都只是在其中一两步动手
  Part 1  注意力变体        —— 省 KV 体积 / 省注意力计算量
  Part 1b MLA weight absorption 详解
  Part 2  投机解码          —— 省"串行前向的次数"
  Part 3  分层 KV Cache     —— 省重复的 prefill
  Part 4  三者串起来 + 冲突点
"""

# ============================================================
# Part 0  一次注意力的全部步骤
# ============================================================

def attention(x, cache, pos):
    B, T, D = x.shape                    # batch, 新来的 token 数, 模型维度

    # ① 投影：一个线性层，切成 H 个头
    q = Wq(x).view(B, T, H, dh).transpose(1, 2)      # (B,H,T,dh)
    k = Wk(x).view(B, T, H, dh).transpose(1, 2)      # ← GQA 把这里的 H 改成 G
    v = Wv(x).view(B, T, H, dh).transpose(1, 2)      # ← MLA 把这里换成低秩 c

    # ② 位置编码：只加在 q/k 上，v 不加（v 是"内容"，不需要位置）
    q, k = RoPE(q, pos), RoPE(k, pos)

    # ③ 拼 KV Cache
    k, v = cache.append(k, v)                        # (B,H,S,dh)，S = 历史 + T
    #     ★ 显存瓶颈就在这行: 2 * H * dh * S * layers * dtype

    # ④ 打分
    score = q @ k.transpose(-1, -2)                  # (B,H,T,S)

    # ⑤ 缩放：防止内积随维度变大把 softmax 压成 one-hot
    score = score / dh**0.5

    # ⑥ 因果掩码：注意 q 的绝对位置是 S-T+i，不是 i
    score = score.masked_fill(~(q_pos[:, None] >= k_pos[None, :]), -inf)
    #     ← Sparse 在这里额外挖掉一大片

    # ⑦ 归一化
    p = score.softmax(-1)
    #     ← Linear Attention 就是删掉这一步，从而能用结合律

    # ⑧ 加权求和
    o = p @ v                                        # (B,H,T,dh)

    # ⑨ 合头 + 输出投影：让各头信息混合
    return Wo(o.transpose(1, 2).reshape(B, T, D))


# 三条容易忽略但决定一切的事实:
#   · ④⑧ 是仅有的两个矩阵乘 (q·kᵀ 和 p·v)，长序列时 O(S²) —— Sparse/Linear 打这两个
#   · ③  是显存瓶颈 —— GQA/MLA 打这个
#   · flash attention 不改数学，只把 ④~⑧ 融成一个 kernel、分块流式算 softmax，
#     从不把 (T,S) 中间矩阵写回显存 —— 省的是带宽和激活，不省 FLOPs


# ============================================================
# Part 1  注意力变体
# ============================================================

# --- MHA: 基线，每个头一套 K/V --------------------------------
def mha(x):
    q, k, v = Wq(x), Wk(x), Wv(x)        # 各 H 个头
    return attn(q, k, v)                  # cache: 2*H*dh 每 token


# --- GQA: 少存几套 K/V，多个 q 头共用 -------------------------
def gqa(x, H, G):                         # G=1 即 MQA, G=H 即 MHA
    q = Wq(x)                             # H 个头
    k, v = Wk(x), Wv(x)                   # 只有 G 组   cache: 2*G*dh ← 缩小 H/G 倍
    k, v = k.repeat(H // G), v.repeat(H // G)     # 计算前才摊开（kernel 里连摊开都省）
    return attn(q, k, v)


# --- MLA: K/V 不存，存低秩"压缩包" ----------------------------
def mla(x):
    c = W_dkv(x)                          # ★ cache 只存 c，维度 r << 2*H*dh
    k_nope, v = W_uk(c), W_uv(c)          # 用时再解压

    # 麻烦点: RoPE 含位置信息，不能和解压矩阵交换顺序（见 Part 1b 末尾）
    #        → 额外拆一小段单独做 RoPE，且所有头共享一份
    k_rope = RoPE(W_kr(x))                # ★ cache 也存它（很小）
    q_nope, q_rope = split(Wq(x))
    score = q_nope @ k_nope.T + q_rope @ k_rope.T
    return softmax(score) @ v
    # decode 时不走这条朴素路径，走 Part 1b 的吸收路径


# --- Sparse: 只看"近的 + 重要的" ------------------------------
def sparse(q, k, v):
    块代表 = k.reshape(块数, 块长, dh).mean(1)      # landmark
    重要块 = topk(q @ 块代表.T, n)
    可见 = 滑动窗口 | 开头几个token(sink) | 重要块
    return attn(q, k[可见], v[可见])                 # O(S) → O(窗口 + n*块长)


# --- Linear: 去掉 softmax，用结合律 (qk)v → q(kv) --------------
def linear_decode(q, k, v):
    S = 0                                 # ★ 固定大小状态，与序列长度无关
    for t in range(T):
        S = g[t] * S + k[t].T @ v[t]      # g = 遗忘门(RetNet常数 / GLA可学)
        o[t] = q[t] @ S                   # 解码 O(1)，无膨胀的 KV cache
    return o


def linear_train(q, k, v):
    """串行循环训练太慢 → 切 chunk: 块内二次(打满GPU)，块间线性(复杂度不炸)"""
    S = 0
    for 块 in chunks:
        o_块间 = q[块] @ S                                   # 读上一块传下来的状态
        o_块内 = (q[块] @ k[块].T * 因果mask * 衰减) @ v[块]    # 块小，二次可接受
        o[块] = o_块间 + o_块内
        S = 衰减 * S + k[块].T @ v[块]                        # 状态往下传
    return o


# ============================================================
# Part 1b  MLA weight absorption
# ============================================================
#
# 符号（DeepSeek-V3 实际尺寸）:
#   h        ∈ R^d           输入            d  = 7168
#   c = W_dkv h ∈ R^r        KV 隐向量        r  = 512    ★ cache 只存这个
#   W_uk[i] ∈ R^{dc × r}     第 i 头 K 解压矩阵  dc = 128
#   W_uv[i] ∈ R^{dc × r}     第 i 头 V 解压矩阵
#   H = 128 头

def mla_naive(c, q):
    """朴素路径 —— prefill 走这条"""
    k = W_uk @ c                          # (S,128,128) 解压出完整 K
    v = W_uv @ c
    return softmax(q @ k.T) @ v
    # 问题: 每个 decode step 都要解压整段 c，
    #      中间激活比 cache 本身大 ~32 倍，纯属浪费带宽


# --- 吸收 1: K 侧解压矩阵吸进 Q -------------------------------
#   q[i]ᵀ · k[i] = q[i]ᵀ · (W_uk[i] · c) = (W_uk[i]ᵀ · q[i])ᵀ · c
#                                           └──── 记作 q̃[i] ────┘
#   纯矩阵乘结合律，恒等变形。
def absorb_k(c, q):
    q̃ = W_uk.T @ q                        # (r,) = (512,)  query 被"抬"到隐空间
    return q̃ @ c.T                        # ★ 直接和 cache 里的 c 算，K 从不解压

    # 更进一步: q 本身也来自 q_lora (q[i] = W_uq[i] @ c_q)，可离线预乘
    #   W_absorbed_q[i] = W_uk[i].T @ W_uq[i]     # (r × q_lora)，加载时算一次
    #   q̃[i] = W_absorbed_q[i] @ c_q              # 运行时只剩一个 matmul


# --- 吸收 2: V 侧解压矩阵吸进输出投影 --------------------------
#   o[i] = Σ_j a_j·v[i,j] = Σ_j a_j·(W_uv[i]·c_j) = W_uv[i]·(Σ_j a_j·c_j)
#                                                    └─ 先在隐空间加权求和 ─┘
#   线性算子与加权求和可交换顺序。
def absorb_v(c, a):
    õ = a @ c                             # ★ 在 512 维隐空间做加权和，V 从不解压
    return W_absorbed_o @ õ               # W_absorbed_o[i] = Wo[i] @ W_uv[i]，离线预乘


# --- 吸收后的最终形态 -----------------------------------------
def mla_absorbed(c, k_rope, c_q, q_rope):
    """cache 只有 c(512) + k_rope(64) = 576 维 / token / 层"""
    q̃     = W_absorbed_q @ c_q            # (H, 512) 每头一个 512 维 query
    score = q̃ @ c.T + q_rope @ k_rope.T   # ★ 所有头共享同一份 c
    õ     = softmax(score) @ c            # (H, 512)
    return W_absorbed_o @ õ

    # ★ 关键洞察: 吸收后 MLA 在计算形态上退化成 head_dim=576 的 MQA
    #   （所有头共享唯一一份 K/V）→ 可直接复用成熟的 MQA/GQA kernel


# --- 为什么不总用吸收版 ---------------------------------------
#                    朴素              吸收
#   点积维度         128 + 64          512 + 64   (约 3× FLOPs)
#   读取 KV 数据量   (S,128,256)解压后  (S,576) 原始 cache
#   适合             prefill(q多,算力受限)  decode(q少,带宽受限)
#
#   decode 时 q 只有 1~几个 token，彻底 memory-bound：
#   多算 3 倍 FLOPs 不心疼，省 32 倍数据搬运才是关键。
#   prefill 时 q 有几千个，compute-bound：解压一次让所有 q 复用更划算。
#   → 真实实现(SGLang/vLLM)是两套 kernel 按场景切换；
#     投机解码"一次验 k 个 token"正好卡中间，要按 draft 长度定阈值。
#
# 两个实现细节:
#   · W_uk.T @ W_uq 预乘要在 fp32 下做再转回 bf16，否则 512×128 连乘掉精度
#   · RoPE 那 64 维【不能】吸收 —— q_rope 与 k_rope 之间隔着一个位置相关的
#     旋转矩阵 R(m-n)，不满足结合律前提。这正是 MLA 必须把位置编码
#     解耦出来单独存的根本原因。


# ============================================================
# Part 2  投机解码
#   小模型猜 k 个 token，大模型一次并行验证：猜对白赚，猜错退回
# ============================================================

def speculative_step(前缀):
    # ① 猜：便宜，串行 k 次
    draft = [小模型.猜下一个() for _ in range(k)]

    # ② 验：贵，但只做 1 次前向（而非 k 次）—— 加速的根源
    真值 = 大模型(前缀 + draft)
    n = 最长匹配前缀长度(draft, 真值)

    # ③ 收尾
    kv_cache.截断(len(前缀) + n)          # 被拒绝部分的 KV 必须丢掉
    return draft[:n] + [真值[n]]          # 末尾白送一个 bonus token


def 采样版接受判定(x, p_大, q_小):
    """贪心时"相等即接受"；采样时须用拒绝采样，否则分布被小模型污染"""
    if rand() < min(1, p_大[x] / q_小[x]):
        return x
    return 从 normalize(max(0, p_大 - q_小)) 重采样()
    # 结论: 输出分布与直接用大模型采样完全一致 —— 纯加速，不掉质量


def eagle_draft(f, tok):
    """EAGLE: 小模型不预测 token，预测大模型的隐状态（更好学）"""
    f = DecoderLayer(W @ concat(emb(tok), f))     # 一层就够，特征层自回归
    tok = 大模型.lm_head(f)                        # ★ 复用大模型的输出头
    return f, tok


def tree_draft():
    """树形草稿: 每步取 top-k 而非 top-1，仍只需 1 次大模型前向验证所有分支"""
    mask[i][j] = (j 是 i 的祖先) or (i == j)       # 节点只能看到自己这条路径
    # 接受时从根往下走，孩子 token == 大模型贪心结果就继续
    # 之后 KV cache 按接受路径 gather 重排


def mtp(h, ids):
    """DeepSeek MTP: 不外挂小模型，主模型自带 D 个串行小模块"""
    for k in range(D):
        h = Block_k(Proj_k(concat(norm(h), norm(emb(ids[t + k + 1])))))
        logits[k] = 共享输出头(h)                  # 预测 x[t+k+2]
    loss = CE(主模型) + λ * mean(CE(logits))       # 训练时就学多步预测
    # 推理时这些头直接当草稿用


# ============================================================
# Part 3  分层 KV Cache (HiCache)
#   KV 按前缀树共享；GPU 装不下就往下沉一层，而不是扔掉
#   GPU ⇅ CPU ⇅ SSD   往下=淘汰(write-back)，往上=命中后捞回(promote)
# ============================================================

node = {
    "tokens": ..., "children": ...,
    "loc":  ["gpu", "cpu", "disk", ...],   # ★ 每一页各自在哪一层
    "ref":  0,                             # >0 表示正在被用，不许淘汰
    "last": 时间戳,                         # LRU
}


def acquire(tokens):
    链, 命中长度 = 前缀匹配(tokens)           # 命中部分不用重算 prefill
    for n in 链:
        n.ref += 1
        for i, tier in enumerate(n.loc):
            if tier != "gpu":
                n.loc[i] = 搬回GPU(tier)      # promote，只搬缺的那几页
    return 命中长度


def alloc_gpu_page():
    if GPU池.空了():
        受害者 = LRU(所有 ref==0 的节点)        # 优先叶子，复用概率最低
        搬运(受害者, "gpu" -> "cpu")            # ★ write-back，不是删掉
        if CPU池.满了():
            搬运(LRU_cpu, "cpu" -> "disk")      # ★ spill 落盘
    return GPU池.pop()


def 前缀匹配_节点分裂():
    """前缀缓存唯一的技巧点：公共前缀只匹配到节点中间时要分裂"""
    # 已有: ["你是助手，请用中文回答"]
    # 新来: ["你是助手，请用英文回答"]
    # 分裂: 父["你是助手，请用"] → 子1["中文回答"], 子2["英文回答"]
    # 分裂只转移页的所有权，★ 不复制任何 KV 数据


# 关键判断:
#   · 页为粒度(16/32 token)，否则元数据比数据还贵
#   · 沉一层 ≠ 丢弃: 搬回来花带宽，重算花算力，前者通常便宜得多
#   · 只有 搬运时间 < 重算时间 才值得分层，短前缀不必落盘
#   · L1↔L2 用独立 CUDA stream + pinned memory，按 layer 流水，与计算重叠


# ============================================================
# Part 4  三者串起来：一次真实的 decode step
# ============================================================

def decode_step(tokens):
    # ① HiCache: 能复用的前缀直接拿，剩下的才 prefill
    命中 = acquire(tokens)
    if 命中 < len(tokens):
        prefill(tokens[命中:])

    # ② EAGLE: 起草 k 个候选（内部注意力用 MLA 吸收路径，读的是 ① 的 KV 页）
    draft = [eagle_draft(...) for _ in range(k)]

    # ③ 大模型一次前向验证 → 接受 n 个
    n, 输出 = verify(draft)

    # ④ 收尾: 两套 cache 都要按接受结果对齐
    kv_cache.截断(前缀长度 + n)
    release(链)
    return 输出


# ------------------------------------------------------------
# 优化的轴不同，所以可叠加:
#   GQA/MLA        省每 token 的 KV 体积     KV 缩 4~30×，batch 更大
#   Sparse/Linear  省长序列的注意力计算量     O(S²) → O(S)
#   EAGLE/MTP      省串行前向的次数          端到端 2~4×
#   HiCache        省重复的 prefill          多轮/共享 prompt 首 token 延迟大降
#
# 但互相有冲突（工程真正的坑）:
#   · MLA + 投机: 吸收版在小 batch 最划算，而投机验证=小 batch prefill，
#                 两条路径最优 kernel 不同，要按 draft 长度切换
#   · Sparse + 前缀缓存: 跳过部分 KV 但仍要整段存着 —— 省算力不省显存
#   · 投机 + 大 batch: GPU 本已打满，多出的并行度没处用，加速比会退化甚至变负
#                     （投机主要是"低并发下的延迟优化"）
#   · HiCache 搬运和 权重加载/跨卡通信 抢 PCIe，需限流 + 独立 stream
# ------------------------------------------------------------
