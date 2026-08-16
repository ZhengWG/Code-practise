"""
Self-Attention 的几种等价实现，从朴素版逐步演进到 FlashAttention。

  VanillaSelfAttention   —— 标准 softmax attention，作为正确性基线
  FlashAttention         —— 单层循环 online-softmax（q 全量，遍历 KV 分块）
  OptimizedFlashAttention—— 双层 Tiling（query 分块 × KV 分块），即 FA2 的算法形态

三种实现在数学上完全等价，__main__ 里会用 allclose 互相校验。
公共的 QKV 投影 / 多头 reshape / 输出投影抽到 AttentionBase，子类只实现 _attention()。
GPU 上的 Triton kernel 版本见同目录 triton_flash_attention.py。
"""

import torch
import torch.nn as nn


class AttentionBase(nn.Module):
    """所有自注意力变体的公共骨架。

    子类只需实现 ``_attention(q, k, v)``：三者形状均为 (B, H, N, D)，返回同形状。
    QKV 投影、多头切分/合并、输出投影这些重复逻辑都收敛在这里。
    """

    def __init__(self, dim, num_heads=8, causal=False):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _split_heads(self, x):
        # (B, N, C) -> 3 个 (B, H, N, D)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, N, D)
        return qkv[0], qkv[1], qkv[2]

    def _merge_heads(self, x):
        # (B, H, N, D) -> (B, N, C)
        B, H, N, D = x.shape
        return x.transpose(1, 2).reshape(B, N, H * D)

    def _attention(self, q, k, v):
        raise NotImplementedError

    def forward(self, x):
        q, k, v = self._split_heads(x)
        out = self._attention(q, k, v)
        return self.proj(self._merge_heads(out))


class VanillaSelfAttention(AttentionBase):
    """标准实现：一次性算出完整 (N, N) 注意力矩阵。O(N^2) 显存。"""

    def _attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale       # (B, H, N, N)
        if self.causal:
            N = q.shape[-2]
            mask = torch.triu(
                torch.ones(N, N, dtype=torch.bool, device=q.device), diagonal=1
            )
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        return attn @ v


class FlashAttention(AttentionBase):
    """单层循环版：q 保持全量，按块遍历 KV，用 online-softmax 增量更新。

    只是把 (N, N) 中间矩阵拆成 (N, block) 分块，省掉一次性物化整张分数矩阵；
    数学上与 Vanilla 完全一致。
    """

    def __init__(self, dim, num_heads=8, block_size=64, causal=False):
        super().__init__(dim, num_heads, causal)
        self.block_size = block_size

    def _attention(self, q, k, v):
        B, H, N, D = q.shape
        out = torch.zeros_like(q)
        l = torch.zeros(B, H, N, 1, device=q.device, dtype=q.dtype)              # softmax 分母
        m = torch.full((B, H, N, 1), float("-inf"), device=q.device, dtype=q.dtype)  # 行最大值

        row = torch.arange(N, device=q.device)[:, None]        # (N, 1) query 的绝对位置
        for start in range(0, N, self.block_size):
            end = min(start + self.block_size, N)
            k_blk = k[:, :, start:end, :]
            v_blk = v[:, :, start:end, :]

            s = (q @ k_blk.transpose(-2, -1)) * self.scale     # (B, H, N, blk)
            if self.causal:
                col = torch.arange(start, end, device=q.device)[None, :]   # (1, blk)
                s = s.masked_fill(col > row, float("-inf"))

            m_new = torch.maximum(m, s.max(dim=-1, keepdim=True)[0])
            alpha = torch.exp(m - m_new)                       # 旧累积量的修正系数
            p = torch.exp(s - m_new)                           # (B, H, N, blk)

            l = l * alpha + p.sum(dim=-1, keepdim=True)
            out = out * alpha + p @ v_blk
            m = m_new

        return out / l


class OptimizedFlashAttention(AttentionBase):
    """双层 Tiling（FA2 的算法形态）：外层 query 分块，内层 KV 分块。

    每个 query 分块自带一份 (o, l, m) 局部状态，KV 遍历完再一次性归一化写回。
    因果场景下可跳过完全位于未来的 KV 分块。
    """

    def __init__(self, dim, num_heads=8, block_size=64, causal=False):
        super().__init__(dim, num_heads, causal)
        self.block_size = block_size

    def _attention(self, q, k, v):
        B, H, N, D = q.shape
        bs = self.block_size
        out = torch.zeros_like(q)

        for q_start in range(0, N, bs):
            q_end = min(q_start + bs, N)
            q_blk = q[:, :, q_start:q_end, :]
            Bq = q_end - q_start

            o = torch.zeros(B, H, Bq, D, device=q.device, dtype=q.dtype)
            l = torch.zeros(B, H, Bq, 1, device=q.device, dtype=q.dtype)
            m = torch.full((B, H, Bq, 1), float("-inf"), device=q.device, dtype=q.dtype)
            row = torch.arange(q_start, q_end, device=q.device)[:, None]   # (Bq, 1)

            for k_start in range(0, N, bs):
                # 因果掩码下，整块都在未来 (k_start > 当前 query 最大位置) 可直接跳过
                if self.causal and k_start > q_end - 1:
                    break
                k_end = min(k_start + bs, N)
                k_blk = k[:, :, k_start:k_end, :]
                v_blk = v[:, :, k_start:k_end, :]

                s = (q_blk @ k_blk.transpose(-2, -1)) * self.scale     # (B, H, Bq, blk)
                if self.causal:
                    col = torch.arange(k_start, k_end, device=q.device)[None, :]
                    s = s.masked_fill(col > row, float("-inf"))

                m_new = torch.maximum(m, s.max(dim=-1, keepdim=True)[0])
                alpha = torch.exp(m - m_new)
                p = torch.exp(s - m_new)

                l = l * alpha + p.sum(dim=-1, keepdim=True)
                o = o * alpha + p @ v_blk
                m = m_new

            out[:, :, q_start:q_end] = o / l

        return out


# ---------------------------------------------------------------------------
# 自测：所有变体在数值上应与 Vanilla 一致（因/非因果各测一遍）。
# ---------------------------------------------------------------------------
def _self_test():
    torch.manual_seed(0)
    B, N, dim, heads = 2, 100, 64, 8      # N 故意不被 block_size 整除，测边界
    x = torch.randn(B, N, dim)

    for causal in (False, True):
        ref = VanillaSelfAttention(dim, heads, causal=causal)
        variants = {
            "FlashAttention": FlashAttention(dim, heads, block_size=32, causal=causal),
            "OptimizedFlashAttention": OptimizedFlashAttention(dim, heads, block_size=32, causal=causal),
        }
        # 让变体复用同一套权重，才能逐元素比较
        with torch.no_grad():
            y_ref = ref(x)
            tag = "causal" if causal else "full"
            for name, m in variants.items():
                m.qkv.load_state_dict(ref.qkv.state_dict())
                m.proj.load_state_dict(ref.proj.state_dict())
                y = m(x)
                max_err = (y - y_ref).abs().max().item()
                ok = torch.allclose(y, y_ref, atol=1e-5, rtol=1e-4)
                print(f"[{tag:6}] {name:24} max_err={max_err:.2e} -> {'OK' if ok else 'FAIL'}")
                assert ok, f"{name} ({tag}) 与 Vanilla 不一致"

    print("all attention variants matched ✓")


if __name__ == "__main__":
    _self_test()
