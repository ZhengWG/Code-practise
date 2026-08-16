"""
FlashAttention-2 前向的 Triton kernel 版本（GPU-only）。

与 flash_attention.py 里的 PyTorch 实现算法等价，只是把 ④~⑧（打分/softmax/加权和）
融进一个 kernel，用 online-softmax 分块流式计算，不物化 (N, N) 中间矩阵。

⚠️ 需要 GPU + triton 才能运行。无 GPU/triton 的机器上：
   - 导入本模块不会报错（triton 缺失时 _HAS_TRITON=False，kernel 不定义）；
   - 直接运行会打印跳过信息，不做数值校验（本文件未在纯 CPU 环境验证）。

有 GPU 时，__main__ 会拿 flash_attention.VanillaSelfAttention 的输出做 allclose 基线。
"""

import math

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - 本机环境无 triton
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _flash_attn_fwd_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        seq_len, scale,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        """每个 program 负责 (一个 batch*head, 一个 query 分块)。

        标准 online-softmax：维护 (acc, l_i, m_i)，遍历 KV 分块增量更新。
        """
        pid_m = tl.program_id(0)           # query 分块编号
        pid_bh = tl.program_id(1)          # batch * head 编号（B*H 已折叠进第 1 维）

        q_base = q_ptr + pid_bh * stride_qh
        k_base = k_ptr + pid_bh * stride_kh
        v_base = v_ptr + pid_bh * stride_vh
        o_base = o_ptr + pid_bh * stride_oh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)

        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

        # 因果场景下，query 分块 pid_m 只需看 KV 到 (pid_m+1)*BLOCK_M
        n_end = (pid_m + 1) * BLOCK_M if CAUSAL else seq_len
        for start_n in range(0, n_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len, other=0.0)

            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where(offs_n[None, :] < seq_len, s, float("-inf"))    # 尾块 padding
            if CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)                # 旧累积量修正系数
            p = tl.exp(s - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
        tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < seq_len)

    def triton_flash_attention(q, k, v, causal=False, block_m=64, block_n=64):
        """q/k/v: (B, H, N, D)，返回 (B, H, N, D)。仅在有 GPU+triton 时可用。"""
        B, H, N, D = q.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        o = torch.empty_like(q)
        scale = 1.0 / math.sqrt(D)
        grid = (triton.cdiv(N, block_m), B * H)
        _flash_attn_fwd_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            N, scale,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=D,
            CAUSAL=causal,
        )
        return o

else:  # 无 triton：提供同名占位，调用即报错，避免误用
    def triton_flash_attention(*args, **kwargs):
        raise RuntimeError("triton 未安装或无 GPU，无法使用 triton_flash_attention")


# ---------------------------------------------------------------------------
# 自测：仅在有 GPU+triton 时对齐纯 PyTorch 参考实现。
# ---------------------------------------------------------------------------
def _self_test():
    if not (_HAS_TRITON and torch.cuda.is_available()):
        print("跳过 Triton 自测：需要 GPU + triton（本环境不满足，kernel 未验证）")
        return

    from flash_attention import VanillaSelfAttention

    torch.manual_seed(0)
    B, N, dim, heads = 2, 128, 64, 8
    head_dim = dim // heads
    device = "cuda"

    for causal in (False, True):
        q = torch.randn(B, heads, N, head_dim, device=device)
        k = torch.randn(B, heads, N, head_dim, device=device)
        v = torch.randn(B, heads, N, head_dim, device=device)

        out = triton_flash_attention(q, k, v, causal=causal)

        # 参考：直接用 VanillaSelfAttention 的 _attention（同样吃 (B,H,N,D)）
        ref_mod = VanillaSelfAttention(dim, heads, causal=causal).to(device)
        ref = ref_mod._attention(q, k, v)

        max_err = (out - ref).abs().max().item()
        ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)   # fp32 kernel 容差放宽
        tag = "causal" if causal else "full"
        print(f"[{tag:6}] triton_flash_attention max_err={max_err:.2e} -> {'OK' if ok else 'FAIL'}")
        assert ok, f"triton ({tag}) 与 Vanilla 不一致"

    print("triton flash attention matched ✓")


if __name__ == "__main__":
    _self_test()
