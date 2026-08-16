"""
GEMM (通用矩阵乘 C = A @ B) 的几种实现，演示访存/分块相关优化思路。

  gemm_naive       —— 直接 A @ B，作为正确性基线
  gemm_tiled       —— M/N/K 三层 Tiling，改善 cache 命中
  gemm_unrolled    —— 对 M 方向做 4 路循环展开（小矩阵示范）
  gemm_prefetch    —— 用后台线程预取下一行块，与当前块计算重叠
  to_block_major   —— 行主序 -> 块主序的内存重排（配 from_block_major 可逆）

所有实现都以 np.matmul 为基线做 allclose 校验（见 __main__）。
"""

import math

import numpy as np
from concurrent.futures import ThreadPoolExecutor


def determine_tile_size(M, N, K, cache_size_kb=32, element_size=8):
    """按 cache 容量粗略估算 tile_M / tile_N / tile_K。

    约束：一块 A(tile_M×tile_K) + 一块 B(tile_K×tile_N) + 一块 C(tile_M×tile_N)
    要能放进有效 cache。以 cache line 元素数为基本块，逐步放大直到临界。
    """
    cache_bytes = cache_size_kb * 1024
    basic = max(1, 64 // element_size)        # 一条 cache line 能放的元素数
    effective = cache_bytes * 0.8             # 给其他数据留些余量

    tile_M = tile_N = tile_K = basic
    while (tile_M * tile_K + tile_K * tile_N + tile_M * tile_N) * element_size < effective:
        grown = False
        if tile_M < M:
            tile_M += basic; grown = True
        if tile_N < N:
            tile_N += basic; grown = True
        if tile_K < K:
            tile_K += basic; grown = True
        if not grown:                         # 三个维度都已覆盖整矩阵
            break

    def _fit(t, dim):
        # 对齐到 basic 的倍数，且不小于 basic、不超过实际维度
        return min(max(basic, (t // basic) * basic), dim)

    return _fit(tile_M, M), _fit(tile_N, N), _fit(tile_K, K)


def gemm_naive(A, B):
    """基线实现，直接调用 BLAS。"""
    return A @ B


def gemm_tiled(A, B, tile=None):
    """三层 Tiling 的分块矩阵乘。

    tile 为 None 时用 determine_tile_size 自动决定；否则用统一的方块 tile。
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "内维不匹配：A 是 (M,K)，B 应为 (K,N)"

    if tile is None:
        tM, tN, tK = determine_tile_size(M, N, K)
    else:
        tM = tN = tK = tile

    C = np.zeros((M, N), dtype=np.result_type(A, B))
    for i in range(0, M, tM):
        for j in range(0, N, tN):
            acc = C[i:i + tM, j:j + tN]       # 视图，+= 直接写回 C
            for k in range(0, K, tK):
                acc += A[i:i + tM, k:k + tK] @ B[k:k + tK, j:j + tN]
    return C


def gemm_unrolled(A, B):
    """对 M 方向做 4 路循环展开（纯 Python 三重循环，仅示范，勿用于大矩阵）。

    修复点：原实现里 N 未定义、且默认 M 是 4 的倍数；这里补全 N 并处理尾部余数。
    """
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.result_type(A, B))

    m4 = (M // 4) * 4
    for i in range(0, m4, 4):                 # 主体：一次算 4 行
        for j in range(N):
            s0 = s1 = s2 = s3 = 0.0
            for k in range(K):
                bkj = B[k, j]
                s0 += A[i, k] * bkj
                s1 += A[i + 1, k] * bkj
                s2 += A[i + 2, k] * bkj
                s3 += A[i + 3, k] * bkj
            C[i, j], C[i + 1, j], C[i + 2, j], C[i + 3, j] = s0, s1, s2, s3

    for i in range(m4, M):                    # 尾部不足 4 行
        for j in range(N):
            C[i, j] = sum(A[i, k] * B[k, j] for k in range(K))
    return C


def gemm_prefetch(A, B, block_size=64):
    """按行块计算，用一个后台线程预取下一行块，实现"取数/计算"重叠。

    numpy 的 @ 会释放 GIL，所以后台线程的拷贝可与主线程计算真正并行。
    修复点：原实现里 next_i/next_j/block_size 未定义、且缺少线程池 import。
    """
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=np.result_type(A, B))

    def load_row_block(i):
        return A[i:i + block_size].copy()     # 模拟把下一块搬到"快存储"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(load_row_block, 0)          # 预取第一块
        for i in range(0, M, block_size):
            a_blk = future.result()                      # 拿当前块
            nxt = i + block_size
            if nxt < M:
                future = pool.submit(load_row_block, nxt)  # 立刻预取下一块
            C[i:i + a_blk.shape[0]] = a_blk @ B            # 当前块计算与预取重叠
    return C


def to_block_major(A, block=32):
    """行主序 -> 块主序：每个 block×block 子块在内存里连续存放。

    返回 (flat, meta)，flat 是一维数组。配合 from_block_major 可无损还原。
    块主序对分块访问更 cache 友好（同一子块的元素地址相邻）。
    """
    M, N = A.shape
    mb, nb = math.ceil(M / block), math.ceil(N / block)
    padded = np.zeros((mb * block, nb * block), dtype=A.dtype)
    padded[:M, :N] = A
    # (mb, block, nb, block) -> (mb, nb, block, block) -> 展平
    blocks = padded.reshape(mb, block, nb, block).transpose(0, 2, 1, 3)
    return blocks.reshape(-1).copy(), (M, N, block)


def from_block_major(flat, meta):
    """to_block_major 的逆操作。"""
    M, N, block = meta
    mb, nb = math.ceil(M / block), math.ceil(N / block)
    blocks = flat.reshape(mb, nb, block, block).transpose(0, 2, 1, 3)
    return blocks.reshape(mb * block, nb * block)[:M, :N].copy()


# ---------------------------------------------------------------------------
# 自测：所有 GEMM 实现对齐 np.matmul；layout 变换验证可逆。
# ---------------------------------------------------------------------------
def _self_test():
    rng = np.random.default_rng(0)
    M, K, N = 37, 53, 41                       # 故意用不整除的尺寸测边界
    A = rng.standard_normal((M, K))
    B = rng.standard_normal((K, N))
    ref = A @ B

    checks = {
        "gemm_naive": gemm_naive(A, B),
        "gemm_tiled (auto)": gemm_tiled(A, B),
        "gemm_tiled (tile=16)": gemm_tiled(A, B, tile=16),
        "gemm_unrolled": gemm_unrolled(A, B),
        "gemm_prefetch": gemm_prefetch(A, B, block_size=8),
    }
    for name, C in checks.items():
        err = np.abs(C - ref).max()
        ok = np.allclose(C, ref, atol=1e-9, rtol=1e-6)
        print(f"{name:24} max_err={err:.2e} -> {'OK' if ok else 'FAIL'}")
        assert ok, f"{name} 与 np.matmul 不一致"

    flat, meta = to_block_major(A, block=16)
    back = from_block_major(flat, meta)
    assert np.array_equal(back, A), "block-major 往返不可逆"
    print("to_block_major/from_block_major round-trip -> OK")

    print("all gemm implementations matched np.matmul ✓")


if __name__ == "__main__":
    _self_test()
