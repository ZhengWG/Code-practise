"""
GEMM实现: 考虑M，K，N的Tiling设计，考虑是否需要进行分块计算，以及自动分块的逻辑
Implementation of GEMM (General Matrix Multiplication) with tiling optimization
"""

import numpy as np
from multiprocessing import Process, Queue

def determine_tile_size(M, N, K, cache_size_kb=32):
    """
    Automatically determine appropriate tile sizes based on matrix dimensions and cache size,
    using basic-block concept for better cache utilization
    """
    # Assuming double precision (8 bytes per element)
    element_size = 8
    cache_size = cache_size_kb * 1024  # Convert to bytes
    
    # Basic block size (typical L1 cache line size is 64 bytes)
    basic_block_size = 64 // element_size  # Number of elements in one cache line
    
    # Reserve some cache for other operations
    effective_cache = cache_size * 0.8
    
    # For matrix multiplication, we need to store:
    # 1. One block of matrix A (tile_M * tile_K)
    # 2. One block of matrix B (tile_K * tile_N)
    # 3. One block of output matrix C (tile_M * tile_N)
    
    # Start with basic block size and adjust based on cache constraints
    tile_M = basic_block_size
    tile_N = basic_block_size
    tile_K = basic_block_size
    
    # Adjust sizes to fit in cache while maintaining basic block alignment
    while (tile_M * tile_K + tile_K * tile_N + tile_M * tile_N) * element_size < effective_cache:
        if tile_M < M:
            tile_M += basic_block_size
        if tile_N < N:
            tile_N += basic_block_size
        if tile_K < K:
            tile_K += basic_block_size
            
    # Ensure we don't exceed matrix dimensions
    tile_M = min(tile_M, M)
    tile_N = min(tile_N, N)
    tile_K = min(tile_K, K)
    
    # Ensure tiles are multiples of basic block size for better alignment
    tile_M = (tile_M // basic_block_size) * basic_block_size
    tile_N = (tile_N // basic_block_size) * basic_block_size
    tile_K = (tile_K // basic_block_size) * basic_block_size
    
    return tile_M, tile_N, tile_K

def gemm_tiled_double_buffered(A, B, C=None):
    """
    Perform matrix multiplication C = A @ B with tiling optimization and double buffering
    Args:
        A: Matrix of shape (M, K)
        B: Matrix of shape (K, N)
        C: Optional output matrix of shape (M, N)
    Returns:
        Result matrix C
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Matrix dimensions must match for multiplication"
    
    # Initialize output matrix if not provided
    if C is None:
        C = np.zeros((M, N))
    
    # Determine tile sizes
    tile_M, tile_N, tile_K = determine_tile_size(M, N, K)
    
    # Allocate double buffers for tiles
    A_tiles = [np.zeros((tile_M, tile_K)) for _ in range(2)]
    B_tiles = [np.zeros((tile_K, tile_N)) for _ in range(2)]
    C_tile = np.zeros((tile_M, tile_N))
    
    # Helper function to load tiles asynchronously
    def async_load_tiles(A, B, A_tiles, B_tiles, i, j, k, buf_idx, tile_M, tile_N, tile_K, M, N, K, queue):
        i_end = min(i + tile_M, M)
        j_end = min(j + tile_N, N)
        k_end = min(k + tile_K, K)
        
        A_tiles[buf_idx][:i_end-i, :k_end-k] = A[i:i_end, k:k_end]
        B_tiles[buf_idx][:k_end-k, :j_end-j] = B[k:k_end, j:j_end]
        
        queue.put((buf_idx, i_end-i, j_end-j, k_end-k))
    
    # Helper function to compute tile asynchronously
    def async_compute_tile(A_tiles, B_tiles, C_tile, buf_idx, i_size, j_size, k_size, queue):
        C_tile[:i_size, :j_size] += A_tiles[buf_idx][:i_size, :k_size] @ B_tiles[buf_idx][:k_size, :j_size]
        queue.put((i_size, j_size))
    
    # Main loop with double buffering and multiprocessing
    for i in range(0, M, tile_M):
        for j in range(0, N, tile_N):
            # Clear accumulation tile
            C_tile.fill(0)
            
            # Create queues for communication
            load_queue = Queue()
            compute_queue = Queue()
            
            # Load first tiles
            load_process = Process(target=async_load_tiles, args=(A, B, A_tiles, B_tiles, i, j, 0, 0, tile_M, tile_N, tile_K, M, N, K, load_queue))
            load_process.start()
            
            for k in range(0, K, tile_K):
                i_size = min(tile_M, M - i)
                j_size = min(tile_N, N - j)
                k_size = min(tile_K, K - k)
                
                # Start loading next tile
                next_k = k + tile_K
                if next_k < K:
                    load_process = Process(target=async_load_tiles, args=(A, B, A_tiles, B_tiles, i, j, next_k, (k // tile_K + 1) % 2, tile_M, tile_N, tile_K, M, N, K, load_queue))
                    load_process.start()
                
                # Compute current tile
                buf_idx, i_size, j_size, k_size = load_queue.get()
                compute_process = Process(target=async_compute_tile, args=(A_tiles, B_tiles, C_tile, buf_idx, i_size, j_size, k_size, compute_queue))
                compute_process.start()
                
                # Wait for computation to finish
                compute_process.join()
            
            # Write result back to C
            i_size, j_size = compute_queue.get()
            C[i:i+i_size, j:j+j_size] = C_tile[:i_size, :j_size]
    
    return C

def gemm_unrolled(A, B, C):
    # 适合小矩阵的计算
    M, K = A.shape
    for i in range(0, M, 4):  # 展开4次
        for j in range(N):
            sum0 = sum1 = sum2 = sum3 = 0
            for k in range(K):
                sum0 += A[i][k] * B[k][j]
                sum1 += A[i+1][k] * B[k][j] 
                sum2 += A[i+2][k] * B[k][j]
                sum3 += A[i+3][k] * B[k][j]
            C[i][j] = sum0
            C[i+1][j] = sum1
            C[i+2][j] = sum2  
            C[i+3][j] = sum3

def prefetch_block(A, i, j, block_size):
    return A[i:i+block_size, j:j+block_size].copy()

def gemm_prefetch(A, B, C):
    with ThreadPoolExecutor() as executor:
        # 提前预取下一个要使用的数据块
        future = executor.submit(prefetch_block, A, next_i, next_j, block_size)
        # 当前块计算
        current_block = future.result()

def optimize_layout(A):
    # 将矩阵重排成更有利于访问的格式,如将行主序改为块主序
    block_size = 32
    M, N = A.shape
    A_blocked = np.zeros_like(A)
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            A_blocked[i:i+block_size, j:j+block_size] = A[i:i+block_size, j:j+block_size]
    return A_blocked

