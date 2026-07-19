"""
基于Pytorch实现基础版本Self-Attention
基于FlashAttention实现高效版本Self-Attention
在FlashAttention基础上优化Tiling设计
"""

import torch
import torch.nn as nn
import math
import triton
import triton.language as tl

class VanillaSelfAttention(nn.Module):
    """Standard Self-Attention implementation"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class FlashAttention(nn.Module):
    """Flash Attention implementation with block-sparse attention"""
    def __init__(self, dim, num_heads=8, block_size=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Implement block-sparse attention
        output = self._flash_attention(q, k, v)
        
        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        return output
    
    def _flash_attention(self, q, k, v):
        B, H, N, D = q.shape  # batch_size, num_heads, seq_length, head_dim
        
        # 初始化输出和中间结果
        output = torch.zeros_like(q)
        softmax_sum = torch.zeros(B, H, N, 1, device=q.device)
        max_score = torch.full((B, H, N, 1), float('-inf'), device=q.device)
        
        # 按块处理序列
        for block_start in range(0, N, self.block_size):
            block_end = min(block_start + self.block_size, N)
            
            # 当前块的Key和Value
            k_block = k[:, :, block_start:block_end, :]  # B x H x Br x D
            v_block = v[:, :, block_start:block_end, :]  # B x H x Br x D
            
            # 计算注意力分数
            scores = torch.matmul(q, k_block.transpose(-2, -1)) * self.scale  # B x H x N x Br
            
            # 更新最大分数
            block_max_score = torch.max(scores, dim=-1, keepdim=True)[0]
            max_score_prev = max_score
            max_score = torch.maximum(max_score, block_max_score)
            
            # 计算局部softmax
            exp_scores = torch.exp(scores - max_score)
            exp_scores_prev = torch.exp(max_score_prev - max_score)
            
            # 更新累积和
            softmax_sum = softmax_sum * exp_scores_prev + exp_scores.sum(dim=-1, keepdim=True)
            
            # 更新输出
            output = output * exp_scores_prev + torch.matmul(exp_scores, v_block)
        
        # 归一化输出
        output = output / softmax_sum
        
        return output

class OptimizedFlashAttention(nn.Module):
    """Optimized Flash Attention with improved tiling strategy"""
    def __init__(self, dim, num_heads=8, block_size=64):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Implement optimized tiling strategy
        output = self._optimized_flash_attention(q, k, v)
        
        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        return output
    
    def _optimized_flash_attention(self, q, k, v):
        B, H, N, D = q.shape
        output = torch.zeros_like(q)
        softmax_sum = torch.zeros(B, H, N, 1, device=q.device)
        max_scores = torch.full((B, H, N, 1), float('-inf'), device=q.device)
        
        # 对每个query block进行处理
        for q_start in range(0, N, self.block_size):
            q_end = min(q_start + self.block_size, N)
            q_block = q[:, :, q_start:q_end, :]
            
            # 初始化当前q_block的局部累积值
            block_output = torch.zeros_like(q_block)
            block_softmax_sum = torch.zeros(B, H, q_end-q_start, 1, device=q.device)
            block_max_scores = torch.full((B, H, q_end-q_start, 1), float('-inf'), device=q.device)
            
            # 对每个key-value block进行处理
            for k_start in range(0, N, self.block_size):
                k_end = min(k_start + self.block_size, N)
                k_block = k[:, :, k_start:k_end, :]
                v_block = v[:, :, k_start:k_end, :]
                
                # 计算当前块的注意力分数
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                
                # 更新最大分数
                block_max_score = torch.max(scores, dim=-1, keepdim=True)[0]
                max_score_prev = block_max_scores
                block_max_scores = torch.maximum(max_score_prev, block_max_score)
                
                # 计算局部softmax
                exp_scores = torch.exp(scores - block_max_scores)
                exp_scores_prev = torch.exp(max_score_prev - block_max_scores)
                
                # 更新当前q_block的累积和和输出
                block_softmax_sum = block_softmax_sum * exp_scores_prev + exp_scores.sum(dim=-1, keepdim=True)
                block_output = block_output * exp_scores_prev + torch.matmul(exp_scores, v_block)
            
            # 将当前q_block的计算结果存储到最终输出中
            output[:, :, q_start:q_end] = block_output
            softmax_sum[:, :, q_start:q_end] = block_softmax_sum
        
        # 最终归一化
        output = output / softmax_sum
        return output

@triton.jit
def _flash_attn_forward(
    q_ptr, k_ptr, v_ptr, out_ptr, softmax_lse_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    causal: tl.constexpr,
):
    # Compute position in the output matrix
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(seq_len, BLOCK_M)
    block_id = pid // num_blocks
    block_m = pid % num_blocks

    # Offsets for the current block
    offset_m = block_m * BLOCK_M
    offset_n = 0
    
    # Pointers for the current block
    q_block_ptr = q_ptr + offset_m * stride_qm
    k_block_ptr = k_ptr + offset_n * stride_kn
    v_block_ptr = v_ptr + offset_n * stride_vn
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(float(head_dim))
    
    # Load Q block
    q = tl.load(q_block_ptr)
    q = q * scale
    
    # Loop over K,V blocks
    for block_n in range(0, tl.cdiv(seq_len, BLOCK_N)):
        k = tl.load(k_block_ptr + block_n * BLOCK_N * stride_kn)
        v = tl.load(v_block_ptr + block_n * BLOCK_N * stride_vn)
        
        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        
        # Apply causal mask if needed
        if causal:
            mask = tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :]
            scores = tl.where(mask, scores, float("-inf"))
        
        # Apply softmax
        scores = tl.softmax(scores)
        
        # Compute output
        acc += tl.dot(scores, v)
    
    # Store output
    out_block_ptr = out_ptr + offset_m * stride_om
    tl.store(out_block_ptr, acc)

class TritonFlashAttentionV2(nn.Module):
    """FlashAttention V2 implementation using Triton"""
    def __init__(self, dim, num_heads=8, dropout=0.0, causal=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.causal = causal
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Optimal block sizes for A100 GPU
        self.BLOCK_M = 128
        self.BLOCK_N = 128
        self.BLOCK_DMODEL = self.head_dim
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Reshape for Triton kernel
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # Initialize output tensor
        output = torch.empty_like(q)
        
        # Launch Triton kernel
        grid = (B * self.num_heads * triton.cdiv(N, self.BLOCK_M),)
        _flash_attn_forward[grid](
            q, k, v, output,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            B, self.num_heads, N, self.head_dim,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_DMODEL,
            self.causal,
        )
        
        output = output.reshape(B, N, C)
        output = self.proj(output)
        return output