"""
matmul.py — L4 第一颗：fused GEMM（三阶段）

写法：stage0 给你当骨架；stage1（grouped swizzle）、stage2（persistent）留给你填。
每个阶段都是独立的 kernel + host。填完用 bench_matmul.py 对拍 + 对比 cuBLAS。

验收目标（bench_matmul.py 会算）：任一阶段达到 cuBLAS 的 ~80%（ratio ≥ 0.8）。
stage0 → stage1 一般能 +10~15%；stage1 → stage2 在规整 shape 下通常最接近 cuBLAS。
"""

import torch
import triton
import triton.language as tl


# ==========================================================================
# Stage 0 · 基础分块 GEMM（参考骨架，已写好）
#   C[i,j] = sum_k A[i,k]·B[k,j]
#   每块 (BLOCK_M, BLOCK_N) 由 tensor core 通过 tl.dot 累加 BLOCK_K 步
# ==========================================================================
@triton.jit
def _matmul_stage0(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


# ==========================================================================
# Stage 1 · Grouped Ordering（L2 swizzle）——你来写
#   目的：让相邻 program 共享 K 维 tile，提高 L2 命中率。
#   改动点：从 2D grid(pid_m, pid_n) 改成一维 grid(pid)，把 pid 映射到 (pid_m, pid_n)。
#   参考（官方 03-matmul，你笔记里有）：
#     pid = tl.program_id(0)
#     num_pid_m = tl.cdiv(M, BLOCK_M); num_pid_n = tl.cdiv(N, BLOCK_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m
# ==========================================================================
@triton.jit
def _matmul_stage1(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_num_m = tl.cdiv(M,BLOCK_M)
    pid_num_n = tl.cdiv(N,BLOCK_N)
    num_block_per_group = GROUP_SIZE_M * pid_num_n
    group_id = pid // num_block_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    num_m_in_group = min(pid_num_m-first_pid_m,GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_block_per_group % num_m_in_group
    old_pid_n = pid % num_block_per_group // num_m_in_group
    if group_id % 2 != 0:
        pid_n = pid_num_n - 1 - old_pid_n
    else :
        pid_n = old_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(a_ptrs,mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs,mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))



# ==========================================================================
# Stage 2 · Persistent ——你来写
#   目的：固定线程块数 = SMs 数量，循环分派所有 tile，省掉反复切块调度。
#   改动点：grid 一维且长度 = NUM_SMS（launch 时算好）；内核内 while/tl.range 遍历全部 tile。
#   grid = (min(NUM_SMS, cdiv(M,BM)*cdiv(N,BN)),)  ← host 端
#   num_programs = tl.num_programs(0)
#   for pid in tl.range(tl.program_id(0), num_pid_m*num_pid_n, num_programs, flatten=True):
#       ... 每个 pid 再算 swizzle 得 (pid_m, pid_n) ...
# ==========================================================================
@triton.jit
def _matmul_stage2(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    IS_MAJOR_M = M >= N
    pid = tl.program_id(0)
    pid_num_m = tl.cdiv(M,BLOCK_M)
    pid_num_n = tl.cdiv(N,BLOCK_N)
    totol_block = pid_num_m * pid_num_n
    num_programs = tl.num_programs(axis=0)

    for start_pid in tl.range(pid, totol_block, num_programs, flatten=True):
        if IS_MAJOR_M:
            num_block_per_group = GROUP_SIZE_M * pid_num_n
            group_id = start_pid // num_block_per_group
            first_pid_m = group_id * GROUP_SIZE_M   
            num_m_in_group = min(pid_num_m-first_pid_m,GROUP_SIZE_M)
            pid_m = first_pid_m + start_pid % num_block_per_group % num_m_in_group
            old_pid_n = start_pid % num_block_per_group // num_m_in_group
            if group_id % 2 != 0:
                pid_n = pid_num_n - 1 - old_pid_n
            else :
                pid_n = old_pid_n
        else:
            num_block_per_group = GROUP_SIZE_M * pid_num_m
            group_id = start_pid // num_block_per_group
            first_pid_n = group_id * GROUP_SIZE_M
            num_n_in_group = min(pid_num_n-first_pid_n,GROUP_SIZE_M)
            pid_n = first_pid_n + start_pid % num_block_per_group % num_n_in_group
            old_pid_m = start_pid % num_block_per_group // num_n_in_group
            if group_id % 2 != 0:
                pid_m = pid_num_m - 1 - old_pid_m
            else :
                pid_m = old_pid_m
        offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offset_m < M
        mask_n = offset_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_tiles = K // BLOCK_K
        k_remain = K % BLOCK_K
        for k_tile in range(k_tiles):
            offset_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
            b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
        if k_remain > 0:    
            offset_k = k_tiles * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offset_k < K
            a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
            b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
            acc = tl.dot(a, b, acc)
        c_ptrs = c_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))



# ============================ host 启动器 ============================
def _common_args(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    return c, M, N, K


def matmul_stage0(a, b, BM=128, BN=128, BK=64):
    a, b = a.contiguous(), b.contiguous()
    c, M, N, K = _common_args(a, b)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _matmul_stage0[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1),
                         BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c


def matmul_stage1(a, b, BM=128, BN=128, BK=64, GROUP_SIZE_M=8):
    a, b = a.contiguous(), b.contiguous()
    c, M, N, K = _common_args(a, b)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _matmul_stage1[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1),
                         GROUP_SIZE_M=GROUP_SIZE_M,
                         BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c


def matmul_stage2(a, b, BM=128, BN=128, BK=64, GROUP_SIZE_M=8):
    a, b = a.contiguous(), b.contiguous()
    c, M, N, K = _common_args(a, b)
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    grid = (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
    _matmul_stage2[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1),
                         GROUP_SIZE_M=GROUP_SIZE_M,
                         BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
    return c

if __name__ == "__main__":
    torch.manual_seed(0)
    for M, N, K in [(128, 512, 512), (128, 2048, 512), (64, 4096, 1024), (16, 768, 768)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c0 = matmul_stage0(a, b)
        c1 = matmul_stage1(a, b)
        c2 = matmul_stage2(a, b)
        c_ref = torch.matmul(a, b)
        print(f"M={M}, N={N}, K={K}, stage0 vs ref: {torch.allclose(c0, c_ref, rtol=1e-2, atol=1e-2)}, "
                f"stage1 vs ref: {torch.allclose(c1, c_ref, rtol=1e-2, atol=1e-2)}, "
                f"stage2 vs ref: {torch.allclose(c2, c_ref, rtol=1e-2, atol=1e-2)}")