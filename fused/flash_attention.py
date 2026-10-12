"""
flash_attention.py — L4 收官：FlashAttention（在线 softmax 融合注意力）

Phase 1（已过）：forward，online softmax 更新块你已写对。
Phase 2（现在）：backward。
  - forward 现在额外存 LSE（log2 域：lse = m_i + log2(l_i)）
  - 反向拆两个内核：
      A) _flash_bwd_dv_kernel    : dV += P^T @ dO；并原子累加 D_i = rowsum(dP·P) 到 D 缓冲
      B) _flash_bwd_dqdk_kernel  : 用完整 D，算 ds = (dP - D·P)·scale，dq / dk
  - 你只写两个 TODO 块（各 4-5 行核心数学），其余骨架已搭好。

反向推导回顾（标准 flash v2）：
  P = softmax(S), S = scale·QK^T
  dV  = P^T @ dO
  dP  = dO @ V^T
  D_i = rowsum_j(P_ij · dP_ij)          # 每行标量
  dS  = P ⊙ (dP - D·P)                  # 中心化
  dQ += dS @ K ;  dK += dS^T @ Q        # scale 已折进 dS
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


# ==========================================================================
# Phase 1 · forward（已通过；新增 LSE 存储）
# ==========================================================================
@triton.jit
def _flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    softmax_scale_log2e,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lb, stride_lh, stride_lm,
    N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    loop_end = N_CTX
    if CAUSAL:
        loop_end = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)

    for start_n in range(0, loop_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        mask_n = offs_n[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=mask_n, other=0.0)
        v = tl.load(v_ptrs, mask=mask_n, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale_log2e
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))
        else:
            qk = tl.where(offs_n[None, :] < N_CTX, qk, float('-inf'))

        # ---- 在线 softmax 更新（你已写对）----
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        p_typed = p.to(tl.float16) if DTYPE == 0 else p.to(tl.bfloat16)
        acc = tl.dot(p_typed, v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    o = acc.to(tl.float16) if DTYPE == 0 else acc.to(tl.bfloat16)
    o_ptrs = O_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o, mask=offs_m[:, None] < N_CTX)

    # LSE（log2 域）：lse = m_i + log2(l_i)，backward 用 exp2(qk - lse) 直接还原 P
    lse = m_i + tl.math.log2(l_i)
    lse_ptrs = LSE_ptr + pid_bh * stride_lh + offs_m
    tl.store(lse_ptrs, lse, mask=offs_m < N_CTX)


# ==========================================================================
# Phase 2A · 反向内核 1：dV + D（grid 按 KV 块）
#   dV += P^T @ dO（本 KV program 局部累加，循环外 store）
#   D_i = rowsum(P·dP)，跨 KV program 原子累加（D 缓冲 [B,H,N]）
# ==========================================================================
@triton.jit
def _flash_bwd_dv_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, DV_ptr, D_ptr, LSE_ptr,
    softmax_scale_log2e,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lb, stride_lh, stride_lm,
    N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, DTYPE: tl.constexpr,
):
    pid_kv = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offs_n = pid_kv * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < N_CTX

    # 本 KV 块长驻寄存器：k、v
    k_ptrs = K_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    # 内层循环遍历所有 Q 块（dO 块流式进入）
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_CTX
        q_ptrs = Q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = DO_ptr + pid_bh * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        lse_ptrs = LSE_ptr + pid_bh * stride_lh + offs_m
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale_log2e
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))
        else:
            qk = tl.where(offs_n[None, :] < N_CTX, qk, float('-inf'))

        # ===================== TODO（Phase 2A 核心） =====================
        p = tl.math.exp2(qk - lse[:, None])              # 用 LSE 一次还原归一化 P
        p_typed = p.to(tl.float16) if DTYPE == 0 else p.to(tl.bfloat16)
        dv = tl.dot(tl.trans(p_typed), do, dv)          # dV += P^T @ dO
        dp = tl.dot(do, tl.trans(v))                     # dP = dO @ V^T
        di = tl.sum(dp * p, axis=1)                      # 本块对 D_i 的贡献
        tl.atomic_add(D_ptr + pid_bh * stride_lh + offs_m, di, mask=mask_m)
        pass
        # =============================================================

    dv_ptrs = DV_ptr + pid_bh * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dv.to(tl.float16) if DTYPE == 0 else dv.to(tl.bfloat16), mask=mask_n[:, None])


# ==========================================================================
# Phase 2B · 反向内核 2：dQ + dK（grid 按 KV 块，内层遍历 Q 块）
#   用完整 D 缓冲（2A 结束后）：ds = P·(dP - D·P)·scale
#   dK：本 KV program 局部累加（每 program 独占一个 KV 块）
#   dQ：跨 KV program 原子累加（每个 KV 块都对所有 Q 行有贡献）
# ==========================================================================
@triton.jit
def _flash_bwd_dqdk_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr, DQ_ptr, DK_ptr, D_ptr, LSE_ptr,
    softmax_scale, softmax_scale_log2e,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_lb, stride_lh, stride_lm,
    N_CTX: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, DTYPE: tl.constexpr,
):
    pid_kv = tl.program_id(0)
    pid_bh = tl.program_id(1)
    offs_n = pid_kv * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < N_CTX

    k_ptrs = K_ptr + pid_bh * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < N_CTX
        q_ptrs = Q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = DO_ptr + pid_bh * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
        lse_ptrs = LSE_ptr + pid_bh * stride_lh + offs_m
        d_ptrs = D_ptr + pid_bh * stride_lh + offs_m
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        d_i = tl.load(d_ptrs, mask=mask_m, other=0.0)   # 完整 D_i（2A 已算完）

        qk = tl.dot(q, tl.trans(k)) * softmax_scale_log2e
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))
        else:
            qk = tl.where(offs_n[None, :] < N_CTX, qk, float('-inf'))

        # ===================== TODO（Phase 2B 核心） =====================
        p = tl.math.exp2(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - d_i[:, None]) * softmax_scale          # 中心化 + 折 scale
        ds_typed = ds.to(tl.float16) if DTYPE == 0 else ds.to(tl.bfloat16)
        tl.atomic_add(DQ_ptr + pid_bh*stride_dqh + offs_m[:,None]*stride_dqm + offs_d[None,:]*stride_dqd,
                      tl.dot(ds_typed, k), mask=mask_m[:, None])       # dQ 跨 KV 原子累加
        dk = tl.dot(tl.trans(ds_typed), q, dk)                        # dK 本 program 局部累加
        # pass
        # =============================================================

    dk_ptrs = DK_ptr + pid_bh * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dk.to(tl.float16) if DTYPE == 0 else dk.to(tl.bfloat16), mask=mask_n[:, None])


# ============================ host 包装 + autograd ============================
class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, H, N, D = q.shape
        o = torch.empty_like(q)
        lse = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(N, BLOCK_M), B * H)
        softmax_scale_log2e = (1.0 / (D ** 0.5)) * LOG2E
        _flash_fwd_kernel[grid](
            q, k, v, o, lse, softmax_scale_log2e,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            N_CTX=N, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=causal, DTYPE=0 if q.dtype == torch.float16 else 1,
        )
        ctx.save_for_backward(q, k, v, lse)
        ctx.causal = causal
        ctx.D = D
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, lse = ctx.saved_tensors
        B, H, N, D = q.shape
        do = do.contiguous()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        D_buf = torch.zeros(B, H, N, device=q.device, dtype=torch.float32)  # D_i 缓冲

        softmax_scale = 1.0 / (D ** 0.5)
        softmax_scale_log2e = softmax_scale * LOG2E
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(N, BLOCK_N), B * H)

        _flash_bwd_dv_kernel[grid](
            q, k, v, do, dv, D_buf, lse, softmax_scale_log2e,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            N_CTX=N, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=ctx.causal, DTYPE=0 if q.dtype == torch.float16 else 1,
        )

        _flash_bwd_dqdk_kernel[grid](
            q, k, v, do, dq, dk, D_buf, lse, softmax_scale, softmax_scale_log2e,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            N_CTX=N, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            CAUSAL=ctx.causal, DTYPE=0 if q.dtype == torch.float16 else 1,
        )
        return dq, dk, dv, None


def flash_attention(q, k, v, causal=True):
    """融合注意力：o = softmax(QK^T·scale (+ causal)) @ V（带反向）"""
    return FlashAttnFunc.apply(q, k, v, causal)
