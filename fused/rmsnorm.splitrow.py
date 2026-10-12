"""
rmsnorm_splitrow.py — L3: 拆行 2D grid 版 RMSNorm forward（两段式：归约 + 归一）

对比：split-row(拆行) vs 你现有的 fused_rmsnorm(1 program/行) vs torch 两步。
先跑 Nsight 诊断，再用本脚本实测拆分是否真的提升带宽。

用法：
  python rmsnorm_splitrow.py
"""

import torch
import triton
import triton.language as tl
import triton.testing

from fused_rmsnorm import rmsnorm  # 你现有的版本（1 program / 行）


# ---------------- 拆行两段式：归约 + 归一 ----------------
@triton.jit
def _rms_reduce_kernel(x_ptr, sum_sq_ptr, D, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = row * D + chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (row + 1) * D
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.atomic_add(sum_sq_ptr + row, tl.sum(x * x, axis=0))  # 每个 program 局部 sum(x²)，原子累加


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, sum_sq_ptr, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = row * D + chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < (row + 1) * D
    rms = tl.sqrt(tl.load(sum_sq_ptr + row) / D + eps)  # 标量
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + (offs - row * D), mask=mask, other=0.0)  # w 是 [D]
    tl.store(y_ptr + offs, x / rms * w, mask=mask)


def rmsnorm_splitrow(x, w, eps=1e-5, BLOCK=1024):
    x = x.contiguous()
    B, D = x.shape
    nchunk = triton.cdiv(D, BLOCK)
    sum_sq = torch.zeros(B, device=x.device, dtype=torch.float32)  # 每次归零
    grid = (B, nchunk)
    _rms_reduce_kernel[grid](x, sum_sq, D, BLOCK=BLOCK)
    y = torch.empty_like(x)
    _rms_norm_kernel[grid](x, w, y, sum_sq, D, eps, BLOCK=BLOCK)
    return y


# ---------------- torch 参考 ----------------
def rmsnorm_torch(x, w, eps=1e-5):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


if __name__ == "__main__":
    # 正确性：split-row vs torch
    torch.manual_seed(0)
    for B, D in [(128, 512), (128, 2048), (64, 4096), (16, 768)]:
        x = torch.randn(B, D, device="cuda", dtype=torch.float32)
        w = torch.randn(D, device="cuda", dtype=torch.float32)
        y = rmsnorm_splitrow(x, w)
        ref = rmsnorm_torch(x, w)
        assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4), f"mismatch {(B,D)}"
    print("[PASS] split-row correctness (4 shapes)")

    # 基准：split-row vs 你现有的 fused vs torch
    print(f"\n{'D':>6} {'split-row':>12} {'fused(1row)':>13} {'torch':>10}")
    for B, D in [(128, 512), (128, 2048), (128, 4096)]:
        x = torch.randn(B, D, device="cuda", dtype=torch.float32)
        w = torch.randn(D, device="cuda", dtype=torch.float32)
        t_split = triton.testing.do_bench(lambda: rmsnorm_splitrow(x, w))
        t_fused = triton.testing.do_bench(lambda: rmsnorm(x, w))
        t_torch = triton.testing.do_bench(lambda: rmsnorm_torch(x, w))
        print(f"{D:>6} {t_split*1000:>9.2f}us {t_fused*1000:>10.2f}us {t_torch*1000:>8.2f}us")
    print("\n注意：split-row 多读一遍 x（总搬 3·B·D 而非 2·B·D），"
          "只有并发提升带来的带宽收益 > 多读代价才算赢——看哪个快。")
