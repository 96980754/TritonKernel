"""
bench_matmul.py — fused GEMM 验收：对拍 + 对比 cuBLAS（torch.matmul 底层就是 cuBLAS）

用法：
  python bench_matmul.py

输出：
  1. 每个阶段每个 shape 的正确性（最大误差）
  2. 每阶段 vs cuBLAS 的 ratio 表（ratio = cuBLAS_time / kernel_time；ratio ≥ 0.8 即达 cuBLAS ~80%）

先只填了 stage0 也能跑；stage1/stage2 写完 host 后会自动加进来。
"""

import torch
import triton.testing

from fused_gemm import matmul_stage0, matmul_stage1, matmul_stage2


STAGES = {
    "stage0(basic)": matmul_stage0,
    "stage1(swizzle)": matmul_stage1,
    "stage2(persistent)": matmul_stage2,
}

SIZES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (4096, 4096, 2048),
]


def check_correct(name, a, b, c):
    ref = torch.matmul(a.float(), b.float())  # fp32 参考（消除 fp16 累加误差的评判标准）
    err = (c.float() - ref).abs().max().item()
    tol = ref.abs().max().item() * 0.05  # 相对 5%
    ok = err <= max(tol, 1e-2)
    print(f"  {name:<20} max_err={err:.4f}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== 正确性（vs torch.matmul fp32） ===")
    for (M, N, K) in SIZES:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        print(f"\n[M={M} N={N} K={K}]")
        for name, fn in STAGES.items():
            try:
                c = fn(a, b)
                check_correct(name, a, b, c)
            except Exception as e:
                print(f"  {name:<20} ERROR: {type(e).__name__}: {str(e)[:80]}")

    print("\n=== 基准 vs cuBLAS（ratio = cuBLAS/kernel，≥0.8 达标） ===")
    print(f"{'shape':>18} | " + " | ".join(f"{n:>14}" for n in STAGES))
    for (M, N, K) in SIZES:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        def cub():
            return torch.matmul(a, b)

        t_cub = triton.testing.do_bench(cub)
        row = f"[{M}x{N}x{K}]".rjust(18) + " | "
        for name, fn in STAGES.items():
            try:
                t_k = triton.testing.do_bench(lambda: fn(a, b))
                ratio = t_cub / t_k
                row += f"{ratio:>13.2f}x "
            except Exception as e:
                row += f"{'ERR':>14}"
        print(row)
