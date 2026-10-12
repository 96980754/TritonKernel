"""
test_softmax.py — fused softmax 验收脚本（已写好）

运行：
  python test_softmax.py

作用：
  1. 前向正确性：y vs torch.softmax（多 shape，含 D 非 2 的幂）
  2. 反向正确性：ds vs 标准公式（含大 logits 稳定性，不产生 NaN）
  3. 基准：fused vs torch.softmax，do_bench
"""

import torch
import triton.testing

from fused_softmax import softmax


# ---------------------------- 参考实现（评判标准） ----------------------------
def ref_forward(s):
    return torch.softmax(s, dim=-1)


def ref_backward(s, g):
    """标准 softmax backward：ds = y·(g − dot)，dot = Σ g·y。"""
    y = torch.softmax(s, dim=-1)
    dot = (g * y).sum(-1, keepdim=True)
    return y * (g - dot)


# ---------------------------- 正确性 ----------------------------
def test_forward():
    shapes = [(8, 512), (16, 768), (32, 100), (7, 257), (4, 1)]
    for B, D in shapes:
        s = torch.randn(B, D, device="cuda", dtype=torch.float32)
        y = softmax(s)
        ref = ref_forward(s)
        assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), f"fwd mismatch {(B,D)}: {(y-ref).abs().max()}"
    print("[PASS] forward correctness (5 shapes)")


def test_backward():
    shapes = [(8, 512), (16, 768), (32, 100), (7, 257)]
    for B, D in shapes:
        s = torch.randn(B, D, device="cuda", dtype=torch.float32, requires_grad=True)
        y = softmax(s)
        g = torch.randn_like(y)
        y.backward(g)
        ds_ref = ref_backward(s.detach(), g)
        assert s.grad is not None
        assert torch.allclose(s.grad, ds_ref, atol=1e-5, rtol=1e-5), f"bwd mismatch {(B,D)}: {(s.grad-ds_ref).abs().max()}"
    print("[PASS] backward correctness (4 shapes)")

    # 稳定性：大 logits（1e3）与近全等行，都不应出现 NaN
    for B, D in [(8, 512), (4, 257)]:
        for mag in [1e3, 1e-3]:
            s = torch.full((B, D), mag, device="cuda", dtype=torch.float32).requires_grad_(True)
            y = softmax(s)
            y.backward(torch.randn_like(y))
            assert torch.isfinite(s.grad).all() and torch.isfinite(y).all(), f"NaN on mag={mag}"
    print("[PASS] numerical stability (large/tiny logits, no NaN)")


# ---------------------------- 基准（直接 do_bench，不依赖 matplotlib） ----------------------------
def bench():
    print(f"\n{'D':>6} {'fused':>12} {'torch.softmax':>14} {'speedup':>8}")
    for D in [512, 1024, 2048, 4096]:
        B = 128
        s = torch.randn(B, D, device="cuda", dtype=torch.float32)

        def fused():
            softmax(s)

        def torch_ref():
            ref_forward(s)

        ms_fused = triton.testing.do_bench(fused)
        ms_ref = triton.testing.do_bench(torch_ref)
        print(f"{D:>6} {ms_fused*1000:>9.2f}us {ms_ref*1000:>11.2f}us {ms_ref/ms_fused:>6.2f}x")


if __name__ == "__main__":
    test_forward()
    test_backward()
    print("\n[bench] 运行基准（约 1 分钟）：")
    bench()

    # ---- Nsight（在 RMSNorm 那轮已装好则直接可用）----
    # ncu --target-processes all --kernel-name regex:_softmax_fwd python test_softmax.py
    # 看：DRAM Throughput / Achieved Occupancy / Registers Per Thread / Local Load-Store
