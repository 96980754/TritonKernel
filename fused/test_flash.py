"""
test_flash.py — FlashAttention forward 验收（Phase 1）

运行：
  python test_flash.py

作用：
  1. forward 正确性：flash vs 显式 fp32 参考（含 causal / non-causal）
  2. 基准：flash vs torch.nn.functional.scaled_dot_product_attention
"""

import torch
import torch.nn.functional as F
import triton.testing

from flash_attention import flash_attention


def ref_attn(q, k, v, causal):
    """显式 fp32 参考：softmax(QK^T·scale (+ causal)) @ V"""
    B, H, N, D = q.shape
    scale = D ** -0.5
    s = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale
    if causal:
        mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))[None, None]
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, v.float()).to(q.dtype)


def check(B, H, N, D, causal):
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    o = flash_attention(q, k, v, causal=causal)
    ref = ref_attn(q, k, v, causal)
    err = (o.float() - ref.float()).abs().max().item()
    tol = 0.03 * ref.abs().max().item() + 1e-3
    ok = err <= tol
    print(f"  B={B} H={H} N={N} D={D} causal={int(causal)}  max_err={err:.4f}  {'OK' if ok else 'FAIL'}")
    return ok


def check_backward(B, H, N, D, causal):
    """flash 反向 vs fp32 参考 autograd：比 dq/dk/dv"""
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(q)

    o = flash_attention(q, k, v, causal=causal)
    o.backward(do)
    dq, dk, dv = q.grad, k.grad, v.grad

    # fp32 参考（同一份数据）
    qr = q.detach().float().requires_grad_()
    kr = k.detach().float().requires_grad_()
    vr = v.detach().float().requires_grad_()
    or_ = ref_attn(qr, kr, vr, causal)
    or_.backward(do.float())

    res = []
    for name, g, gr in [("dq", dq, qr.grad), ("dk", dk, kr.grad), ("dv", dv, vr.grad)]:
        err = (g.float() - gr).abs().max().item()
        tol = 0.05 * gr.abs().max().item() + 1e-3
        ok = err <= tol
        res.append(ok)
        print(f"  B={B} H={H} N={N} D={D} causal={int(causal)} {name} max_err={err:.4f}  {'OK' if ok else 'FAIL'}")
    return all(res)


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== forward 正确性（vs fp32 显式参考） ===")
    all_ok = True
    for N in [256, 512, 1024, 2048]:
        for causal in [False, True]:
            all_ok &= check(2, 4, N, 64, causal)
    print("[PASS] forward correctness" if all_ok else "[FAIL]")

    print("\n=== backward 正确性（dq/dk/dv vs fp32 autograd） ===")
    all_b = True
    for N in [128, 256]:
        for causal in [False, True]:
            all_b &= check_backward(1, 2, N, 64, causal)
    print("[PASS] backward correctness" if all_b else "[FAIL]")

    print("\n=== 基准 vs torch.sdp（do_bench） ===")
    print(f"{'N':>6} {'flash(ms)':>12} {'torch.sdp(ms)':>13} {'speedup':>8}")
    for N in [512, 1024, 2048, 4096]:
        B, H, D = 4, 8, 128
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        t_flash = triton.testing.do_bench(lambda: flash_attention(q, k, v, causal=True))
        t_torch = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True))
        print(f"{N:>6} {t_flash*1000:>9.2f}ms {t_torch*1000:>10.2f}ms {t_torch/t_flash:>6.2f}x")
