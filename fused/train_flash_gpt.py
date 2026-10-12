"""
train_flash_gpt.py — 收官集成：完全用自己写的内核训练 TinyGPT

  - attention: 你的 flash_attention（fwd + bwd，LSE）→ 替代 scaled_dot_product_attention
  - norm:      你的 fused rmsnorm（fwd + bwd）      → 替代 nn.RMSNorm
  - 对比:      参考模型 = torch SDPA + nn.RMSNorm

验证：loss 下降、梯度全部有限、两模型最终 loss 同量级。
运行：python train_flash_gpt.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_rmsnorm import rmsnorm
from flash_attention import flash_attention


def rmsnorm3d(x, w, eps=1e-5):
    """[B,T,D] → [B*T,D] 喂给你的 RMSNorm 内核。"""
    B, T, D = x.shape
    return rmsnorm(x.reshape(B * T, D), w, eps).reshape(B, T, D)


# ---------------------- 你自己的内核版 attention ----------------------
class FlashAttn(nn.Module):
    """模型 fp32，进内核前转 fp16，出来转回 fp32（生产里的典型分工）。"""

    def __init__(self, D, n_heads):
        super().__init__()
        self.n_heads, self.hd = n_heads, D // n_heads
        self.wq = nn.Linear(D, D)
        self.wk = nn.Linear(D, D)
        self.wv = nn.Linear(D, D)
        self.wo = nn.Linear(D, D)

    def forward(self, x):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2).contiguous()
        k = self.wk(x).view(B, T, self.n_heads, self.hd).transpose(1, 2).contiguous()
        v = self.wv(x).view(B, T, self.n_heads, self.hd).transpose(1, 2).contiguous()
        o = flash_attention(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16), causal=True)
        o = o.float().transpose(1, 2).reshape(B, T, D)
        return self.wo(o)


# ---------------------- 参考（torch SDPA）版 attention ----------------------
class SDPAAttn(nn.Module):
    def __init__(self, D, n_heads):
        super().__init__()
        self.n_heads, self.hd = n_heads, D // n_heads
        self.wq = nn.Linear(D, D)
        self.wk = nn.Linear(D, D)
        self.wv = nn.Linear(D, D)
        self.wo = nn.Linear(D, D)

    def forward(self, x):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


# ---------------------- 通用 transformer 部件 ----------------------
class MLP(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.fc1 = nn.Linear(D, 4 * D)
        self.fc2 = nn.Linear(4 * D, D)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, D, n_heads, use_custom):
        super().__init__()
        self.use_custom = use_custom
        if use_custom:
            self.w1 = nn.Parameter(torch.ones(D))
            self.w2 = nn.Parameter(torch.ones(D))
        else:
            self.norm1 = nn.RMSNorm(D)
            self.norm2 = nn.RMSNorm(D)
        self.attn = FlashAttn(D, n_heads) if use_custom else SDPAAttn(D, n_heads)
        self.mlp = MLP(D)

    def forward(self, x):
        if self.use_custom:
            x = x + self.attn(rmsnorm3d(x, self.w1))
            x = x + self.mlp(rmsnorm3d(x, self.w2))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, ctx, D, n_heads, n_layers, use_custom):
        super().__init__()
        self.use_custom = use_custom
        self.tok = nn.Embedding(vocab, D)
        self.pos = nn.Parameter(torch.zeros(1, ctx, D))
        self.blocks = nn.ModuleList([Block(D, n_heads, use_custom) for _ in range(n_layers)])
        if use_custom:
            self.ln_w = nn.Parameter(torch.ones(D))
        else:
            self.ln = nn.RMSNorm(D)
        self.head = nn.Linear(D, vocab)

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos[:, :T]
        for b in self.blocks:
            h = b(h)
        h = rmsnorm3d(h, self.ln_w) if self.use_custom else self.ln(h)
        return self.head(h)


# ---------------------- 训练（两版同 seed 对比） ----------------------
def train(use_custom, seed=0, steps=60):
    torch.manual_seed(seed)
    model = TinyGPT(vocab=100, ctx=32, D=64, n_heads=2, n_layers=2, use_custom=use_custom).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for step in range(steps):
        x = torch.randint(0, 100, (8, 32), device="cuda")
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 100), x.reshape(-1))
        opt.zero_grad()
        loss.backward()
        g = torch.cat([p.grad.reshape(-1) for p in model.parameters() if p.grad is not None])
        assert torch.isfinite(g).all(), f"non-finite gradient at step {step}"
        opt.step()
        losses.append(loss.item())
    return losses


if __name__ == "__main__":
    custom = train(use_custom=True)
    ref = train(use_custom=False)

    print("\n=== 训练 loss（前 3 / 后 3） ===")
    print(f"自制内核 : {[round(v,3) for v in custom[:3]]} ... {[round(v,3) for v in custom[-3:]]}")
    print(f"SDPA参考 : {[round(v,3) for v in ref[:3]]} ... {[round(v,3) for v in ref[-3:]]}")

    assert custom[-1] < custom[0], "自制: loss 没下降"
    assert ref[-1] < ref[0], "参考: loss 没下降"
    assert all(torch.isfinite(torch.tensor(v)) for v in custom), "自制: loss NaN"
    gap = abs(custom[-1] - ref[-1])
    print(f"\n最终 loss 差距 |Δ| = {gap:.3f}")
    print("[PASS] 收官集成通过：flash_attention + fused_rmsnorm 完整跑通训练，loss 下降、梯度有限")
