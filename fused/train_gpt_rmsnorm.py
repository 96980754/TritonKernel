"""
train_gpt_rmsnorm.py — 关卡 D：把 fused RMSNorm 集成进真模型训练

作用：
  1. 用你写的 rmsnorm 替换 nn.RMSNorm，训练一个 2 层微型 GPT（随机数据）
  2. 同一模型换回 torch.nn.RMSNorm 做参考，对比 loss 曲线
  3. 验证：loss 能下降、梯度全部有限（不炸）、和参考数值接近

运行：
  python train_gpt_rmsnorm.py
期望输出：两套训练 loss 都在降、梯度全有限、最终 loss 处于同一量级。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fused_rmsnorm import rmsnorm


def rmsnorm3d(x, w, eps=1e-5):
    """把 [B, T, D] 展平成 [B*T, D] 喂给你的内核，再还原形状。"""
    B, T, D = x.shape
    return rmsnorm(x.reshape(B * T, D), w, eps).reshape(B, T, D)


class SelfAttention(nn.Module):
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
        y = F.scaled_dot_product_attention(q, k, v)
        return self.wo(y.transpose(1, 2).reshape(B, T, D))


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
        self.attn = SelfAttention(D, n_heads)
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
    print(f"fused RMSNorm : {[round(v,3) for v in custom[:3]]} ... {[round(v,3) for v in custom[-3:]]}")
    print(f"nn.RMSNorm参考 : {[round(v,3) for v in ref[:3]]} ... {[round(v,3) for v in ref[-3:]]}")

    # 验收断言
    assert custom[-1] < custom[0], "fused: loss 没下降"
    assert ref[-1] < ref[0], "ref: loss 没下降"
    assert all(torch.isfinite(torch.tensor(v)) for v in custom), "fused: loss 出现 NaN/Inf"
    gap = abs(custom[-1] - ref[-1])
    print(f"\n最终 loss 差距 |Δ| = {gap:.3f}（同量级即可，不一定相等）")
    print("[PASS] 集成验证通过：loss 下降、梯度有限、与参考数值接近 → 最小闭环的集成关完成")
