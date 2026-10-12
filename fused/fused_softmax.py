"""
fused_softmax.py — 第二颗融合内核（L2 收尾）

任务：补完两个内核 TODO。host / autograd / 对拍 / 基准已写好。

约定：
  - s 形状 [B, D]（logits），D 不一定是 2 的幂。
  - 反向不需要重算 exp：forward 存下 y，backward 用 y 算 dot。

Softmax forward:
    m = max(s)                      # 沿 D
    p = exp(s - m)                  # 减 max 保证数值稳定
    l = sum(p)
    y = p / l

Softmax backward（标准公式）:
    dot = sum_j( g_j * y_j )        # 沿 D 归约，标量
    ds_i = y_i * ( g_i - dot )
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# 1) 前向内核（你来写）
# --------------------------------------------------------------------------
@triton.jit
def _softmax_fwd_kernel(
    s_ptr,          # [B, D] logits 输入
    y_ptr,          # [B, D] softmax 输出
    n_cols,         # D（运行时值）
    stride_row,     # s 的行 stride（通常 = D）
    BLOCK: tl.constexpr,
):
    # 提示：
    pid = tl.program_id(axis = 0)
    offset = tl.arange(0,BLOCK)
    mask = offset < n_cols
    s = tl.load(s_ptr + pid * stride_row + offset , mask = mask , other = -float('inf'))
    m = tl.max(s , axis = 0)
    p = tl.exp(s - m)
    l = tl.sum(p , axis = 0)
    y = p / l
    tl.store(y_ptr + pid * stride_row + offset , y , mask = mask)

    #   row = tl.program_id(0)
    #   cols = tl.arange(0, BLOCK); mask = cols < n_cols
    #   s = tl.load(s_ptr + row*stride_row + cols, mask=mask, other=-float('inf'))
    #      # 关键：mask 位填 -inf，这样 exp(-inf - m)=0，不会污染 l
    #   m = tl.max(s, axis=0)
    #   p = tl.exp(s - m)
    #   l = tl.sum(p, axis=0)
    #   y = p / l
    #   tl.store(y_ptr + row*stride_row + cols, y, mask=mask)



# --------------------------------------------------------------------------
# 2) 反向内核（你来写）
# --------------------------------------------------------------------------
@triton.jit
def _softmax_bwd_kernel(
    g_ptr,          # [B, D] 上游梯度
    y_ptr,          # [B, D] 前向保存的 softmax 输出
    ds_ptr,         # [B, D] 输出：对 logits 的梯度
    n_cols,
    stride_row,     # y 的行 stride
    BLOCK: tl.constexpr,
):

    pid = tl.program_id(axis = 0)
    offset = tl.arange(0,BLOCK)
    mask = offset < n_cols 
    g = tl.load(g_ptr + pid* stride_row + offset,mask = mask ,other = 0.0)
    y = tl.load(y_ptr + pid*stride_row + offset,mask = mask ,other = 0.0)
    dot = tl.sum(g*y,axis = 0)
    ds = y*(g - dot)
    tl.store(ds_ptr + pid*stride_row + offset, ds, mask=mask)

    # 提示：
    #   row = tl.program_id(0)
    #   g = tl.load(g_ptr + row*stride_row + cols, mask=mask, other=0.0)
    #   y = tl.load(y_ptr + row*stride_row + cols, mask=mask, other=0.0)
    #   dot = tl.sum(g * y, axis=0)         # 标量
    #   ds = y * (g - dot)
    #   tl.store(ds_ptr + row*stride_row + cols, ds, mask=mask)



# --------------------------------------------------------------------------
# 3) host 包装 + autograd（已写好）
# --------------------------------------------------------------------------
class SoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s):
        s = s.contiguous()
        B, D = s.shape
        y = torch.empty_like(s)
        BLOCK = triton.next_power_of_2(D)
        grid = (B,)
        _softmax_fwd_kernel[grid](s, y, D, s.stride(0), BLOCK=BLOCK)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        y, = ctx.saved_tensors
        B, D = y.shape
        g = g.contiguous()
        ds = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(D)
        grid = (B,)
        _softmax_bwd_kernel[grid](g, y, ds, D, y.stride(0), BLOCK=BLOCK)
        return ds


def softmax(s):
    """可微分的 fused softmax：y = softmax(s, dim=-1)。"""
    return SoftmaxFunc.apply(s)


if __name__ == "__main__":
    torch.manual_seed(0)
    s = torch.randn(8, 512, device="cuda", dtype=torch.float32) * 10  # 大 logits 考验稳定性
    y = softmax(s)
    print("forward shape:", y.shape, "row-sum≈1:", y[0].sum().item())
