"""
fused_rmsnorm.py — 最小闭环启动套件

任务：把下面两个内核的 TODO 补完（这是你负责的部分）。
host / autograd / 对拍 / 基准都已写好，直接可跑。

约定：
  - x 形状 [B, D]（2 维），w 形状 [D]，D 不一定是 2 的幂。
  - RMSNorm 数学见 README，backward 的难点是那个跨行的 dot 归约。
  - 提交前必须通过 test_rmsnorm.py 全部用例。

RMSNorm forward:
    rms = sqrt( mean(x^2) + eps )            # 沿最后一维 D
    y_i = x_i / rms * w_i

RMSNorm backward（建议实现，标量 r = 1/rms）:
    dot = sum_j( dy_j * x_j * w_j )          # 沿 D 归约，标量
    dx_i = ( w_i * dy_i - x_i * dot * r^2 / D ) * r
    dw_i = sum_rows( dy_i * x_i * r )        # 跨行归约 → 用 atomic_add 累加
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# 1) 前向内核（你来写）
# --------------------------------------------------------------------------
@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,          # [B, D] 输入
    w_ptr,          # [D]   可学习权重
    y_ptr,          # [B, D] 输出
    n_cols,         # D（运行时值）
    eps,            # 数值稳定常数
    stride_row,     # x 的行 stride（通常 = D）
    BLOCK: tl.constexpr,   # = next_power_of_2(D)
):
    # 提示：
    #   row = tl.program_id(0)
    #   row_ptr = x_ptr + row * stride_row
    #   cols = tl.arange(0, BLOCK); mask = cols < n_cols
    #   x = tl.load(row_ptr + cols, mask=mask, other=0.0)

    pid = tl.program_id(axis = 0)
    offset = tl.arange(0,BLOCK)
    mask = offset < n_cols 
    x = tl.load(x_ptr + pid * stride_row + offset , mask = mask ,other = 0.0 )
    ms = tl.sum(x*x ,axis = 0)/n_cols 
    rm = tl.sqrt(ms + eps)
    w = tl.load(w_ptr + offset , mask = mask , other = 0.0)
    y = x / rm * w
    tl.store(y_ptr + pid * stride_row + offset , y , mask = mask)   

    #   # 注意：mask 填 0 后，sum(x^2) 除以的是 n_cols（真实个数），不是 BLOCK
    #   ms = tl.sum(x * x, axis=0) / n_cols
    #   rms = tl.sqrt(ms + eps)
    #   w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    #   y = x / rms * w
    #   tl.store(y_ptr + row * stride_row + cols, y, mask=mask)
    pass


# --------------------------------------------------------------------------
# 2) 反向内核（你来写，这是难点）
# --------------------------------------------------------------------------
@triton.jit
def _rmsnorm_bwd_kernel(
    dy_ptr,         # [B, D] 上游梯度
    x_ptr,          # [B, D] 前向保存的输入
    w_ptr,          # [D]   前向保存的权重
    dx_ptr,         # [B, D] 输出：对 x 的梯度
    dw_ptr,         # [D]   输出：对 w 的梯度（传入前已清零，用 atomic_add 累加）
    n_cols,
    eps,
    stride_row,
    BLOCK: tl.constexpr,
):
    # 提示：
    #   row = tl.program_id(0)
    #   load dy, x（同 row）；w（同 row）
    #   ms = tl.sum(x * x, axis=0) / n_cols
    #   r2 = 1.0 / (ms + eps)              # r^2
    #   r = tl.sqrt(r2)                    # 1/rms
    pid = tl.program_id(0)
    offset = tl.arange(0,BLOCK)
    mask = offset < n_cols 
    dy = tl.load(dy_ptr + pid * stride_row + offset , mask = mask , other = 0.0 )
    x = tl.load(x_ptr + pid * stride_row + offset , mask = mask ,other = 0.0 )
    w = tl.load(w_ptr + offset , mask = mask , other = 0.0)
    ms = tl.sum(x*x ,axis = 0)/n_cols
    r2 = 1.0/(ms + eps)
    r = tl.sqrt(r2)     
    dot = tl.sum(dy * w * x , axis = 0)
    dx = (w * dy - x * dot * r2 / n_cols) * r
    tl.store(dx_ptr + pid * stride_row + offset , dx , mask = mask)
    tl.atomic_add(dw_ptr + offset , dy * x * r , mask = mask)
    
    #   dot = tl.sum(dy * w * x, axis=0)   # 跨行内归约，标量
    #   dx = (w * dy - x * dot * r2 / n_cols) * r
#   tl.store(dx_ptr + row * stride_row + cols, dx, mask=mask)
    #   # dw：每行贡献 dy * x * r，跨行归约 → atomic_add 到 dw_ptr
    #   tl.atomic_add(dw_ptr + cols, dy * x * r, mask=mask)
    pass


# --------------------------------------------------------------------------
# 3) host 包装 + autograd（已写好）
# --------------------------------------------------------------------------
class RMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps=1e-5):
        x = x.contiguous()
        w = w.contiguous()
        B, D = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D)
        grid = (B,)
        _rmsnorm_fwd_kernel[grid](
            x, w, y, D, eps, x.stride(0),
            BLOCK=BLOCK,
        )
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        ctx.D = D
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        B, D = x.shape
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        dw = torch.zeros_like(w)
        BLOCK = triton.next_power_of_2(D)
        grid = (B,)
        _rmsnorm_bwd_kernel[grid](
            dy, x, w, dx, dw, D, ctx.eps, x.stride(0),
            BLOCK=BLOCK,
        )
        return dx, dw, None  # 对应 (x, w, eps)


def rmsnorm(x, w, eps=1e-5):
    """可微分的 RMSNorm：y = rmsnorm(x, w)。直接可 torch.compile / 训练用。"""
    return RMSNormFunc.apply(x, w, eps)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 512, device="cuda", dtype=torch.float16)
    w = torch.randn(512, device="cuda", dtype=torch.float16)
    y = rmsnorm(x.float(), w.float())  # fwd+bwd 用 fp32 跑通，性能用 fp16
    print("forward shape:", y.shape, "sample:", y[0, :3].tolist())
