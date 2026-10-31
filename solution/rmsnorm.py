
import torch
import triton
import triton.language as tl


@triton.jit
def rms_fwd(X, W, Y, INV,
            H: tl.constexpr, EPS: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)

    x = tl.load(X + row * H + col, col < H, 0).to(tl.float32)
    w = tl.load(W + col, col < H, 0).to(tl.float32)

    mean_sq = tl.sum(x * x, 0) / H
    inv = tl.rsqrt(mean_sq + EPS)

    tl.store(Y + row * H + col, x * inv * w, col < H)
    tl.store(INV + row, inv)


@triton.jit
def rms_bwd(X, W, DY, INV, DX,
            H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)

    x = tl.load(X + row * H + col, col < H, 0).to(tl.float32)
    w = tl.load(W + col, col < H, 0).to(tl.float32)
    dy = tl.load(DY + row * H + col, col < H, 0).to(tl.float32)
    inv = tl.load(INV + row)

    g = dy * w
    dot = tl.sum(g * x, 0) / H
    dx = inv * (g - x * inv * inv * dot)

    tl.store(DX + row * H + col, dx, col < H)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        h = shape[-1]
        x2 = x.contiguous().view(-1, h)

        y = torch.empty_like(x2)
        inv = torch.empty(
            (x2.shape[0],), device=x.device, dtype=torch.float32
        )

        block = triton.next_power_of_2(h)
        rms_fwd[(x2.shape[0],)](
            x2, weight, y, inv, h, eps, block
        )

        ctx.save_for_backward(x2, weight, inv)
        ctx.shape = shape
        ctx.block = block
        return y.view(shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv = ctx.saved_tensors
        dy = grad_output.contiguous().view_as(x)
        dx = torch.empty_like(x)

        rms_bwd[(x.shape[0],)](
            x, weight, dy, inv, dx, x.shape[1], ctx.block
        )

        dw = (
            dy.float() * x.float() * inv[:, None]
        ).sum(dim=0).to(weight.dtype)

        return dx.view(ctx.shape), dw, None


def patch_rmsnorm():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            x = hidden_states.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            return self.weight * (
                x * torch.rsqrt(variance + self.variance_epsilon)
            ).to(hidden_states.dtype)

        return RMSNormFunction.apply(
            hidden_states, self.weight, self.variance_epsilon
        )

    cls.forward = forward
