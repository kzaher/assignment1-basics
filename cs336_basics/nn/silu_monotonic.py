import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLUELUMonotonicFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, emulate_relu):
        silu_out = input * torch.sigmoid(input)   # SiLU = Swish

        ctx.save_for_backward(input, silu_out)
        return silu_out if not emulate_relu else torch.clamp(input, min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, silu_out = ctx.saved_tensors

        # SiLU derivative: σ(x) + x * σ(x) * (1 - σ(x))
        sig = torch.sigmoid(input)
        silu_grad = sig + input * sig * (1 - sig)

        # Condition: if output < 0 and upstream gradient < 0 → use ELU gradient
        condition = (silu_grad < 0) & (grad_output < 0)

        grad_input = torch.where(condition, grad_output * -silu_grad, grad_output * silu_grad)
        return grad_input, None

class SiLUELUMonotonic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SiLUELUMonotonicFn.apply(x, False)


class SiLUReluMonotonic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SiLUELUMonotonicFn.apply(x, True)
