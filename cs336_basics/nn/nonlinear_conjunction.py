import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


ACTIVATIONS={
    "sigmoid": torch.sigmoid,
    "gelu": F.gelu,
    "silu": F.silu,
}

class HardMinSoftGrad(Function):
    @staticmethod
    def forward(ctx, x, dim=-1, tau=0.1):
        ctx.dim = dim
        ctx.tau = tau

        if ctx.needs_input_grad[0]:  # only compute/save if grad required
            w = torch.softmax(-x / tau, dim=dim)
            ctx.save_for_backward(w)
        else:
            ctx.save_for_backward(None)

        # Hard min forward
        y, _ = torch.min(x, dim=dim)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (w,) = ctx.saved_tensors
        if w is None:  # no gradient needed
            return None, None, None

        dim = ctx.dim
        grad = grad_out.unsqueeze(dim) * w
        return grad, None, None  # only return grad for `x`

def hardmin_softgrad(x, dim=-1, tau=0.1):
    return HardMinSoftGrad.apply(x, dim, tau)


class ConjunctionFeedForward(nn.Module):
    