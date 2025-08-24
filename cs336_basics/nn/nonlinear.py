from torch import nn
import torch
from jaxtyping import Float
from cs336_basics.nn import linear
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


class SwiGlu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_bias: bool,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = linear.Linear(
            in_features=d_model,
            out_features=d_ff,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )
        self.w2 = linear.Linear(
            in_features=d_ff,
            out_features=d_model,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )
        self.w3 = linear.Linear(
            in_features=d_model,
            out_features=d_ff,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.w1(x)
        return self.w2(
            w1x * torch.sigmoid(w1x) * self.w3(x),
        )


class SiLU(nn.Module):
    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return x * torch.sigmoid(x)


class ReLUSoftGradFn(Function):
    @staticmethod
    def forward(ctx, x, squeeze_factor, min_gradient):
        # Save input and beta for backward
        ctx.save_for_backward(
            x,
            torch.tensor(squeeze_factor, dtype=x.dtype, device=x.device),
            torch.tensor(min_gradient, dtype=x.dtype, device=x.device),
        )
        # Standard ReLU forward
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, squeeze_factor, min_gradient = ctx.saved_tensors

        grad = torch.max(torch.sigmoid(squeeze_factor * x), min_gradient)

        # Chain rule
        grad_input = grad_output * grad

        # No gradient w.r.t. others
        return grad_input, None, None


class ReLUSoftGrad(nn.Module):
    def __init__(self, squeeze_factor: float = 10, min_gradient=0.01):
        super().__init__()
        self.squeeze_factor = squeeze_factor
        self.min_gradient = min_gradient

    def forward(self, x):
        # Pass beta as a second argument to the custom Function
        return ReLUSoftGradFn.apply(x, self.squeeze_factor, self.min_gradient)

class ReluSoft(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        squeeze_factor: float,
        min_gradient: float,
        use_bias: bool,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = linear.Linear(
            in_features=d_model, out_features=d_ff, use_bias=use_bias, device=device, dtype=dtype
        )
        self.w2 = linear.Linear(
            in_features=d_ff, out_features=d_model, use_bias=use_bias, device=device, dtype=dtype
        )
        self.w3 = linear.Linear(
            in_features=d_model, out_features=d_ff, use_bias=use_bias, device=device, dtype=dtype
        )
        self.ff = ReLUSoftGrad(squeeze_factor=squeeze_factor, min_gradient=min_gradient)

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.w1(x)
        return self.w2(
            self.ff(w1x) * self.w3(x),
        )