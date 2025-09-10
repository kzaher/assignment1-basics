import torch
import torch.nn as nn
from cs336_basics.nn import linear
from jaxtyping import Float


class ParabolaCappedLinear(nn.Module):
    def __init__(self, a: float = 2.0, y_offset: float = 0):
        super().__init__()
        self.a = float(a)
        self.y_offset = float(y_offset)

    def forward(self, x):
        a = self.a
        # piecewise definition
        left = x
        mid = x - (x**2) / a
        right = a - x

        return torch.where(x < 0, left, torch.where(x < a, mid, right)) + self.y_offset


class ParabolaGlu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_bias: bool,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
        y_offset: float = 1.0,
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
        self.parabola = ParabolaCappedLinear(y_offset=y_offset)

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return self.w2(
            self.parabola(self.w1(x)) * self.w3(x),
        )
