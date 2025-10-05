import torch
from torch import nn
from cs336_basics.nn import linear
from cs336_basics.nn import atan
from cs336_basics.nn import extensions
from jaxtyping import Float


class ConjunctionFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_bias: bool,
        max_gradient_guard: float = 30.0,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = linear.Linear(
            in_features=d_model,
            out_features=3 * d_ff // 2,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )
        self.w3 = linear.Linear(
            in_features=3 * d_ff // 2 + d_model,
            out_features=d_model,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )

        self.and_or_ternary1 = atan.AtanTernary(dtype=dtype, device=device)
        # self.and_or_ternary2 = atan.AtanTernary(max_grad=experiments.input_gradient_guard, dtype=dtype, device=device)

        self.guard = atan.Atan(d_model=d_model, max_grad=max_gradient_guard, dtype=dtype, device=device)

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        safe_x = self.guard(x)
        return extensions.compose(
            self.w1,
            self.and_or_ternary1,
            lambda x: torch.concat([x, safe_x], dim=-1),
            self.w3,
        )(safe_x)
