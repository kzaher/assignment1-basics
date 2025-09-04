import torch
from torch import nn
from cs336_basics.nn import linear
from cs336_basics.nn import atan
from cs336_basics.nn import extensions
from cs336_basics.pretraining import configuration
from jaxtyping import Float
import einops
import functools


class ConjunctionFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_bias: bool,
        and_group_size: int,
        experiments: configuration.ArchitectureExperiments,
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

        self.guard = atan.Atan(max_grad=experiments.output_gradient_guard, dtype=dtype, device=device)

        self.and_group_size = and_group_size

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
