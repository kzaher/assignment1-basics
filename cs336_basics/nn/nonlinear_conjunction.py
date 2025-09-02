import torch
from torch import nn
from cs336_basics.nn import linear
from cs336_basics.nn import sigmoid
from jaxtyping import Float
import einops


class ConjunctionFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        use_bias: bool,
        and_group_size: int,
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
        self.w3 = linear.Linear(
            in_features=d_ff,
            out_features=d_model,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )

        self.and_or_ternary1 = sigmoid.SigmoidTernary(
            mu=1 * 10,
            s=0.5
        )

        self.and_or_ternary2 = sigmoid.SigmoidTernary(
            mu=1 * 10,
            s=0.5
        )

        self.guard_sigmoid = sigmoid.SigmoidCustomGrad(
            mu=0,
            s=0.5,
        )

        self.and_group_size = and_group_size

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        safe_x = self.w1(2 * self.guard_sigmoid(x) - 1)
        layer_1_output = self.and_or_ternary1(safe_x)
        # grouped_for_and = einops.rearrange(
        #     projected_x, "... (d g) -> ... d g", g=self.and_group_size
        # )
        return self.and_or_ternary2(self.w3(layer_1_output))
