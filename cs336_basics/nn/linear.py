from torch import nn
import torch
import math
from jaxtyping import Float


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(size=(out_features, in_features), device=device, dtype=dtype)
        )
        sigma = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
        )

    def forward(
        self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        return torch.einsum("...i,oi->...o", x, self.weight)
