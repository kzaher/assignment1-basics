from torch import nn
import torch
import math
from jaxtyping import Float
import typing


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int | typing.Tuple[int, ...],
        use_bias: bool,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        complete_dimension = tuple(out_features) if isinstance(out_features, (tuple, list)) else (out_features,)
        out_features = out_features[-1] if isinstance(out_features, (tuple, list)) else out_features
        self.dimension_identifiers = 'abcdefgh'[:len(complete_dimension)]
        assert len(self.dimension_identifiers) == len(complete_dimension)
        self.weight = nn.Parameter(
            torch.empty(size=complete_dimension + (in_features,), device=device, dtype=dtype)
        )
        sigma = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        self._use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(
                torch.empty(size=complete_dimension, device=device, dtype=dtype)
            )
            torch.nn.init.uniform_(
                self.bias, a=-sigma, b=sigma
            )

    def forward(
        self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        x = torch.einsum(f"...i,{self.dimension_identifiers}i->...{self.dimension_identifiers}", x.contiguous(), self.weight)
        if self._use_bias:
            x = self.bias + x
        return x