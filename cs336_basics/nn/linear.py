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
        self.weight = nn.Parameter(
            torch.empty(size=(math.prod(complete_dimension), in_features,), device=device, dtype=dtype)
        )
        sigma = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma
        )
        self._complete_dimension = complete_dimension
        self._use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(
                torch.empty(size=(math.prod(complete_dimension),), device=device, dtype=dtype)
            )
            torch.nn.init.uniform_(
                self.bias, a=-sigma, b=sigma
            )

    def forward(
        self, x: Float[torch.Tensor, "... in_features"]
    ) -> Float[torch.Tensor, "... out_features"]:
        x =  x @ self.weight.mT
        if self._use_bias:
            x = self.bias + x
        if len(self._complete_dimension) > 1:
            x = x.view(*x.shape[:-1], *self._complete_dimension)
        return x