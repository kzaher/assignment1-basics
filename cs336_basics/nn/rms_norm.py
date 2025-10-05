from typing import Any, Mapping
from torch import nn
import torch
from jaxtyping import Float

class RmsNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        self.eps = torch.tensor(eps, device=device, dtype=torch.float32)
        self.dtype = dtype
        self.d_model = d_model

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        x_at_least32 = x.to(torch.float32)
        return (
            self.weight
            * x_at_least32
            / torch.sqrt(
                torch.sum(x_at_least32.square(), dim=-1, keepdim=True)
                / self.d_model
                +self.eps
            )
        ).to(self.dtype)
