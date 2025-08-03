from torch import nn
import torch
from jaxtyping import Float
from cs336_basics.nn import linear


class SwiGlu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = linear.Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.w2 = linear.Linear(
            in_features=d_ff, out_features=d_model, device=device, dtype=dtype
        )
        self.w3 = linear.Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.w1(x)
        return self.w2(
            w1x * torch.sigmoid(w1x) * self.w3(x),
        )
