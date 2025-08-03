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
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = torch.tensor(eps, device=device, dtype=torch.float32)
        self.dtype = dtype
        self.d_model = d_model

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        x_at_least32 = x.to(torch.float32)
        rms_inverse = 1.0 / torch.sqrt(
            torch.einsum("...d,...d->...", x_at_least32, x_at_least32) / self.d_model
            + self.eps
        )
        return torch.einsum("...d,...,d->...d", x, rms_inverse, self.weight).to(self.dtype)
