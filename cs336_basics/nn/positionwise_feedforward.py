from torch import nn
import torch
import math
from jaxtyping import Float

class SwiGlu(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: torch.types.Device = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype, device=device))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), dtype=dtype, device=device))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype, device=device))
        sigma = math.sqrt(2.0 / (d_model + d_ff))
        nn.init.trunc_normal_(self.w1, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.w2, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
        nn.init.trunc_normal_(self.w3, mean=0, std=sigma, a=-3*sigma, b=3*sigma)
    
    def forward(self, x: Float[torch.Tensor, "... d_model"]) ->Float[torch.Tensor, "... d_model"]:
        w1x = torch.einsum('fm,...m->...f', self.w1, x)
        return torch.einsum(
          '...f,mf->...m',
            w1x * torch.sigmoid(w1x) * torch.einsum('fm,...m->...f', self.w3, x),
            self.w2
        )

