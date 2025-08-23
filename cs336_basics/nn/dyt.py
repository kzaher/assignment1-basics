from torch import nn
import torch


class DyT(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_alpha: int,
        alpha_init_value=0.5,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.ones(d_alpha, dtype=dtype, device=device) * alpha_init_value
        )
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=dtype, device=device))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
