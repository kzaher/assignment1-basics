from torch import nn
from jaxtyping import Float
import torch


class Softmax(nn.Module):
    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
      normalized = torch.exp(x - x.max(dim=-1, keepdim=True).values)
      return normalized / normalized.sum(dim=-1, keepdim=True)
