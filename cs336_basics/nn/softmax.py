from torch import nn
from jaxtyping import Float
import torch


class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
      super().__init__()
      self.dim = dim
    
    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
      normalized = torch.exp(x - x.max(dim=self.dim, keepdim=True).values)
      return normalized / normalized.sum(dim=self.dim, keepdim=True)
