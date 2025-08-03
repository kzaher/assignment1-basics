from cs336_basics.nn import softmax
import torch
from torch import nn
from jaxtyping import Float, Bool
from torch import Tensor
import math

class ScaledDotProductAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.softmax = softmax.Softmax(dim=-1)

  def forward(self, Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:
    search = torch.einsum('...qd,...kd->...qk', Q, K)/math.sqrt(Q.size(-1))
    if mask is not None:
      search = torch.where(~mask, -torch.inf, search)
    return torch.einsum('...qk,...kd->...qd', self.softmax(search), V)
