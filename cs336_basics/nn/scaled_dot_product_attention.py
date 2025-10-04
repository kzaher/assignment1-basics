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

    def forward(
        self,
        Q: Float[Tensor, " ... seq_length d_k"],
        K: Float[Tensor, " ... seq_length d_k"],
        V: Float[Tensor, " ... seq_length d_v"],
        mask: Bool[Tensor, " ... seq_length seq_length"] | None = None,
    ) -> Float[Tensor, " ... seq_length d_v"]:
        search: Float[Tensor, " ... seq_length seq_length"] = (Q @ K.mT) / math.sqrt(Q.size(-1))
        if mask is not None:
            search = torch.where(~mask, -torch.inf, search)
        return self.softmax(search) @ V
