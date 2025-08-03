from torch import nn
import torch
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight: Float[torch.Tensor, "num_embeddings embedding_dim"] = (
            torch.nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
            )
        )
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(
        self, x: Int[torch.Tensor, "... seq_len"]
    ) -> Float[torch.Tensor, "... seq_len embedding_dim"]:
        return self.weight[x]
