from torch import nn
from cs336_basics.nn import transformer
from cs336_basics.nn import rms_norm
from cs336_basics.nn import embedding
from cs336_basics.nn import linear
from cs336_basics.nn import softmax
import torch
from jaxtyping import Float, Int
from torch import Tensor


class TransformerLm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = embedding.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                transformer.TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = rms_norm.RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = linear.Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype
        )

    def forward(
        self,
        in_indices: Int[Tensor, "...  batch_size sequence_length"],
        token_positions: Int[Tensor, "...  batch_size sequence_length"] | None = None,
    ) -> Float[Tensor, "... batch_size sequence_length vocab_size"]:
        propagate: Float[Tensor, "... batch_size sequence_length d_model"] = (
            self.token_embeddings(in_indices)
        )
        for layer in self.layers:
            propagate = layer(propagate, token_positions=token_positions)
        return self.lm_head(self.ln_final(propagate))
