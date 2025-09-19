from torch import nn
from cs336_basics.nn import transformer
from cs336_basics.nn import rms_norm
from cs336_basics.nn import embedding
from cs336_basics.nn import linear
from cs336_basics.nn import softmax
from cs336_basics.nn import atan
from cs336_basics.pretraining import configuration
import torch
from jaxtyping import Float, Int
from torch import Tensor


class TransformerLm(nn.Module):
    def __init__(
        self,
        configuration: configuration.TransformerLlmConfiguration,
    ):
        super().__init__()
        self.experiments = configuration.experiments
        d_model = configuration.d_model
        device = configuration.device
        dtype = configuration.dtype
        vocab_size = configuration.vocab_size
        use_bias = configuration.use_bias
        self.token_embeddings = embedding.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                transformer.TransformerBlock(configuration)
                for _ in range(configuration.num_layers)
            ]
        )
        self.ln_final = configuration.experiments.create_final_normalization_layer(
            d_model=d_model, device=device, dtype=dtype
        )
        self.lm_head = linear.Linear(
            in_features=d_model,
            out_features=vocab_size,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
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
