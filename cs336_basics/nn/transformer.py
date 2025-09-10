from torch import nn
from cs336_basics.nn import multi_head_self_attention
import torch
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.pretraining import configuration
from cs336_basics.nn import extensions


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_sequence_length: int,
        theta: float,
        use_bias: bool,
        experiments: configuration.ArchitectureExperiments = configuration.ArchitectureExperiments(),
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = experiments.create_default_normalization_layer(d_model=d_model, device=device, dtype=dtype)
        self.attn = multi_head_self_attention.MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_key=d_model,
            d_value=d_model,
            max_sequence_length=max_sequence_length,
            theta=theta,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
            experiments=experiments,
        )
        self.ln2 = experiments.create_default_normalization_layer(d_model=d_model, device=device, dtype=dtype)
        self.ffn = experiments.create_ffn(
            d_model=d_model,
            d_ff=d_ff,
            use_bias=use_bias,
            device=device,
            dtype=dtype)
        self.experiments = experiments

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length d_model"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        attention_output: Float[Tensor, "... sequence_length d_model"] = (
            x
            + self.attn(
                self.ln1(x),
                token_positions=(
                    torch.arange(x.size(-2))
                    if token_positions is None
                    else token_positions
                ),
            )
        )
        return attention_output + self.ffn(self.ln2(attention_output))
