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
        configuration: configuration.TransformerLlmConfiguration,
        packed_rope = False
    ):
        super().__init__()

        d_model=configuration.d_model
        use_bias=configuration.use_bias
        experiments=configuration.experiments
        device=configuration.device
        dtype=configuration.dtype

        self.ln1 = experiments.create_default_normalization_layer(d_model=d_model, device=device, dtype=dtype)
        self.attn = multi_head_self_attention.MultiHeadSelfAttention(
            d_model=d_model,
            num_query_heads=configuration.num_query_heads,
            num_key_value_heads=configuration.num_key_value_heads,
            d_head=configuration.d_head,
            max_sequence_length=configuration.max_sequence_length,
            theta=configuration.rope_theta,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
            experiments=experiments,
            packed_rope=packed_rope
        )
        self.ln2 = experiments.create_default_normalization_layer(d_model=d_model, device=device, dtype=dtype)
        self.ffn = experiments.create_ffn(
            d_model=d_model,
            d_hidden=configuration.d_hidden,
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
                token_positions=token_positions,
            )
        )
        return attention_output + self.ffn(self.ln2(attention_output))
