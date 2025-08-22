from torch import nn
from cs336_basics.nn import multi_head_self_attention
from cs336_basics.nn import nonlinear
from cs336_basics.nn import rms_norm
import torch
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.pretraining import configuration


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_sequence_length: int,
        theta: float,
        experiments: configuration.ArchitectureExperiments = configuration.ArchitectureExperiments(),
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = rms_norm.RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = multi_head_self_attention.MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_key=d_model,
            d_value=d_model,
            max_sequence_length=max_sequence_length,
            theta=theta,
            device=device,
            dtype=dtype,
            experiments=experiments
        )
        self.ln2 = rms_norm.RmsNorm(d_model=d_model, device=device, dtype=dtype)
        if experiments.ff_type == 'silu':
            self.ffn = nonlinear.SiLU()
        elif experiments.ff_type is None:
            self.ffn = nonlinear.SwiGlu(
                d_model=d_model, d_ff=d_ff, device=device, dtype=dtype
            )
        else:
            raise Exception(f'ff_type is unknown: {experiments.ff_type}')
        self.experiments = experiments

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length d_model"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        if self.experiments.rms_post_norm:
            attention_output = self.ln1(
                x
                + self.attn(
                    x,
                    token_positions=(
                        torch.arange(x.size(-2))
                        if token_positions is None
                        else token_positions
                    ),
                )
            )
            return self.ln2(attention_output + self.ffn(attention_output))

        attention_output: Float[Tensor, "... sequence_length d_model"] = x + self.attn(
            self.ln1(x),
            token_positions=(
                torch.arange(x.size(-2)) if token_positions is None else token_positions
            ),
        )
        return attention_output + self.ffn(self.ln2(attention_output))
