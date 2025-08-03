from torch import nn
from cs336_basics.nn import multi_head_self_attention
from cs336_basics.nn import nonlinear
from cs336_basics.nn import rms_norm
import torch
from jaxtyping import Float, Int
from torch import Tensor


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
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
            max_seq_length=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = rms_norm.RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = nonlinear.SwiGlu(
            d_model=d_model, d_ff=d_ff, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length d_model"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        attention_output: Float[Tensor, "... sequence_length d_model"] = x + self.attn(
            self.ln1(x),
            token_positions=(
                torch.arange(x.size(-2)) if token_positions is None else token_positions
            ),
        )
        return attention_output + self.ffn(self.ln2(attention_output))
