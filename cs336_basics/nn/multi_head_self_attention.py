from cs336_basics.nn import scaled_dot_product_attention
from cs336_basics.nn import rope
from cs336_basics.nn import linear
from cs336_basics.pretraining import configuration
from torch import nn
import torch
from torch import Tensor
from jaxtyping import Float, Int
import einops


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_key: int,
        d_value: int,
        use_bias: bool,
        experiments: configuration.ArchitectureExperiments = configuration.ArchitectureExperiments(),
        max_sequence_length: int | None = None,
        theta: float | None = None,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        assert d_model % num_heads == 0
        if theta and max_sequence_length and not experiments.use_nope:
            self.rope = rope.Rope(
                theta=theta,
                d_k=d_model // num_heads,
                max_seq_len=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            self.rope = None
        self.scaled_dot_product_attention = (
            scaled_dot_product_attention.ScaledDotProductAttention()
        )
        self.q_proj = linear.Linear(
            in_features=d_model, out_features=d_key, use_bias=use_bias, device=device, dtype=dtype
        )
        self.k_proj = linear.Linear(
            in_features=d_model, out_features=d_key, use_bias=use_bias, device=device, dtype=dtype
        )
        self.v_proj = linear.Linear(
            in_features=d_model, out_features=d_value, use_bias=use_bias, device=device, dtype=dtype
        )
        self.output_proj = linear.Linear(
            in_features=d_value, out_features=d_model, use_bias=use_bias, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        Q = einops.rearrange(
            self.q_proj(x),
            "... sequence_length (head head_dim)->... head sequence_length head_dim",
            head=self.num_heads,
        )
        K = einops.rearrange(
            self.k_proj(x),
            "... sequence_length (head head_dim)->... head sequence_length head_dim",
            head=self.num_heads,
        )
        V = einops.rearrange(
            self.v_proj(x),
            "... sequence_length (head head_dim)->... head sequence_length head_dim",
            head=self.num_heads,
        )

        if self.rope and token_positions is not None:
            Q = self.rope(Q, token_positions=token_positions)
            K = self.rope(K, token_positions=token_positions)

        sequence_length = x.size(-2)
        causal_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), device=self.device)
        ).to(torch.bool)
        return self.output_proj(
            einops.rearrange(
                self.scaled_dot_product_attention(Q=Q, K=K, V=V, mask=causal_mask),
                "... head sequence_length per_head -> ... sequence_length (head per_head)",
            )
        )
