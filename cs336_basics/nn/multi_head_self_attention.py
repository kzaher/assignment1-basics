from cs336_basics.nn import scaled_dot_product_attention
from cs336_basics.nn import rope
from cs336_basics.nn import linear
from cs336_basics.pretraining import configuration
from torch import nn
import torch
from torch import Tensor
from jaxtyping import Float, Int
import einops
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_key_value_heads: int,
        d_head: int,
        use_bias: bool,
        experiments: configuration.ArchitectureExperiments = configuration.ArchitectureExperiments(),
        max_sequence_length: int | None = None,
        theta: float | None = None,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.device = device
        self.d_head = d_head
        assert num_query_heads % num_key_value_heads == 0
        self.replicate_key_value_heads = num_query_heads // num_key_value_heads
        assert d_model % num_query_heads == 0
        if theta and max_sequence_length and not experiments.use_nope:
            self.rope = rope.Rope(
                theta=theta,
                d_k=d_head,
                max_seq_len=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            self.rope = None
        self.scaled_dot_product_attention = (
            scaled_dot_product_attention.ScaledDotProductAttention()
        )
        self.qkv_proj = linear.Linear(
            in_features=d_model,
            out_features=d_head * (num_query_heads + 2 * num_key_value_heads),
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )
        self.output_proj = linear.Linear(
            in_features=d_head * num_query_heads,
            out_features=d_model,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )

        if experiments.zero_output:
            self.output_proj.weight.detach().zero_()

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        input_proj = self.qkv_proj(x)
        q, k, v = torch.split(
            input_proj,
            (
                self.num_query_heads * self.d_head,
                self.num_key_value_heads * self.d_head,
                self.num_key_value_heads * self.d_head,
            ),
            dim=-1,
        )
        q_heads = einops.rearrange(
            q,
            "... sequence_length (head head_dim)->... head sequence_length head_dim",
            head=self.num_query_heads,
            head_dim=self.d_head,
        )
        k_heads = einops.rearrange(
            k,
            "... sequence_length (head head_dim)->... head sequence_length head_dim",
            head=self.num_key_value_heads,
            head_dim=self.d_head,
        )
        v_heads = einops.rearrange(
            v,
            "... sequence_length (head head_dim)->... head sequence_length head_dim",
            head=self.num_key_value_heads,
        )
        if self.replicate_key_value_heads > 1 or True:
            k_heads = torch.repeat_interleave(
                k_heads, self.replicate_key_value_heads, dim=-3
            )
            v_heads = torch.repeat_interleave(
                v_heads, self.replicate_key_value_heads, dim=-3
            )

        if self.rope and token_positions is not None:
            q_heads = self.rope(q_heads, token_positions=token_positions)
            k_heads = self.rope(k_heads, token_positions=token_positions)

        sequence_length = x.size(-2)
        causal_mask = torch.tril(
            torch.ones((sequence_length, sequence_length), device=self.device)
        ).to(torch.bool)
        return self.output_proj(
            einops.rearrange(
                self.scaled_dot_product_attention(
                    Q=q_heads, K=k_heads, V=v_heads, mask=causal_mask
                ),
                "... head sequence_length per_head -> ... sequence_length (head per_head)",
            )
        )
