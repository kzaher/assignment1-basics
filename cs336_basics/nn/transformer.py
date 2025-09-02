from torch import nn
from cs336_basics.nn import multi_head_self_attention
from cs336_basics.nn import nonlinear
from cs336_basics.nn import nonlinear_conjunction
from cs336_basics.nn import nonlinear_mixture
from cs336_basics.nn import rms_norm
from cs336_basics.nn import dyt
import torch
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.pretraining import configuration
from cs336_basics.nn import sigmoid


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
        self.ln1 = rms_norm.RmsNorm(d_model=d_model, device=device, dtype=dtype)
        d_alpha = {"dyt": 1, "dyt_full": d_model}.get(experiments.rms_norm, None)
        self.lnd1 = (
            dyt.DyT(d_model=d_model, d_alpha=d_alpha, device=device, dtype=dtype)
            if d_alpha is not None
            else None
        )
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
        self.ln2 = rms_norm.RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.lnd2 = (
            dyt.DyT(d_model=d_model, d_alpha=d_alpha, device=device, dtype=dtype)
            if d_alpha is not None
            else None
        )
        self.guard_sigmoid1 = sigmoid.SigmoidCustomGrad(
            mu=0,
            s=0.5,
            # Prevent gradient exploding
            max_grad=0.01,
        )
        self.guard_sigmoid2 = sigmoid.SigmoidCustomGrad(
            mu=0,
            s=0.5,
        )
        if experiments.ff_type is None:
            self.ffn = nonlinear.SwiGlu(
                d_model=d_model,
                d_ff=d_ff,
                use_bias=use_bias,
                device=device,
                dtype=dtype,
            )
        elif experiments.ff_type == "silu":
            self.ffn = nonlinear.SiLU()
        elif experiments.ff_type == "relu_soft":
            assert experiments.ff_relu_squeeze_factor
            assert experiments.ff_relu_min
            self.ffn = nonlinear.ReluSoft(
                d_model=d_model,
                d_ff=d_ff,
                squeeze_factor=experiments.ff_relu_squeeze_factor,
                min_gradient=experiments.ff_relu_min,
                use_bias=use_bias,
                device=device,
                dtype=dtype,
            )
        elif experiments.ff_type == "mixture":
            assert experiments.enabled_nonlinear
            self.ffn = nonlinear_mixture.MixtureOfNonlinearFeedForward(
                d_model=d_model,
                # To normalize for 2 projections
                d_ff=d_ff * 3 // 2,
                use_bias=use_bias,
                enabled=experiments.enabled_nonlinear,
                device=device,
                dtype=dtype,
            )
        elif experiments.ff_type == "conjunction":
            assert experiments.and_group_size
            self.ffn = nonlinear_conjunction.ConjunctionFeedForward(
                d_model=d_model,
                use_bias=use_bias,
                d_ff=d_ff,
                and_group_size=experiments.and_group_size,
                device=device,
                dtype=dtype,
            )
        else:
            raise Exception(f"ff_type is unknown: {experiments.ff_type}")
        self.experiments = experiments

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_model"],
        token_positions: Int[Tensor, "... sequence_length d_model"] | None = None,
    ) -> Float[Tensor, "... sequence_length d_model"]:
        if self.experiments.rms_norm is None or self.experiments.rms_norm == "post":
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
        elif (
            self.experiments.rms_norm == "dyt"
            or self.experiments.rms_norm == "dyt_full"
        ):
            assert self.lnd1
            assert self.lnd2
            attention_output: Float[Tensor, "... sequence_length d_model"] = (
                x
                + self.attn(
                    self.lnd1(x),
                    token_positions=(
                        torch.arange(x.size(-2))
                        if token_positions is None
                        else token_positions
                    ),
                )
            )
            return attention_output + self.ffn(self.lnd2(attention_output))
        elif self.experiments.rms_norm == "pre":
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
        elif self.experiments.rms_norm == "remove":
            attention_output: Float[Tensor, "... sequence_length d_model"] = (
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
            return attention_output + self.ffn(attention_output)
        elif self.experiments.rms_norm == "guard_attention":
            attention_output: Float[Tensor, "... sequence_length d_model"] = x + (
                2
                * self.guard_sigmoid2(
                    self.attn(
                        2 * self.guard_sigmoid1(x) - 1,
                        token_positions=(
                            torch.arange(x.size(-2))
                            if token_positions is None
                            else token_positions
                        ),
                    )
                )
                - 1
            )
            return attention_output + self.ffn(attention_output)
        else:
            raise Exception(f"Unknown rms norm: {self.experiments.rms_norm}")
