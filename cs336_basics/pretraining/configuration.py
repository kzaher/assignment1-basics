import dataclasses
from cs336_basics import serialization

from cs336_basics.nn import nonlinear
from cs336_basics.nn import nonlinear_conjunction
from cs336_basics.nn import nonlinear_mixture
from cs336_basics.nn import rms_norm
from cs336_basics.nn import dyt
from cs336_basics.nn import atan
from cs336_basics.nn import parabola
from cs336_basics.nn import silu_monotonic
from cs336_basics.nn import muon
from cs336_basics.nn import adam_w
from cs336_basics import extensions
from torch.optim import adamw as torch_adamw
import re

import torch


@dataclasses.dataclass(frozen=True)
class AdamWOptimizerConfiguration:
    lr: float
    weight_decay: float
    betas: list[float]
    eps: float


@dataclasses.dataclass(frozen=True)
class MuonOptimizerConfiguration:
    lr: float
    momentum: float
    weight_decay: float


@dataclasses.dataclass(frozen=True)
class AnnealingConfiguration:
    zero_iters: int
    warmup_iters: int
    use_cosine_rampup: bool
    max_learning_rate: float
    min_learning_rate: float
    cosine_cycle_iters: int


@dataclasses.dataclass(frozen=True)
class ArchitectureExperiments:
    use_nope: bool | None = None
    rms_norm: str | None = None
    ff_type: str | None = None
    ff_relu_squeeze_factor: float | None = None
    ff_relu_min: float | None = None
    activation: str | None = None
    input_gradient_guard: float | None = None
    output_gradient_guard: float | None = None

    def create_final_normalization_layer(
        self, d_model: int, device: str, dtype: torch.dtype | None
    ):
        return self.create_default_normalization_layer(
            d_model=d_model, device=device, dtype=dtype
        )

    def create_default_normalization_layer(
        self, d_model: int, device: str, dtype: torch.dtype | None
    ):
        match self.rms_norm:
            case None:
                return rms_norm.RmsNorm(d_model=d_model, device=device, dtype=dtype)
            case "dyt" | "dyt_full":
                d_alpha = {"dyt": 1, "dyt_full": d_model}.get(self.rms_norm, None)
                assert d_alpha
                return dyt.DyT(
                    d_model=d_model, d_alpha=d_alpha, device=device, dtype=dtype
                )
            case "atan_learnable":
                return atan.Atan(
                    d_model=d_model, device=device, dtype=dtype, learnable_weight=True
                )
            case "atan":
                return atan.Atan(
                    d_model=d_model, device=device, dtype=dtype, learnable_weight=False
                )
            case _:
                raise Exception(f"Unknown rms_norm: {self.rms_norm}")

    def create_ffn(
        self,
        d_model: int,
        device: str,
        dtype: torch.dtype | None,
        d_hidden: int,
        use_bias: bool,
    ):
        match self.ff_type:
            case None:
                return nonlinear.ActivatedLu(
                    d_model=d_model,
                    d_ff=d_hidden,
                    use_bias=use_bias,
                    device=device,
                    dtype=dtype,
                    activation=nonlinear.Swish(),
                )
            case "silu":
                return nonlinear.SiLU()
            case "relu_soft":
                assert self.ff_relu_squeeze_factor
                assert self.ff_relu_min
                return nonlinear.ReluSoft(
                    d_model=d_model,
                    d_ff=d_hidden,
                    squeeze_factor=self.ff_relu_squeeze_factor,
                    min_gradient=self.ff_relu_min,
                    use_bias=use_bias,
                    device=device,
                    dtype=dtype,
                )
            case "parabola":
                return parabola.ParabolaGlu(
                    d_model=d_model,
                    use_bias=use_bias,
                    d_ff=d_hidden,
                    device=device,
                    dtype=dtype,
                )
            case "parabola_raised":
                return parabola.ParabolaGlu(
                    d_model=d_model,
                    use_bias=use_bias,
                    d_ff=d_hidden,
                    device=device,
                    dtype=dtype,
                    y_offset=1.0,
                )
            case "parabola_lowered":
                return parabola.ParabolaGlu(
                    d_model=d_model,
                    use_bias=use_bias,
                    d_ff=d_hidden,
                    device=device,
                    dtype=dtype,
                    y_offset=-1.0,
                )
            case "silu_monotonic_lu":
                return nonlinear.ActivatedLu(
                    d_model=d_model,
                    d_ff=d_hidden,
                    use_bias=use_bias,
                    device=device,
                    dtype=dtype,
                    activation=silu_monotonic.SiLUELUMonotonic(),
                )
            case "silu_relu":
                return nonlinear.ActivatedLu(
                    d_model=d_model,
                    d_ff=d_hidden,
                    use_bias=use_bias,
                    device=device,
                    dtype=dtype,
                    activation=silu_monotonic.SiLUReluMonotonic(),
                )
            case "relu":
                return nonlinear.ActivatedLu(
                    d_model=d_model,
                    d_ff=d_hidden,
                    use_bias=use_bias,
                    device=device,
                    dtype=dtype,
                    activation=torch.nn.ReLU(),
                )
            case _:
                raise Exception(f"ff_type is unknown: {self.ff_type}")


@dataclasses.dataclass(frozen=True)
class TransformerLlmConfiguration:
    vocab_size: int
    max_sequence_length: int
    d_model: int
    num_layers: int
    num_query_heads: int
    num_key_value_heads: int
    d_head: int
    d_hidden: int
    rope_theta: float
    device: str
    dtype_str: str
    use_bias: bool
    experiments: ArchitectureExperiments
    dtype: torch.dtype | None = None


@dataclasses.dataclass(frozen=True)
class LlmPretrainingTrainingLoopConfiguration:
    name: str
    training_data_path: str
    validation_data_path: str
    checkpoint_persist_modulus: int
    adamw_optimizer_configuration: AdamWOptimizerConfiguration
    muon_optimizer_configuration: MuonOptimizerConfiguration
    transformer_llm: TransformerLlmConfiguration
    annealing_configuration: AnnealingConfiguration
    batch_size: int
    context_length: int
    initial_max_l2_norm: float
    max_iterations: int | None = None
    time_limit_in_seconds: int | None = None
    write_checkpoint: bool = False
    optimizer_type: str | None = None
    muon_parameters_filter: str | None = None

    def create_optimizer(
        self, named_parameters
    ) -> tuple[torch.optim.Optimizer, dict[str, object]]:
        match self.optimizer_type:
            case "muon_with_adam" | "muon_with_adam_adam":
                def create_param(name: str, parameters: torch.Tensor):
                    assert self.muon_parameters_filter
                    assert self.muon_optimizer_configuration
                    if (
                        re.findall(self.muon_parameters_filter, name)
                        and self.optimizer_type == "muon_with_adam"
                    ):
                        return dict(
                            use_muon=True,
                            lr=self.muon_optimizer_configuration.lr,
                            momentum=self.muon_optimizer_configuration.momentum,
                            weight_decay=self.muon_optimizer_configuration.weight_decay,
                            params=parameters,
                        )
                    else:
                        return dict(
                            use_muon=False,
                            lr=self.adamw_optimizer_configuration.lr,
                            betas=self.adamw_optimizer_configuration.betas,
                            weight_decay=self.adamw_optimizer_configuration.weight_decay,
                            eps=1e-5,
                            params=parameters
                        )

                parameters = [
                    create_param(name, value)
                    for name, value in named_parameters
                ]
                exclude_keys_for_logging = {"params"}
                parameters_for_logging = [
                    {k: v for k, v in p.items() if k not in exclude_keys_for_logging}
                    for p in parameters
                ]
                return muon.SingleDeviceMuonWithAuxAdam(parameters), {
                    "params": parameters_for_logging
                }
            case "torch_adamw":
                adam_optimizer_configuration = self.adamw_optimizer_configuration
                assert len(adam_optimizer_configuration.betas) == 2
                return torch_adamw.AdamW(
                    named_parameters.values(),
                    lr=adam_optimizer_configuration.lr,
                    betas=(
                        adam_optimizer_configuration.betas[0],
                        adam_optimizer_configuration.betas[1],
                    ),
                    eps=adam_optimizer_configuration.eps,
                    weight_decay=adam_optimizer_configuration.weight_decay,
                ), {}
            case None:
                adam_optimizer_configuration = self.adamw_optimizer_configuration
                assert len(adam_optimizer_configuration.betas) == 2
                return (
                    adam_w.AdamW(
                        named_parameters.values(),
                        lr=adam_optimizer_configuration.lr,
                        weight_decay=adam_optimizer_configuration.weight_decay,
                        betas=(
                            adam_optimizer_configuration.betas[0],
                            adam_optimizer_configuration.betas[1],
                        ),
                        eps=adam_optimizer_configuration.eps,
                    ),
                    {},
                )
            case _:
                raise Exception("Unknown optimizer type")

        return

    @classmethod
    def from_dict(cls, object: dict) -> "LlmPretrainingTrainingLoopConfiguration":
        pretraining_configuration: LlmPretrainingTrainingLoopConfiguration = (
            serialization.from_dict(cls, object)
        )
        return extensions.replace_recursively(
            pretraining_configuration,
            lambda x: x.transformer_llm,
            transform=lambda transformer_llm: dataclasses.replace(
                transformer_llm, dtype=getattr(torch, transformer_llm.dtype_str)
            ),
        )


@dataclasses.dataclass(frozen=True)
class ParameterOverride:
    path: str
    float_values: list[float] | None
    int_values: list[int] | None
    string_values: list[str] | None
    bool_values: list[bool] | None
    string_list_values: list[list[str]] | None

    @property
    def values(self):
        if self.float_values is not None:
            return self.float_values
        elif self.int_values is not None:
            return self.int_values
        elif self.string_values is not None:
            return self.string_values
        elif self.bool_values is not None:
            return self.bool_values
        elif self.string_list_values is not None:
            return self.string_list_values
        else:
            raise Exception("Values need to be specified")


@dataclasses.dataclass(frozen=True)
class ParameterSweepConfiguration:
    values: list[ParameterOverride]

    @classmethod
    def from_dict(cls, object: dict) -> "ParameterSweepConfiguration":
        return serialization.from_dict(cls, object)


@dataclasses.dataclass(frozen=True)
class PretrainingConfiguration:
    output_path: str
    training_loop: LlmPretrainingTrainingLoopConfiguration
    checkpoint: int | None
    suffix: str | None = None

    @property
    def experiment_output_path(self):
        assert self.output_path
        assert self.training_loop.name
        suffix = self.suffix or ""
        if suffix:
            return f"{self.output_path}/{self.training_loop.name}/{suffix}"
        else:
            return f"{self.output_path}/{self.training_loop.name}"

    @property
    def vocabulary_path(self) -> str:
        assert self.training_loop.transformer_llm.vocab_size
        return f"{self.experiment_output_path}/vocabulary.{self.training_loop.transformer_llm.vocab_size}"

    @property
    def checkpoint_dir(self):
        return f"{self.experiment_output_path}"

    def cached_tokens(self, original_path: str) -> str:
        vocab_size = self.training_loop.transformer_llm.vocab_size
        assert vocab_size
        return f"{original_path}.tokens.vocab_size={vocab_size}.npy"

    @property
    def tokenized_training_data_path(self):
        return self.cached_tokens(self.training_loop.training_data_path)

    @property
    def tokenized_validation_data_path(self):
        return self.cached_tokens(self.training_loop.validation_data_path)

    @property
    def tokenizer_path(self) -> tuple[str, str]:
        vocab_size = self.training_loop.transformer_llm.vocab_size
        assert vocab_size
        return (
            f"{self.training_loop.training_data_path}.bpe-tokenizer.vocab_size={vocab_size}",
            f"{self.training_loop.training_data_path}.bpe-tokenizer.vocab_size={vocab_size}.merges",
        )

    def checkpoint_path(self, i: int) -> str:
        return f"{self.checkpoint_dir}/{i:09}.torch"

    def checkpoint_written_path(self, i: int) -> str:
        return f"{self.checkpoint_path(i)}.done"

    @classmethod
    def get_output_metadata_path(cls, experiment_output_path: str):
        return f"{experiment_output_path}/configuration.json"

    @property
    def output_metadata_path(self) -> str:
        return PretrainingConfiguration.get_output_metadata_path(
            self.experiment_output_path
        )

    @classmethod
    def from_dict(cls, object: dict) -> "PretrainingConfiguration":
        return serialization.from_dict(cls, object)
