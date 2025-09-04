import dataclasses
from cs336_basics import serialization


@dataclasses.dataclass(frozen=True)
class AdamWOptimizerConfiguration:
    lr: float
    weight_decay: float
    betas: list[float]
    eps: float


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
    enabled_nonlinear: list[str] | None = None
    activation: str | None = None
    and_group_size: int | None = None
    input_gradient_guard: float | None = None
    output_gradient_guard: float | None = None

@dataclasses.dataclass(frozen=True)
class TransformerLlmConfiguration:
    vocab_size: int
    max_sequence_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str
    dtype: str
    use_bias: bool
    experiments: ArchitectureExperiments

@dataclasses.dataclass(frozen=True)
class LlmPretrainingTrainingLoopConfiguration:
    name: str
    training_data_path: str
    validation_data_path: str
    checkpoint_persist_modulus: int
    adamw_optimizer_configuration: AdamWOptimizerConfiguration
    transformer_llm: TransformerLlmConfiguration
    annealing_configuration: AnnealingConfiguration
    batch_size: int
    context_length: int
    initial_max_l2_norm: float
    max_iterations: int | None = None
    time_limit_in_seconds: int | None = None

    @classmethod
    def from_dict(cls, object: dict) -> "LlmPretrainingTrainingLoopConfiguration":
        return serialization.from_dict(cls, object)


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
        return f'{experiment_output_path}/configuration.json'
    
    @property
    def output_metadata_path(self) -> str:
        return PretrainingConfiguration.get_output_metadata_path(self.experiment_output_path)

    @classmethod
    def from_dict(cls, object: dict) -> "PretrainingConfiguration":
        return serialization.from_dict(cls, object)