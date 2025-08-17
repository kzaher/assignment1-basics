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
class TransformerLlmConfiguration:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str
    dtype: str


@dataclasses.dataclass(frozen=True)
class LlmPretrainingTrainingLoopConfiguration:
    name: str
    checkpoint_persist_modulus: int
    adamw_optimizer_configuration: AdamWOptimizerConfiguration
    transformer_llm: TransformerLlmConfiguration
    annealing_configuration: AnnealingConfiguration
    batch_size: int
    initial_max_l2_norm: float

    @classmethod
    def from_dict(cls, object: dict) -> "LlmPretrainingTrainingLoopConfiguration":
        return serialization.from_dict(cls, object)


@dataclasses.dataclass(frozen=True)
class PretrainingConfiguration:
    input_path: str
    output_path: str
    training_loop: LlmPretrainingTrainingLoopConfiguration
    checkpoint: int | None

    @property
    def vocabulary_path(self) -> str:
        assert self.training_loop.transformer_llm.vocab_size
        return f"{self.output_path}/vocabulary.{self.training_loop.transformer_llm.vocab_size}"

    @property
    def checkpoint_dir(self):
        return f"{self.tokenized_input_path}.checkpoints"

    @property
    def tokenized_input_path(self):
        assert self.output_path
        return f"{self.output_path}/input.tokens.npy"

    @property
    def tokenizer_path(self) -> tuple[str, str]:
        output_path = self.output_path
        vocab_size = self.training_loop.transformer_llm.vocab_size
        assert output_path
        assert vocab_size
        return (
            f"{output_path}/input.tokenizer.{vocab_size}",
            f"{output_path}/input.tokenizer.{vocab_size}.merges",
        )

    def checkpoint_path(self, i: int) -> str:
        return f"{self.checkpoint_dir}/{i:09}.torch"

    def checkpoint_written_path(self, i: int) -> str:
        return f"{self.checkpoint_path(i)}.done"
