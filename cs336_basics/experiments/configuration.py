import dataclasses
from cs336_basics import serialization
from __future__ import annotations

@dataclasses.dataclass(frozen=True)
class OptimizerConfiguration:
  lr: float
  weight_decay: float
  betas: list[float]
  eps: float

@dataclasses.dataclass(frozen=True)
class AnnealingConfiguration:
  max_learning_rate: float
  min_learning_rate: float
  warmup_iters: int
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

@dataclasses.dataclass(frozen=True)
class LlmPretrainingConfiguration:
  source_input_path: str
  configuration_path: str
  checkpoint_persist_modulus: int
  optimizer_configuration: OptimizerConfiguration
  transformer_llm: TransformerLlmConfiguration
  annealing_configuration: AnnealingConfiguration
  batch_size: int
  context_length: int
  max_l2_norm: float

  @property
  def checkpoint_dir(self):
    return f'{self.tokenized_input_path}.checkpoints'
  
  @property
  def tokenized_input_path(self):
    assert self.configuration_path
    return f'{self.configuration_path}.input.tokens'

  def checkpoint_path(self, i: int) -> str:
    return f"{self.checkpoint_dir}/{i:09}.torch"

  def checkpoint_written_path(self, i: int) -> str:
    return f"{self.checkpoint_path(i)}.done"

  @classmethod
  def from_dict(cls, object: dict) -> LlmPretrainingConfiguration:
    return serialization.from_dict(cls, object)