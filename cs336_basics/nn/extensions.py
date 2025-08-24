from torch import nn
from collections import abc
from typing import TypeVar
from collections import abc
import math
import torch
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided
import typing
import os
import wandb
import dataclasses
import torch

T = TypeVar("T")


def cosine_learning_rate(
    it: int,
    zero_iters: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    use_cosine_rampup=False,
) -> float:
    if it < zero_iters:
        return 0
    if it < warmup_iters:
        return (
            0.5
            * (1 - math.cos((it - zero_iters) / (warmup_iters - zero_iters) * math.pi))
            * max_learning_rate
            if use_cosine_rampup
            else (it - zero_iters) / (warmup_iters - zero_iters) * max_learning_rate
        )
    if it > cosine_cycle_iters:
        return min_learning_rate

    return min_learning_rate + 0.5 * (
        1
        + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
    ) * (max_learning_rate - min_learning_rate)


def gradient_clipping(
    parameters: abc.Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
) -> bool:
    return gradient_clipping_with_gradient_value(
        parameters=parameters, max_l2_norm=max_l2_norm, eps=eps
    )[0]


def gradient_clipping_with_gradient_value(
    parameters: abc.Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
) -> tuple[bool, float]:
    total_norm = torch.linalg.vector_norm(
        torch.tensor(
            [
                torch.linalg.vector_norm(parameter.grad).item()
                for parameter in parameters
                if parameter.grad is not None
            ]
        )
    ).item()
    if total_norm < max_l2_norm:
        return (False, total_norm)

    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter.grad.data = max_l2_norm / (total_norm + eps) * parameter.grad

    return (True, total_norm)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # print('dataset:', dataset)
    assert len(dataset.shape) == 1
    assert context_length <= dataset.shape[0]
    context_strided_dataset = as_strided(
        dataset,
        shape=(dataset.size - context_length + 1, context_length),
        strides=[dataset.itemsize, dataset.itemsize],
    )
    sample_indices = torch.randint(
        low=0, high=context_strided_dataset.shape[0] - 1, size=(batch_size,)
    )
    return (
        torch.stack(
            [torch.tensor(context_strided_dataset[i]) for i in sample_indices], dim=0
        ).to(device),
        torch.stack(
            [torch.tensor(context_strided_dataset[i + 1]) for i in sample_indices],
            dim=0,
        ).to(device),
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, typing.Any],
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": metadata,
        },
        out,
    )


def load_checkpoint(
    src: str, model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, object]:
    model_to_load = model._orig_mod if hasattr(model, "_orig_mod") else model
    state = torch.load(src)
    model_to_load.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["metadata"]


@dataclasses.dataclass
class HistogramRecorderResult:
    histogram: wandb.Histogram
    mean: float
    std: float


class HistogramRecorder:
    def __init__(self, resolution: int = 101, min=-1, max=1):
        self._cached_range = {}
        self._resolution = resolution
        self._min = -1
        self._max = 1

    @torch.no_grad()
    @torch._dynamo.disable   # <— keep this OUT of the compiled graph
    def calculate_histogram(self, name: str, x: torch.Tensor) -> wandb.Histogram:
        x = x.detach()

        mean = x.mean()
        std = x.std()

        max_limit = max(abs(mean + 3 * std), abs(mean - 3 * std), 0.1)

        self._cached_range[name] = (
            self._cached_range.get(name, 3.0) * 0.99 + 0.01 * max_limit.cpu().item()
        )

        x = 1 / self._cached_range[name] * x

        return HistogramRecorderResult(
            histogram=wandb.Histogram(
                np_histogram=(
                    torch.histc(
                        x.to(torch.float32),
                        min=self._min,
                        max=self._max,
                        bins=self._resolution,
                    )
                    .cpu()
                    .numpy(),
                    np.linspace(
                        self._min * self._cached_range[name],
                        self._max * self._cached_range[name],
                        self._resolution + 1,
                    ),
                )
            ),
            mean=mean,
            std=std,
        )


class PendingActivationRecording:
    def __init__(
        self,
        hooks: list[object],
        input_activations: dict[str, HistogramRecorderResult],
        output_activations: dict[str, HistogramRecorderResult],
    ):
        self._hooks = hooks
        self._input_activations = input_activations
        self._output_activations = output_activations

    @property
    def input_activations(self):
        return self._input_activations

    @property
    def output_activations(self):
        return self._output_activations

    @property
    def output_activations_std(self, order: list[str]):
        return [self._output_activations[name].std for name in order]

    @property
    def activation_histograms(self) -> dict[str, wandb.Histogram]:
        return {
            f"activation/input/histogram/{name}": activation.histogram
            for name, activation in recorder.input_activations.items()
        } | {
            f"activation/output/histogram/{name}": activation.histogram
            for name, activation in recorder.output_activations.items()
        }

    @property
    def activation_std(self):
        return {
            f"activation/input/std/{name}": result.std
            for name, result in self.input_activations.items()
        } | {
            f"activation/output/std/{name}": result.std
            for name, result in self.output_activations.items()
        }

    def remove_all(self):
        for hook in self._hooks:
            hook.remove()


class PendingActivationRecording:
    def __init__(
        self,
        hooks: list[object],
        input_activations: dict[str, HistogramRecorderResult],
        output_activations: dict[str, HistogramRecorderResult],
    ):
        self._hooks = hooks
        self._input_activations = input_activations
        self._output_activations = output_activations

    @property
    def input_activations(self):
        return self._input_activations

    @property
    def output_activations(self):
        return self._output_activations

    @property
    def activation_histograms(self) -> dict[str, wandb.Histogram]:
        return {
            f"activation/input/histogram/{name}": activation.histogram
            for name, activation in self.input_activations.items()
        } | {
            f"activation/output/histogram/{name}": activation.histogram
            for name, activation in self.output_activations.items()
        }

    @property
    def activation_std(self):
        return {
            f"activation/input/std/{name}": result.std
            for name, result in self.input_activations.items()
        } | {
            f"activation/output/std/{name}": result.std
            for name, result in self.output_activations.items()
        }

    def remove_all(self):
        for hook in self._hooks:
            hook.remove()


class ActivationRecorder:
    def __init__(self, module: nn.Module, filter_types: set[type] | None = None):
        self._module = module
        self._filter_types = filter_types
        self._input_activation_recorder = HistogramRecorder()
        self._output_activation_recorder = HistogramRecorder()

    def intercept_activations(self) -> PendingActivationRecording:
        input_activations = {}
        output_activations = {}
        hooks = []
        for name, module in self._module.named_modules():
            if not (self._filter_types is None or type(module) in self._filter_types):
                continue

            def register_activation(
                module: nn.Module, i: torch.Tensor, o: torch.Tensor, name=name
            ):
                input_activations[name] = (
                    self._input_activation_recorder.calculate_histogram(
                        name=name, x=i[0]
                    )
                )
                output_activations[name] = (
                    self._output_activation_recorder.calculate_histogram(
                        name=name, x=o[0]
                    )
                )

            hooks.append(module.register_forward_hook(register_activation))

        return PendingActivationRecording(
            hooks=hooks,
            input_activations=input_activations,
            output_activations=output_activations,
        )
