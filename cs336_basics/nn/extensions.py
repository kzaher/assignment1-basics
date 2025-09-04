from torch import nn
from collections import abc
from typing import TypeVar
from collections import abc
import math
import torch
import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import as_strided
import typing
import os
import wandb
import dataclasses
import torch
from contextlib import contextmanager

T = TypeVar("T")


def compose(*args: typing.Callable[[T], T]) -> typing.Callable[[T], T]:
    def pipeline(x: T):
        for transform in args:
            x = transform(x)
        return x

    return pipeline


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
    name: str

    def logs(self, group: str):
        return {
            f"{group}/histogram/{self.name}": self.histogram,
            f"{group}/std/{self.name}": self.std,
            f"{group}/mean/{self.name}": self.mean,
        }


class HistogramRecorder:
    def __init__(self, resolution: int = 101, min=-1, max=1):
        self._cached_range = {}
        self._resolution = resolution
        self._min = -1
        self._max = 1

    @torch.no_grad()
    @torch._dynamo.disable  # <— keep this OUT of the compiled graph
    def calculate_histogram(self, name: str, x: torch.Tensor) -> wandb.Histogram:
        x = x.detach()

        mean = x.mean().cpu().item()
        std = x.std().cpu().item()

        max_limit = max(abs(mean + 3 * std), abs(mean - 3 * std), 1e-11)

        self._cached_range[name] = (
            self._cached_range.get(name, 3 * std) * 0.95 + 0.05 * max_limit
        )

        x = 1 / self._cached_range[name] * x

        return HistogramRecorderResult(
            name=name,
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


class PendingModuleActivationRecording:
    def __init__(
        self,
        hooks: list[object],
        input_activations: dict[str, HistogramRecorderResult],
        output_activations: dict[str, HistogramRecorderResult],
        input_gradient_activations: dict[str, HistogramRecorderResult],
        output_gradient_activations: dict[str, HistogramRecorderResult],
    ):
        self._hooks = hooks
        self._input_activations = input_activations
        self._output_activations = output_activations
        self._input_gradient_activations = input_gradient_activations
        self._output_gradient_activations = output_gradient_activations

    @property
    def logs(self) -> dict[str, wandb.Histogram]:
        return (
            {
                k: v
                for activation in self._input_activations.values()
                for k, v in activation.logs("activation/input").items()
            }
            | {
                k: v
                for activation in self._output_activations.values()
                for k, v in activation.logs("activation/output").items()
            }
            | {
                k: v
                for activation in self._input_gradient_activations.values()
                for k, v in activation.logs("gradient/input").items()
            }
            | {
                k: v
                for activation in self._output_gradient_activations.values()
                for k, v in activation.logs("gradient/output").items()
            }
        )

    def remove_all(self):
        for hook in self._hooks:
            hook.remove()


SUPPORTED_HISTOGRAM_TYPES = {torch.bfloat16, torch.float32, torch.float64}


class ModuleActivationRecorder:
    def __init__(self, module: nn.Module, filter_types: set[type] | None = None):
        self._module = module
        self._filter_types = filter_types
        self._input_activation_recorder = HistogramRecorder()
        self._output_activation_recorder = HistogramRecorder()
        self._input_gradient_recorder = HistogramRecorder()
        self._output_gradient_recorder = HistogramRecorder()

    @classmethod
    def _record_activation(
        cls,
        name: str,
        x: torch.Tensor,
        activations: dict[str, torch.Tensor],
        activation_recorder: HistogramRecorder,
    ):
        x = x if isinstance(x, torch.Tensor) else (x[0] if len(x) > 0 else None)
        if x is not None and x.dtype in SUPPORTED_HISTOGRAM_TYPES:
            activations[name] = activation_recorder.calculate_histogram(name=name, x=x)

    @contextmanager
    def intercept_activations(self, intercept=True):
        if not intercept:
            yield None
            return

        input_activations = {}
        output_activations = {}
        input_gradient_activations = {}
        output_gradient_activations = {}
        hooks = []
        for name, module in self._module.named_modules():
            if not (self._filter_types is None or type(module) in self._filter_types):
                continue

            @torch._dynamo.disable
            def register_activation(
                module: nn.Module, i: torch.Tensor, o: torch.Tensor, name=name
            ):
                ModuleActivationRecorder._record_activation(
                    name, i, input_activations, self._input_activation_recorder
                )
                ModuleActivationRecorder._record_activation(
                    name, o, output_activations, self._output_activation_recorder
                )

            @torch._dynamo.disable
            def gradient_activation(
                module: nn.Module, i: torch.Tensor, o: torch.Tensor, name=name
            ):
                ModuleActivationRecorder._record_activation(
                    name, i, input_gradient_activations, self._input_gradient_recorder
                )
                ModuleActivationRecorder._record_activation(
                    name, o, output_gradient_activations, self._output_gradient_recorder
                )

            hooks.append(module.register_forward_hook(register_activation))
            hooks.append(module.register_full_backward_hook(gradient_activation))

        activation_recording = PendingModuleActivationRecording(
            hooks=hooks,
            input_activations=input_activations,
            output_activations=output_activations,
            input_gradient_activations=input_gradient_activations,
            output_gradient_activations=output_gradient_activations,
        )

        try:
            yield activation_recording
        finally:
            activation_recording.remove_all()


@torch.no_grad()
@torch._dynamo.disable
def record_weight_gradients(
    model: nn.Module, histogram_recorder: HistogramRecorder
) -> dict[str, object]:
    args = {}
    for name, param in model.named_parameters():
        if param.dtype not in SUPPORTED_HISTOGRAM_TYPES:
            continue
        if param.grad is None:
            continue
        args |= histogram_recorder.calculate_histogram(name=name, x=param.grad).logs(
            "gradient"
        )
        abs_param = param.abs()
        args |= histogram_recorder.calculate_histogram(
            name=name,
            x=torch.log(
                param.grad.abs()
                / torch.where(
                    abs_param
                    == torch.tensor(0, dtype=abs_param.dtype, device=abs_param.device),
                    torch.ones_like(
                        abs_param, dtype=abs_param.dtype, device=abs_param.device
                    ),
                    abs_param,
                )
            ),
        ).logs("gradient_ratio")
    return args


@torch.no_grad()
@torch._dynamo.disable
def record_weights(
    model: nn.Module, weight_recorder: HistogramRecorder
) -> dict[str, object]:
    args = {}
    for name, param in model.named_parameters():
        if param.dtype not in SUPPORTED_HISTOGRAM_TYPES:
            continue
        args |= weight_recorder.calculate_histogram(name=name, x=param).logs("weight")
    return args
