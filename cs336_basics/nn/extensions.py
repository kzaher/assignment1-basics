# %%
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

T = TypeVar("T")


def visit(
    root: nn.Module,
    visitor: abc.Callable[[str | None, nn.Module | None, nn.Module], T],
    key=None,
    parent=None,
) -> abc.Iterator[T]:
    yield visitor(key, parent, root)
    for key, value in root._modules.items():
        if not value:
            continue
        yield from visit(value, visitor, key=key, parent=root)


def cosine_learning_rate(
    it: int,
    zero_iters: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < zero_iters:
        return 0
    if it < warmup_iters:
        return (it - zero_iters) / (warmup_iters - zero_iters) * max_learning_rate
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
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str, model: nn.Module, optimizer: torch.optim.Optimizer
) -> int:
    state = torch.load(src)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["iteration"]


# %%
