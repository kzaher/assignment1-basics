from torch import nn
from collections import abc
from typing import TypeVar
from collections import abc
import math
import torch

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
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate

    return min_learning_rate + 0.5 * (
        1
        + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
    ) * (max_learning_rate - min_learning_rate)


def gradient_clipping(
    parameters: abc.Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
):
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
        return

    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter.grad.data = max_l2_norm / (total_norm + eps) * parameter.grad
