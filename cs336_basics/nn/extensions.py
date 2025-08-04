from torch import nn
from collections import abc
from typing import TypeVar
import math

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
