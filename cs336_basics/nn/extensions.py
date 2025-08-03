from torch import nn
from collections import abc
from typing import TypeVar

T = TypeVar("T")


def visit(
    root: nn.Module,
    visitor: abc.Callable[[str | None, nn.Module | None, nn.Module], T],
    key=None,
    parent=None,
) -> abc.Iterator[T]:
    yield visitor(key, parent, root)
    for key, value in root._modules.items():
        if not value: continue
        yield from visit(value, visitor, key=key, parent=root)
