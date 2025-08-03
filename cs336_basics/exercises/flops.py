# %%
from cs336_basics.nn import extensions
from cs336_basics.nn import linear
from cs336_basics.nn import transformer_lm
from torch import nn
from __future__ import annotations
import dataclasses
from collections import abc
import pandas as pd
import torch

import importlib

importlib.reload(extensions)

gpt2_small_params = dict(
    vocab_size=50257,
    context_length=1024,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=6400,
    rope_theta=10000,
    device=None,
    dtype=None,
)

gpt2_medium_params = dict(
    vocab_size=50257,
    context_length=1024,
    d_model=1024,
    num_layers=24,
    num_heads=16,
    d_ff=6400,
    rope_theta=10000,
    device=None,
    dtype=None,
)

gpt2_large_params = dict(
    vocab_size=50257,
    context_length=1024,
    d_model=1280,
    num_layers=36,
    num_heads=20,
    d_ff=6400,
    rope_theta=10000,
    device=None,
    dtype=None,
)

gpt2_xl_params = dict(
    vocab_size=50257,
    context_length=1024,
    d_model=1600,
    num_layers=48,
    num_heads=25,
    d_ff=6400,
    rope_theta=10000,
    device=None,
    dtype=None,
)

gpt2s = {
    "gpt2-small": gpt2_small_params,
    "gpt2-medium": gpt2_medium_params,
    "gpt2-large": gpt2_large_params,
    "gpt2-xl": gpt2_xl_params,
}

test_params = dict(
    vocab_size=50257,
    context_length=1024,
    d_model=100,
    num_layers=2,
    num_heads=2,
    d_ff=1000,
    rope_theta=10000,
    device=None,
    dtype=None,
)


@dataclasses.dataclass
class Node:
    parent: nn.Module | None
    attribute: str | None
    module: nn.Module
    children: list[nn.Module]
    child_nodes: list[Node]

    @classmethod
    def build(cls, root: nn.Module) -> Node:
        node_mapping: dict[nn.Module, Node] = {}

        def visitor(key: str | None, parent: nn.Module | None, current: nn.Module):
            if parent is not None and key is not None:
                node_mapping[parent].children.append(current)
            node_mapping[current] = Node(
                parent=parent,
                attribute=key,
                module=current,
                children=[],
                child_nodes=[],
            )

        list(extensions.visit(root, visitor))
        for value in node_mapping.values():
            value.child_nodes = [
                node_mapping[child_module] for child_module in value.children
            ]
        return node_mapping[root]

    def print(self, get_node_attributes: abc.Callable[[Node], dict]) -> pd.DataFrame:
        width = 2

        def print_internal(node: Node, indent: int) -> abc.Iterator[dict]:
            attributes = get_node_attributes(node)
            if indent:
                indent_part = (width * (indent - 1)) * " " + "└" + ("─" * (width - 1))
                print(
                    f"{indent_part} {node.attribute}:{type(node.module)} - {attributes}"
                )
            yield attributes
            for child_node in node.child_nodes:
                yield from print_internal(child_node, indent=indent + 1)

        return pd.DataFrame(list(print_internal(self, indent=0)))

    def all(self):
        yield self
        for child_node in self.child_nodes:
            yield from child_node.all()


for k, params in gpt2s.items():
    print(f"Model: {k}")
    instance = transformer_lm.TransformerLm(**params)
    node = Node.build(instance)

    def find_flops_inject(node: Node):
        if not isinstance(node.module, linear.Linear):
            return {"injected": False}
        node._distinct_sizes = set[int]()
        node.module._original_forward = node.module.forward

        def save_operand(x):
            node._distinct_sizes.add(x.size(-2))
            return node.module._original_forward(x)

        node.module.forward = save_operand
        return {"injected": True}

    for inject_node in node.all():
        find_flops_inject(inject_node)
    x = torch.zeros((1, params["context_length"])).to(torch.int32)
    instance.forward(x)

    def get_flops(node: Node):
        if not isinstance(node.module, linear.Linear):
            return {}
        (multiplication_factor,) = node._distinct_sizes
        total = 2 * multiplication_factor
        for size in node.module.weight.size():
            total *= size
        return {"linear_flops": total}

    print(len(list(node.all())))
    gflop = node.print(get_flops)["linear_flops"].sum() / 1e9
    print(f"Gflops {gflop}")
