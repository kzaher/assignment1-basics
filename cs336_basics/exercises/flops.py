# %%
from __future__ import annotations
from cs336_basics.nn import linear
from cs336_basics.nn import transformer_lm
from cs336_basics.pretraining import configuration
from cs336_basics import extensions
from torch import nn
import pandas as pd
import torch
import numpy as np
import json

import importlib

gpt2_small_params = dict(
    vocab_size=50257,
    max_sequence_length=1024,
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
    max_sequence_length=1024,
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
    max_sequence_length=1024,
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
    max_sequence_length=1024,
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
    max_sequence_length=1024,
    d_model=100,
    num_layers=2,
    num_heads=2,
    d_ff=1000,
    rope_theta=10000,
    device=None,
    dtype=None,
)


class FlopCounter:
    def __init__(self):
        self.recorded_input_data = {}
        self.hooks = []

    def register_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, linear.Linear):
                self.recorded_input_data[name] = {
                    "module": module,
                    "input_sizes": set(),
                }
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)

    def _create_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(module, linear.Linear):
                x = input[0]
                self.recorded_input_data[name]["input_sizes"].add(x.size(-2))

        return hook

    def calculate_flop(self) -> pd.DataFrame:
        def flops(module_name, recorded_input_data):
            module = recorded_input_data["module"]
            input_sizes = recorded_input_data["input_sizes"]

            if not input_sizes:
                return
            (input_size,) = input_sizes
            multiplication_factor = input_size
            total = 2 * multiplication_factor
            for size in module.weight.size():
                total *= size
            return {"module_name": module_name, "linear_flop": total}

        return pd.DataFrame(
            [
                flops(module_name, recoded_input_data)
                for module_name, recoded_input_data in self.recorded_input_data.items()
            ]
        )

    def print_structure(self, model: nn.Module, get_attributes_fn=None):
        def print_module(name: str, module: nn.Module, depth: int = 0):
            attributes = get_attributes_fn(name, module) if get_attributes_fn else None

            if depth > 0:
                width = 2
                indent_part = (width * (depth - 1)) * " " + "└" + ("─" * (width - 1))
                display_name = f"{indent_part} {name.split('.')[-1]}:{type(module).__name__} {attributes}"
                print(display_name)
            else:
                print(f"{type(module).__name__}")

            for child_name, child_module in module.named_children():
                child_full_name = f"{name}.{child_name}" if name else child_name
                print_module(child_full_name, child_module, depth + 1)

        print_module("", model, 0)

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.recorded_input_data.clear()


exp_path = "/workspace/cs336_basics/pretraining/configurations/owt_gemma_270M.exp.json"
with open(exp_path, "rt") as f:
    exp_configuration_instance = configuration.PretrainingConfiguration(
        output_path="",
        checkpoint=0,
        training_loop=configuration.LlmPretrainingTrainingLoopConfiguration.from_dict(
            json.load(f)
        ),
    )

# for k, params in gpt2s.items():
if True:
    exp_configuration_instance = extensions.replace_recursively(
        exp_configuration_instance,
        lambda x: x.training_loop.transformer_llm.dtype,
        torch.float32,
    )
    exp_configuration_instance = extensions.replace_recursively(
        exp_configuration_instance,
        lambda x: x.training_loop.transformer_llm.device,
        "cpu",
    )
    instance = transformer_lm.TransformerLm(
        exp_configuration_instance.training_loop.transformer_llm
    )
    count_parameters = pd.DataFrame(
        [
            {"name": name, "params": np.prod(param.size())}
            for name, param in instance.named_parameters()
        ]
    )
    print(f"# Params\n{count_parameters}")

    flop_counter = FlopCounter()
    flop_counter.register_hooks(instance)

    def get_flops_attributes(name: str, module: nn.Module):
        if not isinstance(module, linear.Linear):
            return None
        if (
            name in flop_counter.recorded_input_data
            and flop_counter.recorded_input_data[name]["input_sizes"]
        ):
            (multiplication_factor,) = flop_counter.recorded_input_data[name][
                "input_sizes"
            ]
            total = 2 * multiplication_factor
            for size in module.weight.size():
                total *= size
            return {f"Gflop": total / 1e9}

    x = torch.zeros(
        (
            1,
            exp_configuration_instance.training_loop.transformer_llm.max_sequence_length,
        )
    ).to(torch.int32)
    instance.forward(x)

    print("\nModel Structure:")
    flop_counter.print_structure(instance, get_flops_attributes)

    # Calculate and print FLOPS summary
    flop_df = flop_counter.calculate_flop()
    print(flop_df)
    total_linear_modules = len(flop_df)
    print(f"\nNumber of Linear modules: {total_linear_modules}")
    gflop = flop_df["linear_flop"].sum() / 1e9
    print(f"Gflop {gflop}")

    # Clean up hooks
    flop_counter.cleanup()
    print("-" * 50)

# %%
