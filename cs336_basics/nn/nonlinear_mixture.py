from torch import nn
import torch
from jaxtyping import Float
from cs336_basics.nn import linear
import torch
from torch import nn
import torch.nn.functional as F


class FunctionalModule(nn.Module):
    """Wrapper for functional operations to make them proper nn.Modules"""

    def __init__(self, function, w):
        super().__init__()
        self._function = function
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._function(self.w(x))


class MixtureOfNonlinearFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        enabled: list[str],
        use_bias: bool,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        options = {
            k: v
            for k, v in {
                "identity": lambda x: x,
                "square": lambda x: x.square(),
                "cube": lambda x: x.pow(3),
                "sigmoid": torch.sigmoid,
                "gelu": F.gelu,
            }.items()
            if k in enabled
        }

        layers = []
        for i, (name, function) in enumerate(options.items()):
            w = linear.Linear(
                in_features=d_model,
                out_features=d_ff * (i + 1) // len(options) - d_ff * i // len(options),
                use_bias=use_bias,
                device=device,
                dtype=dtype,
            )
            layer = FunctionalModule(function, w)
            setattr(self, name, layer)
            layers.append(layer)
        self.layers = layers
        self.num_segments = len(self.layers)
        self.w2 = linear.Linear(
            in_features=d_ff,
            out_features=d_model,
            use_bias=use_bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ffn_output = torch.concat([layer(x) for layer in self.layers], dim=-1)
        return self.w2(ffn_output)