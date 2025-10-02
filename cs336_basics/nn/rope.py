from torch import nn
import torch
from jaxtyping import Float, Int


class Rope(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
        packed_rope: bool = False,
    ):
        super().__init__()
        assert d_k % 2 == 0
        d_k2 = d_k // 2
        self.d_k = d_k
        self.theta = theta
        self.packed_rope = packed_rope
        indices: Float[torch.Tensor, "max_seq_len"] = torch.arange(
            max_seq_len, device=device
        )
        ks: Float[torch.Tensor, "d_k2"] = torch.pow(
            theta, torch.arange(d_k2, device=device) / float(d_k2)
        )
        thetas: Float[torch.Tensor, "max_seq_len d_k2"] = torch.einsum(
            "m,k->mk", indices, 1 / ks
        )
        cos: Float[torch.Tensor, "max_seq_len d_k2"] = (
            torch.cos(thetas).to(dtype).to(device=device)
        )
        sin: Float[torch.Tensor, "max_seq_len d_k2"] = (
            torch.sin(thetas).to(dtype).to(device=device)
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "seq_len"] | None,
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        def separate(tensor: torch.Tensor):
            first = tensor[..., 0::2]
            second = tensor[..., 1::2]
            return torch.concat((first, second), dim=-1)

        def interleave(tensor: torch.Tensor):
            first, second = torch.chunk(tensor, chunks=2, dim=-1)
            return torch.flatten(torch.stack((first, second), dim=-1), start_dim=-2)

        if self.packed_rope:
            x = separate(x)

        dim0, dim1 = torch.chunk(x, chunks=2, dim=-1)
        cos_collected: Float[torch.Tensor, "... seq_len d_k2"] = (
            self.cos[token_positions, :]
            if token_positions is not None
            else self.cos[: x.size(-2), ...]
        )
        sin_collected: Float[torch.Tensor, "... seq_len d_k2"] = (
            self.sin[token_positions, :]
            if token_positions is not None
            else self.sin[: x.size(-2), ...]
        )
        result = torch.cat(
            [
                dim0 * cos_collected - dim1 * sin_collected,
                dim0 * sin_collected + dim1 * cos_collected,
            ],
            dim=-1,
        )

        if self.packed_rope:
            result = interleave(result)

        return result
