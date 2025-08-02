from torch import nn
import torch
from jaxtyping import Float, Int
import einops


class Rope(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.types.Device = None,
    ):
        super().__init__()
        assert d_k % 2 == 0
        d_k2 = d_k // 2
        self.d_k = d_k
        self.theta = theta
        indices: Float[torch.Tensor, "max_seq_len"] = torch.arange(
            max_seq_len, device=device
        )
        ks: Float[torch.Tensor, "d_k2"] = torch.pow(
            theta, torch.arange(d_k2, device=device) / float(d_k2)
        )
        thetas: Float[torch.Tensor, "max_seq_len d_k2"] = torch.einsum(
            "m,k->mk", indices, 1 / ks
        )
        rs: Float[torch.Tensor, "max_seq_len d_k 2 2"] = torch.stack(
            [
                torch.stack([torch.cos(thetas), -torch.sin(thetas)], dim=-1),
                torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1),
            ],
            dim=-2,
        )
        self.register_buffer("rs", rs, persistent=False)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        x_pairs = einops.rearrange(x, "... seq_len (d_k g) -> ... seq_len d_k g", g=2)
        rs: Float[torch.Tensor, "... seq_len d_k2 2 2"] = self.rs[token_positions]
        stacked: Float[torch.Tensor, "... seq_len d_k2 2"] = torch.stack(
            [
                torch.einsum("...t,...t->...", rs[..., 0, :], x_pairs),
                torch.einsum("...t,...t->...", rs[..., 1, :], x_pairs),
            ],
            dim=-1,
        )
        return einops.rearrange(stacked, "... d_k2 g->... (d_k2 g)")
