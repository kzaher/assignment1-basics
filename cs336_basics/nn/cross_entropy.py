from torch import nn
import torch
from jaxtyping import Float, Int
from torch import Tensor


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        o: Float[Tensor, "... batch seq_length vocab_size"],
        target: Int[Tensor, "... batch seq_length"],
    ) -> Float[Tensor, "..."]:
        normalized_o: Float[Tensor, "... batch seq_length vocab_size"] = (
            o - torch.max(o, dim=-1, keepdim=True).values
        )
        loss = -torch.squeeze(
          torch.gather(
              input=normalized_o, index=target[..., torch.newaxis], dim=-1
          ),
          dim=-1,
        ) + torch.log(torch.sum(torch.exp(normalized_o), dim=-1))
        return loss.sum(-1).sum(-1) / (loss.size(-1) * loss.size(-2))
