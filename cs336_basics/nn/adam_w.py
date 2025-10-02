import torch
from typing import Callable
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        assert lr >= 0
        assert weight_decay >= 0
        assert eps > 0
        assert betas[0] > 0
        assert betas[1] > 0
        super().__init__(params, defaults)
        for group in self.param_groups:
            group_lr = group["lr"]
            for p in group["params"]:
                p_state = self.state[p]
                p_state["m"] = torch.zeros_like(p)
                p_state["v"] = torch.zeros_like(p)
                p_state["alpha_t"] = group_lr

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p_state = self.state[p]
                t = p_state.get("t", 0) + 1
                p_state["t"] = t
                p_state["m"].lerp_(p.grad.data, 1 - betas[0])
                p_state["v"].mul_(betas[1]).addcmul_(
                    p.grad.data, p.grad.data, value=1 - betas[1]
                )
                lr = group["lr"] * getattr(p, "lr_mul", 1)
                alpha_t = (
                    lr
                    * math.sqrt((1 - math.pow(betas[1], t)))
                    / (1 - math.pow(betas[0], t))
                )
                denominator = torch.sqrt(p_state["v"]).add_(eps)
                p.data.addcdiv_(p_state["m"], denominator, value=-alpha_t)
                p.data.mul_(1 - lr * weight_decay)

        return loss
