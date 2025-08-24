from torch import nn
import torch
from jaxtyping import Float
from cs336_basics.nn import linear
import torch
from torch import nn
from torch.autograd import Function


class SwiGlu(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = linear.Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.w2 = linear.Linear(
            in_features=d_ff, out_features=d_model, device=device, dtype=dtype
        )
        self.w3 = linear.Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.w1(x)
        return self.w2(
            w1x * torch.sigmoid(w1x) * self.w3(x),
        )

class SiLU(nn.Module):
    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        return x * torch.sigmoid(x)

class ReLUWithELUSurrogateGradFn(Function):
    @staticmethod
    def forward(ctx, x, beta):
        # Save input and beta for backward
        ctx.save_for_backward(x, torch.tensor(beta, dtype=x.dtype, device=x.device))
        # Standard ReLU forward
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, beta_t = ctx.saved_tensors

        # ELU-style derivative with lower bound beta on the negative side
        # For x < 0: max(exp(x), beta); for x >= 0: 1
        neg_grad = torch.maximum(torch.exp(x), beta_t)
        grad = torch.where(x < 0, neg_grad, torch.ones_like(x))

        # Chain rule
        grad_input = grad_output * grad

        # No gradient w.r.t. beta (treat beta as hyperparameter)
        return grad_input, None


class ReLUWithELUSurrogateGrad(nn.Module):
    """
    ReLU forward, ELU-like surrogate gradient in backward, with min slope beta on the negative side.
    """
    def __init__(self, beta: float = 0.01):
        super().__init__()
        if beta <= 0 or beta > 1:
            raise ValueError("beta should be in (0, 1].")
        self.beta = float(beta)

    def forward(self, x):
        # Pass beta as a second argument to the custom Function
        return ReLUWithELUSurrogateGradFn.apply(x, self.beta)
    

class ReluSoft(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = linear.Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.w2 = linear.Linear(
            in_features=d_ff, out_features=d_model, device=device, dtype=dtype
        )
        self.w3 = linear.Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.ff = ReLUWithELUSurrogateGrad(beta=0.05)

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        w1x = self.w1(x)
        return self.w2(
            self.ff(w1x) * self.w3(x),
        )