import torch
from torch import nn
from torch.autograd import Function


class _AtanGradBoost(Function):
    @staticmethod
    def forward(ctx, x, grad_boost, max_grad, horizontal_scale):
        ctx.save_for_backward(x)
        ctx.grad_boost = grad_boost
        ctx.max_grad = max_grad
        ctx.horizontal_scale = horizontal_scale
        return torch.atan(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_boost = ctx.grad_boost
        max_grad = ctx.max_grad
        horizontal_scale = ctx.horizontal_scale

        # Standard atan gradient is 1/(1 + x**2)
        grad_input = grad_output / (1 + x**2)

        # Apply gradient boost
        if grad_boost != 1.0:
            grad_input = grad_input * grad_boost

        # Apply max grad clipping
        if max_grad is not None and max_grad > 0:
            actual_max = grad_input.abs().max()
            max_grad_tensor = torch.tensor(
                max_grad, device=grad_input.device, dtype=grad_input.dtype
            )
            if actual_max > max_grad_tensor:
                grad_input = grad_input / actual_max * max_grad_tensor

        # Always apply chain rule with horizontal_scale for the input gradient
        # This is required because forward does: scaled_x = x * horizontal_scale
        grad_x = grad_input * horizontal_scale

        # Only compute gradient w.r.t. horizontal_scale if it's learnable (requires_grad=True)
        grad_horizontal_scale = None
        if horizontal_scale is not None and horizontal_scale.requires_grad:
            grad_horizontal_scale = (grad_input * x).sum()
            max_horizontal_scale_value = torch.tensor(
                10.0,
                dtype=grad_horizontal_scale.dtype,
                device=grad_horizontal_scale.device,
            )
            grad_horizontal_scale = (
                grad_horizontal_scale
                / torch.max(grad_horizontal_scale.abs(), max_horizontal_scale_value)
                * max_horizontal_scale_value
            )

        return grad_x, None, None, grad_horizontal_scale


class Atan(nn.Module):
    def __init__(
        self,
        d_model: int,
        grad_boost=1.0,
        max_grad: float | None = None,
        horizontal_scale=1.0,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
        learnable_horizontal_scale=False,
        learnable_weight=True,
    ):
        super().__init__()
        self.grad_boost = grad_boost
        self.max_grad = max_grad

        # Use provided dtype or default to torch.float32
        tensor_dtype = dtype if dtype is not None else torch.float32
        self.d_model = d_model

        self.weight = (
            torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
            if learnable_weight
            else None
        )

        if learnable_horizontal_scale:
            self.horizontal_scale = nn.Parameter(
                torch.tensor(horizontal_scale, device=device, dtype=tensor_dtype)
            )
        else:
            self.register_buffer(
                "horizontal_scale",
                torch.tensor(horizontal_scale, device=device, dtype=tensor_dtype),
            )

    def forward(self, x):
        # Apply horizontal scaling
        if self.weight is not None:
            x = x * self.weight
        scaled_x = x * self.horizontal_scale
        # Always pass horizontal_scale for proper chain rule, but only compute parameter gradients if it's a Parameter
        return _AtanGradBoost.apply(
            scaled_x, self.grad_boost, self.max_grad, self.horizontal_scale
        )


class AtanTernary(nn.Module):
    def __init__(
        self,
        grad_boost=1.0,
        max_grad: float | None = None,
        offset=10.0,
        horizontal_scale=1.0,
        device: torch.types.Device = None,
        dtype: torch.dtype | None = None,
        learnable_horizontal_scale=False,
        learnable_offset=True
    ):
        super().__init__()
        self.grad_boost = grad_boost
        self.max_grad = max_grad
        self.offset = offset

        if learnable_horizontal_scale:
            self.horizontal_scale = nn.Parameter(
                torch.tensor(horizontal_scale, device=device, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "horizontal_scale",
                torch.tensor(horizontal_scale, device=device, dtype=dtype),
            )

        if learnable_offset:
            self.offset = nn.Parameter(
                torch.tensor(offset, device=device, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "offset",
                torch.tensor(offset, device=device, dtype=dtype)
            )

    def forward(self, x):
        # Apply horizontal scaling first, then offset
        scaled_x = (x * self.horizontal_scale).to(x.dtype)
        # print('horizontal scale ', self.horizontal_scale.detach().item())
        # Always pass horizontal_scale for proper chain rule, but only compute parameter gradients if it's a Parameter
        return _AtanGradBoost.apply(
            scaled_x - self.offset,
            self.grad_boost,
            self.max_grad,
            self.horizontal_scale,
        ) + _AtanGradBoost.apply(
            scaled_x + self.offset,
            self.grad_boost,
            self.max_grad,
            self.horizontal_scale,
        )
