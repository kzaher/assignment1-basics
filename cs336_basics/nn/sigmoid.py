# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class _SigmoidClampGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, mu, s, min_grad, max_grad, beta):
#         """
#         mode_flag: 0 -> hard floor  |  1 -> smooth floor (softplus)
#         Only x gets custom gradient in this implementation (mu, W, s treated as constants).
#         """
#         mu = torch.as_tensor(mu, dtype=x.dtype, device=x.device)
#         s = torch.as_tensor(s, dtype=x.dtype, device=x.device)

#         a = (x - mu) / s
#         y = torch.sigmoid(a)

#         # Save tensors for backward
#         ctx.save_for_backward(x, y)
#         ctx.beta = float(beta)
#         ctx.mu = mu
#         ctx.s = s
#         ctx.min_grad = float(min_grad)
#         ctx.max_grad = float(max_grad)
#         return y

#     @staticmethod
#     def backward(ctx, grad_out):
#         x, y = ctx.saved_tensors
#         min_grad = ctx.min_grad
#         max_grad = ctx.max_grad
#         s = ctx.s
#         beta = ctx.beta

#         grad = (y * (1 - y)) / s
        
#         if min_grad > 0:
#             grad = torch.maximum(grad, min_grad)

#         if beta > 0:
#             # abs_mod ≈ max(abs_g, gmin) but smooth around gmin
#             grad = min_grad + F.softplus(grad - min_grad, beta=beta)

#         grad = grad * grad_out

#         if max_grad > 0:
#             actual_max = grad.abs().max()
#             if actual_max > max_grad:
#                 grad = grad / actual_max * max_grad

#         return grad, None, None, None, None, None



# class SigmoidCustomGrad(nn.Module):
#     """
#     Forward: σ((x-μ)/s)
#     Backward: gradient wrt x is floored (hard or smooth) to prevent vanishing tails.

#     Args:
#         mu (float): center of the window
#         W  (float): half-width of the window (controls plateau width)
#         s  (float): edge softness (smaller s -> sharper edges)
#         grad_floor (float): minimum gradient magnitude enforced in backward
#         smooth (bool): use smooth floor (softplus) if True, else hard clamp
#         softplus_beta (float): steepness for the smooth floor knee
#     """

#     def __init__(
#         self,
#         mu=0.0,
#         s=0.05,
#         max_grad: float = 0.0,
#         min_grad: float = 0.0,
#         gradient_mode: str = "default",
#         softplus_beta=0.0,
#     ):
#         super().__init__()
#         self.mu = mu
#         self.s = float(s)
#         self.max_grad = float(max_grad)
#         self.min_grad = float(min_grad)
#         self.gradient_mode = gradient_mode
#         self.softplus_beta = float(softplus_beta)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return _SigmoidClampGrad.apply(
#             x,
#             self.mu,
#             self.s,
#             self.min_grad,
#             self.max_grad,
#             self.softplus_beta,
#         )


# class SigmoidTernary(nn.Module):
#     def __init__(
#         self,
#         mu=0.0,
#         s=0.05,
#         max_grad: float = -1,
#         min_grad: float = -1,
#         gradient_mode: str = "default",
#         softplus_beta=10.0,
#     ):
#         super().__init__()
#         self.positive = SigmoidCustomGrad(
#             mu=mu,
#             s=s,
#             max_grad=max_grad,
#             min_grad=min_grad,
#             gradient_mode=gradient_mode,
#             softplus_beta=softplus_beta,
#         )
#         self.negative = SigmoidCustomGrad(
#             mu=-mu,
#             s=-s,
#             max_grad=max_grad,
#             min_grad=min_grad,
#             gradient_mode=gradient_mode,
#             softplus_beta=softplus_beta,
#         )

#     def forward(self, x):
#         return self.positive(x) - self.negative(x)
