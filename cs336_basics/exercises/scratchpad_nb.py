# %%
import torch
from torch.autograd import profiler

x = torch.randn((1, 1), requires_grad=True)

with profiler.emit_nvtx():
    y = x ** 2
    with torch.autograd.profiler.record_function("forward_z"):
        z = y ** 3

    # The backward graph will be connected to these forward ranges
    z.backward()
# %%
