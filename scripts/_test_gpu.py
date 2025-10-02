import torch

M = N = K = 8192
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

# Warmup
for _ in range(10):
    torch.matmul(a, b)

# Events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()         # make sure warmup is done
start.record()                   # record start *after* sync

for _ in range(100000):
    torch.matmul(a, b)

end.record()                     # record stop right after last op
torch.cuda.synchronize()         # wait for kernels to finish

# Elapsed time in ms
elapsed_ms = start.elapsed_time(end)
elapsed_s = elapsed_ms / 1000.0

flops = 2 * M * N * K * 100000
tflops = flops / elapsed_s / 1e12

print(f"Elapsed: {elapsed_ms:.2f} ms")
print(f"Throughput: {tflops:.1f} TFLOP/s")
