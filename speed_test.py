import torch, time
import torch.nn as nn
import intel_extension_for_pytorch as ipex

class HeavyModel(nn.Module):
    def __init__(self, depth=12, dim=4096):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim), nn.GELU())
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def benchmark(device, batch=256, repeat=50, use_ipex=False):
    model = HeavyModel().to(device).eval()

    if device.type == "xpu" and use_ipex:
        model = ipex.optimize(model)

    x = torch.randn(batch, 4096, device=device)

    # warmup
    for _ in range(10):
        _ = model(x)

    if device.type == "xpu":
        torch.xpu.synchronize()

    start = time.time()
    for _ in range(repeat):
        _ = model(x)
    if device.type == "xpu":
        torch.xpu.synchronize()

    elapsed = time.time() - start
    throughput = batch * repeat / elapsed

    return elapsed, throughput


cpu_t, cpu_tp = benchmark(torch.device("cpu"))
xpu_t, xpu_tp = benchmark(torch.device("xpu"), use_ipex=True)

print(f"CPU  time: {cpu_t:.3f}s, throughput: {cpu_tp:.1f} samples/s")
print(f"XPU  time: {xpu_t:.3f}s, throughput: {xpu_tp:.1f} samples/s")
print(f"Speedup: {xpu_tp / cpu_tp:.2f}x")
