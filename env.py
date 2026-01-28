import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex

# 1️⃣ 设备选择
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"🔥 Current Device: {device}")

# 2️⃣ 模型：换一个“大压力”模型 (3层 4096 宽度的 MLP，模拟高负载矩阵乘法)
class HeavyPressureModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 增大维度：从 1024 -> 4096，计算量翻 16 倍
        self.fc1 = nn.Linear(4096, 4096)
        self.act1 = nn.GELU()  # GELU 比 ReLU 计算稍微重一点
        
        self.fc2 = nn.Linear(4096, 4096)
        self.act2 = nn.GELU()
        
        self.fc3 = nn.Linear(4096, 4096) # "三" 压力 -> 第三层

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x

model = HeavyPressureModel().to(device)
model.eval()

# 3️⃣ 数据：增大 Batch Size (增加吞吐压力)
# 4096维 * 64 batch size = 很大的矩阵
BATCH_SIZE = 64
input_data = torch.randn(BATCH_SIZE, 4096, device=device)

# 4️⃣ IPEX 优化：开启 BF16 (BFloat16)
# Intel 硬件(CPU/Arc/Data Center GPU) 跑 BF16 效率最高，压力测试必开
print("🛠️  Optimizing with IPEX (BF16)...")
try:
    # 尝试开启 BF16 优化
    model = ipex.optimize(model, dtype=torch.bfloat16)
    use_bf16 = True
    print("✅ BF16 Optimization Enabled.")
except Exception as e:
    # 如果硬件不支持 BF16，回退到 FP32
    print(f"⚠️  BF16 not supported ({e}), fallback to FP32.")
    model = ipex.optimize(model)
    use_bf16 = False

# 5️⃣ 推理 (Forward)
print("🚀 Running Forward Pass (Stress Test)...")

# 根据设备类型选择 AMP 上下文
amp_device_type = "xpu" if device.type == "xpu" else "cpu"

with torch.no_grad():
    # 预热 (Warmup) - 让硬件进入高性能状态
    for _ in range(5):
        if use_bf16:
            with torch.autocast(device_type=amp_device_type, enabled=True, dtype=torch.bfloat16):
                _ = model(input_data)
        else:
            _ = model(input_data)
            
    # 正式运行
    import time
    start = time.time()
    
    if use_bf16:
        with torch.autocast(device_type=amp_device_type, enabled=True, dtype=torch.bfloat16):
            output = model(input_data)
    else:
        output = model(input_data)
        
    cost = time.time() - start

print(f"✅ Forward OK. Output shape: {output.shape}")
print(f"⏱️  Time cost: {cost * 1000:.2f} ms")