import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex
import time

# =================配置区 (在这里调节压力)=================
# 1. 增加宽度：矩阵运算量是宽度的平方级。4096 -> 8192 计算量翻4倍
HIDDEN_DIM = 8192   
# 2. 增加深度：层数越多，串行计算越久
NUM_LAYERS = 20     
# 3. 增加 Batch Size：这是填满计算单元(EU/Core)的关键。
#    如果显存溢出(OOM)，请减小这个值；如果显存没满，往死里加。
BATCH_SIZE = 2048   
# 4. 持续循环次数：单次运行不够热，必须持续轰炸
LOOP_COUNT = 100    
# ========================================================

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"🔥 Current Device: {device}")

# 动态构建超重模型
class SuperHeavyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        model_list = []
        # 输入层
        model_list.append(nn.Linear(input_dim, hidden_dim))
        model_list.append(nn.GELU())
        
        # 中间层 (大量堆叠)
        for _ in range(layers - 2):
            model_list.append(nn.Linear(hidden_dim, hidden_dim))
            model_list.append(nn.GELU()) # GELU 包含 exp/tanh 运算，比 ReLU 累
            
        # 输出层
        model_list.append(nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*model_list)

    def forward(self, x):
        return self.net(x)

print(f"🏗️ Building Model: {NUM_LAYERS} Layers, {HIDDEN_DIM} Width...")
model = SuperHeavyModel(4096, HIDDEN_DIM, NUM_LAYERS).to(device)
model.eval()

# 数据生成 (消耗大量带宽)
print(f"📦 Generating Data (Batch: {BATCH_SIZE})...")
try:
    input_data = torch.randn(BATCH_SIZE, 4096, device=device)
except RuntimeError as e:
    print("❌ 显存不足 (OOM)，请减小 BATCH_SIZE 或 HIDDEN_DIM")
    raise e

# IPEX 优化
print("🛠️ Optimizing with IPEX (BF16)...")
try:
    model = ipex.optimize(model, dtype=torch.bfloat16)
    use_bf16 = True
except Exception:
    model = ipex.optimize(model)
    use_bf16 = False
    print("⚠️ Fallback to FP32")

# 压力测试主循环
print(f"🚀 Starting Stress Loop ({LOOP_COUNT} iterations)...")
amp_device_type = "xpu" if device.type == "xpu" else "cpu"

# 预热
for _ in range(5):
    with torch.autocast(device_type=amp_device_type, enabled=use_bf16, dtype=torch.bfloat16):
        _ = model(input_data)

torch.xpu.synchronize() if device.type == "xpu" else None
start_time = time.time()

# 持续轰炸
for i in range(LOOP_COUNT):
    with torch.autocast(device_type=amp_device_type, enabled=use_bf16, dtype=torch.bfloat16):
        output = model(input_data)
    
    # 每 50 次同步一次，防止 CPU 跑太快 GPU 队列堆积导致测量不准，
    # 但为了最大化压力，通常不需要频繁同步，只需要让 GPU 队列塞满。
    if i % 1 == 0:
            # 强制 CPU 等 GPU 算完这一步再打印，这样进度条就是实时的了
            torch.xpu.synchronize() 
            print(f"   Step {i}/{LOOP_COUNT} completed...")

# 确保所有计算完成
torch.xpu.synchronize() if device.type == "xpu" else None
end_time = time.time()
total_time = end_time - start_time

print(f"✅ Stress Test Finished.")
print(f"⏱️ Total Time: {total_time:.2f}s")
print(f"⚡ Throughput: {LOOP_COUNT / total_time:.2f} iter/s")