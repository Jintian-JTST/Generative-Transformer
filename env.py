import torch
import intel_extension_for_pytorch as ipex

print(f"Torch 版本: {torch.__version__}")  # 必须包含 +cxx11.abi 或 similar，绝对不能是 +cpu
print(f"IPEX 版本: {ipex.__version__}")

if torch.xpu.is_available():
    print(f"✅ 成功！检测到显卡: {torch.xpu.get_device_name(0)}")
    x = torch.randn(2, 2).to("xpu")
    print("   计算测试: 通过")
else:
    print("⚠️ 安装成功，但未检测到设备 (请确保显卡驱动已更新)")