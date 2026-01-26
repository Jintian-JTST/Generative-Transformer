import os
import warnings

# 1. 屏蔽 Python层面的 UserWarning (比如那个 pkg_resources is deprecated)
warnings.filterwarnings("ignore")

# 2. 屏蔽 PyTorch/IPEX 底层 C++ 的 Info 和 Warning 信息
# 必须在 import torch 之前设置
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR" 
os.environ["IGC_EnableCheck"] = "0" # 针对 Intel 显卡驱动的一些检查日志

import torch
import intel_extension_for_pytorch as ipex

# --- 你的后续代码 ---
print(f"清爽模式启动！显卡: {torch.xpu.get_device_name(0)}")