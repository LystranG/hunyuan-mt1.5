import torch

if torch.cuda.is_available():
    print("✅ CUDA 可用")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
else:
    print("❌ CUDA 不可用，请检查环境")

# 创建张量并移动至GPU
x = torch.rand(3, 3).to('cuda')
y = torch.rand(3, 3).to('cuda')
z = x @ y  # 执行矩阵乘法
print(f"运算设备: {z.device}")  # 应输出 cuda:0
