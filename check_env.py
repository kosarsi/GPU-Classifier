import torch, time

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))  # want (5, 3)

# GPU matmul, checked against CPU
a = torch.randn(512, 512)
b = torch.randn(512, 512)
cpu_out = a @ b
gpu_out = (a.cuda() @ b.cuda()).cpu()
print("matmul matches CPU:", torch.allclose(cpu_out, gpu_out, atol=1e-3))

# Conv layer through cuDNN
conv = torch.nn.Conv2d(3, 16, 3).cuda()
x = torch.randn(8, 3, 32, 32).cuda()
y = conv(x)
torch.cuda.synchronize()
print("conv output shape:", tuple(y.shape))  # want (8, 16, 30, 30)
print("cudnn enabled:", torch.backends.cudnn.enabled)

# Memory after the above
print("allocated MB:", torch.cuda.memory_allocated() / 1e6)