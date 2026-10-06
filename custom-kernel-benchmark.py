from Net import Net

import time

import json

import numpy as np
import torch

from torch.utils.cpp_extension import load

import ctypes

lib = ctypes.CDLL("./libnormalize.so")
lib.launch_kernel.restype = None 
lib.launch_kernel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
]

device = "cuda"

model = Net()
model.to(device)
model.load_state_dict(torch.load("cifar_net_best.pt", map_location=device, weights_only=True))
model.eval()

with open("cifar10_stats.json") as f:
    stats = json.load(f)
mean = stats["mean"]
std = stats["std"]

# Load batch of images and outputs
x = torch.from_numpy(np.load("test_x.npy")).to(device)
y = torch.from_numpy(np.load("test_y.npy")).to(device)

# Preprocess using the custom normalization kernel
x = x.contiguous()
N, H, W, _ = x.shape
x_norm = torch.empty((N, 3, H, W), dtype=torch.float32, device=device)

lib.launch_kernel(
    x.data_ptr(), x_norm.data_ptr(), torch.cuda.current_stream().cuda_stream, N, H, W, mean[0], mean[1], mean[2], std[0], std[1], std[2]
)
x = x_norm
correct = 0

# Model forward pass
with torch.no_grad():
    for i in range(0, len(x), 128):
        batch = x[i:i+128]
        outs = model(batch)
        predictions = torch.argmax(outs, 1)
        correct_tensor = predictions == y[i:i+128]
        correct += correct_tensor.sum().item()

# Calculate accuracy 
accuracy = correct / len(x)
print("Accuracy: " + str(accuracy))
print("Correct: " + str(correct))