from Net import Net

import time

import json

import numpy as np
import torch

device = "cpu"

model = Net()
model.to(device)
model.load_state_dict(torch.load("cifar_net_best.pt", map_location=device, weights_only=True))
model.eval()

with open("cifar10_stats.json") as f:
    stats = json.load(f)
mean_t = torch.tensor(stats["mean"]).view(1, 3, 1, 1).to(device)
std_t = torch.tensor(stats["std"]).view(1, 3, 1, 1).to(device)

# Load batch of images and outputs
test_x = torch.from_numpy(np.load("test_x.npy")).to(device)
y = torch.from_numpy(np.load("test_y.npy")).to(device)

# Preprocess images
start_time = time.perf_counter()

x = test_x / 255.0
x = x.permute(0, 3, 1, 2)
x = ((x - mean_t) / std_t).contiguous()

end_time = time.perf_counter()

elapsed_time = end_time - start_time

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
print("Preprocessing time: " + str(elapsed_time))