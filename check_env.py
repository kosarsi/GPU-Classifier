#!/usr/bin/env python3
"""Environment check for the Jetson Orin Nano GPU classifier project.

Usage:
    python3 check_env.py                  # full check, including a CUDA kernel compile
    python3 check_env.py --skip-compile   # skip the (slow) nvcc compile test
"""
import glob
import os
import re
import shutil
import subprocess
import sys

SKIP_COMPILE = "--skip-compile" in sys.argv
results = []  # (name, status, detail) where status is OK / WARN / FAIL


def check(name, fn, required=True):
    try:
        detail = fn() or ""
        results.append((name, "OK", detail))
        print(f"[ OK ] {name}: {detail}")
    except Exception as e:
        status = "FAIL" if required else "WARN"
        results.append((name, status, str(e)))
        print(f"[{status}] {name}: {e}")


def python_version():
    v = sys.version_info
    assert v[:2] == (3, 10), f"expected Python 3.10 (matches Jetson wheels), got {sys.version.split()[0]}"
    return sys.version.split()[0]


def numpy_version():
    import numpy
    major = int(numpy.__version__.split(".")[0])
    assert major < 2, f"numpy {numpy.__version__} is 2.x; run: pip install 'numpy==1.26.4'"
    return numpy.__version__


def torch_build():
    import torch
    assert "+cpu" not in torch.__version__, f"{torch.__version__} is a CPU-only build"
    assert "cu13" not in torch.__version__, f"{torch.__version__} targets CUDA 13; Orin needs a CUDA 12.6 build"
    assert torch.version.cuda, "torch was built without CUDA"
    return f"torch {torch.__version__}, built for CUDA {torch.version.cuda}"


def cuda_available():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    return f"{name}, compute capability {major}.{minor} (Orin should be 8.7)"


def gpu_matmul():
    import torch
    a = torch.randn(512, 512, device="cuda")
    gpu = (a @ a).cpu()
    torch.cuda.synchronize()
    cpu = a.cpu() @ a.cpu()
    diff = (gpu - cpu).abs().max().item()
    assert diff < 1e-2, f"GPU and CPU matmul disagree (max diff {diff})"
    return f"matmul matches CPU (max diff {diff:.2e})"


def cudnn_conv():
    import torch
    import torch.nn as nn
    assert torch.backends.cudnn.is_available(), "cuDNN not available"
    block = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU()).cuda().eval()
    x = torch.randn(8, 3, 32, 32, device="cuda")
    with torch.no_grad():
        y = block(x)
    torch.cuda.synchronize()
    assert y.shape == (8, 16, 32, 32), f"unexpected output shape {tuple(y.shape)}"
    return f"cuDNN {torch.backends.cudnn.version()}, conv+BN+ReLU forward OK"


def torchvision_cuda():
    import torch
    import torchvision
    from torchvision.ops import nms
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    keep = nms(boxes, scores, 0.5)  # exercises torchvision's compiled CUDA ops
    assert keep.numel() == 2, f"NMS returned {keep.tolist()}, expected 2 boxes"
    return f"torchvision {torchvision.__version__}, CUDA ops OK"


def import_pkg(name):
    def _check():
        mod = __import__(name)
        return f"{name} {getattr(mod, '__version__', '?')}"
    return _check


def find_nvcc():
    path = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    assert path, "nvcc not found; add /usr/local/cuda/bin to PATH (and try: sudo apt install nvidia-jetpack)"
    out = subprocess.run([path, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release (\d+\.\d+)", out)
    assert m, "could not parse nvcc version"
    return f"{path} (CUDA {m.group(1)})"


def find_nsys():
    path = shutil.which("nsys")
    if not path:
        hits = glob.glob("/opt/nvidia/nsight-systems/*/bin/nsys") + glob.glob("/opt/nvidia/nsight-systems-cli/*/bin/nsys")
        path = hits[0] if hits else None
    assert path, "nsys not found (needed for the profiling step; usually installed with nvidia-jetpack)"
    return path


def kernel_compile():
    import torch
    assert shutil.which("ninja"), "ninja not found; run: pip install ninja"
    from torch.utils.cpp_extension import load_inline

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")  # Orin only; speeds up compile
    cuda_src = r"""
    __global__ void add_one_kernel(const float* x, float* y, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) y[i] = x[i] + 1.0f;
    }
    torch::Tensor add_one(torch::Tensor x) {
        auto y = torch::empty_like(x);
        int n = x.numel();
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        add_one_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
        return y;
    }
    """
    cpp_src = "torch::Tensor add_one(torch::Tensor x);"
    print("       (compiling a tiny CUDA kernel; the first run can take a minute or two...)")
    mod = load_inline(name="env_check_ext", cpp_sources=cpp_src, cuda_sources=cuda_src,
                      functions=["add_one"], verbose=False)
    x = torch.arange(1000, dtype=torch.float32, device="cuda")
    y = mod.add_one(x)
    torch.cuda.synchronize()
    assert torch.equal(y.cpu(), x.cpu() + 1), "custom kernel produced wrong results"
    return "custom CUDA kernel compiled, loaded, and ran correctly"


print("=== Jetson GPU project environment check ===\n")
check("Python version", python_version)
check("NumPy", numpy_version)
check("PyTorch build", torch_build)
check("CUDA device", cuda_available)
check("GPU matmul vs CPU", gpu_matmul)
check("cuDNN conv forward", cudnn_conv)
check("torchvision CUDA ops", torchvision_cuda)
check("pandas", import_pkg("pandas"))
check("matplotlib", import_pkg("matplotlib"))
check("nvcc", find_nvcc)
check("Nsight Systems (nsys)", find_nsys, required=False)
if SKIP_COMPILE:
    print("[SKIP] custom kernel compile (--skip-compile)")
else:
    check("Custom CUDA kernel compile", kernel_compile)

fails = [r for r in results if r[1] == "FAIL"]
warns = [r for r in results if r[1] == "WARN"]
print(f"\n=== {len(results) - len(fails) - len(warns)} OK, {len(warns)} warnings, {len(fails)} failures ===")
for name, _, detail in fails:
    print(f"  FAIL: {name}: {detail}")
sys.exit(1 if fails else 0)