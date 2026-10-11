"""Compare float32 accuracy and runtime of Torch CPU and GPU."""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eigen_pid import eigen_pid
from eigen_pid.tolerances import DTYPE_TOLERANCES


DM, DX, DY = 3000, 3000, 3000
REPEATS = 10

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this comparison")

total_dimension = DM + DX + DY
generator = torch.Generator().manual_seed(2026)
factor = torch.randn((total_dimension, total_dimension), dtype=torch.float32,
                     generator=generator) / total_dimension**0.5
covariance = factor @ factor.mT + torch.eye(total_dimension)
standard_deviations = torch.sqrt(torch.diagonal(covariance))
covariance /= standard_deviations[:, None] * standard_deviations[None, :]
gpu_covariance = covariance.cuda()

# Warm up both devices; allocation and device transfer are excluded.
eigen_pid(covariance, DM, DX, DY)
eigen_pid(gpu_covariance, DM, DX, DY)
torch.cuda.synchronize()

cpu_times = []
for _ in range(REPEATS):
    start = perf_counter()
    cpu_result = eigen_pid(covariance, DM, DX, DY)
    cpu_times.append(perf_counter() - start)

torch_gpu_times = []
for _ in range(REPEATS):
    torch.cuda.synchronize()
    start = perf_counter()
    torch_gpu_result = eigen_pid(gpu_covariance, DM, DX, DY)
    torch.cuda.synchronize()
    torch_gpu_times.append(perf_counter() - start)

_, _, rtol, atol = DTYPE_TOLERANCES[torch.float32]
maximum_torch_difference = 0.0
print(f"Dimensions: M={DM}, X={DX}, Y={DY}; covariance={tuple(covariance.shape)}")
print(f"{'result':<20} {'Torch GPU vs CPU':>20}")
for name, cpu_value in cpu_result.items():
    torch_difference = torch.max(torch.abs(cpu_value - torch_gpu_result[name].cpu())).item()
    maximum_torch_difference = max(maximum_torch_difference, torch_difference)
    print(f"{name:<20} {torch_difference:>20.3e}")
    torch.testing.assert_close(cpu_value, torch_gpu_result[name].cpu(), rtol=rtol, atol=atol)


cpu_seconds = median(cpu_times)
torch_gpu_seconds = median(torch_gpu_times)
print(f"PASS: maximum Torch GPU difference {maximum_torch_difference:.3e}")
print(f"Torch CPU median: {cpu_seconds:.6f} seconds")
print(f"Torch GPU median: {torch_gpu_seconds:.6f} seconds")
timings = {"Torch CPU": cpu_seconds, "Torch GPU": torch_gpu_seconds}
fastest = min(timings, key=timings.get)
print(f"Fastest backend: {fastest}")
