"""Run a small Torch Eigen-PID decomposition."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eigen_pid import eigen_pid


device = "cuda" if torch.cuda.is_available() else "cpu"
covariance = torch.tensor([[1.0, 0.6, 0.4],
                           [0.6, 1.0, 0.2],
                           [0.4, 0.2, 1.0]], dtype=torch.float64, device=device)
result = eigen_pid(covariance, dm=1, dx=1, dy=1)

print(f"device:     {device}")
print(f"unique X:   {result['uix'].item():.9f} bits")
print(f"unique Y:   {result['uiy'].item():.9f} bits")
print(f"redundancy: {result['red'].item():.9f} bits")
print(f"synergy:    {result['syn'].item():.9f} bits")


#Numpy version for comparison
import numpy as np

covariance_np = np.array([[1.0, 0.6, 0.4],
                          [0.6, 1.0, 0.2],
                          [0.4, 0.2, 1.0]], dtype=np.float64)
result_np = eigen_pid(covariance_np, dm=1, dx=1, dy=1)

print(f"library:     numpy")
print(f"unique X:   {result_np['uix'].item():.9f} bits")
print(f"unique Y:   {result_np['uiy'].item():.9f} bits")
print(f"redundancy: {result_np['red'].item():.9f} bits")
print(f"synergy:    {result_np['syn'].item():.9f} bits")