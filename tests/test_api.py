from __future__ import annotations

import runpy
from pathlib import Path

import torch

import eigen_pid as eigen_pid_package
from eigen_pid import conditional_covariance, eigen_pid


COVARIANCE = torch.tensor([[1.0, 0.6, 0.4],
                           [0.6, 1.0, 0.2],
                           [0.4, 0.2, 1.0]], dtype=torch.float64)
RESULT_KEYS = ["mix", "miy", "mixy", "union_info", "uix", "uiy", "red", "syn",
               "gen_ev", "log2_gen_ev"]
OPTIMAL_KEYS = ["gamma_star", "optimal_cov", "optimal_cov_xy"]


def test_package_exports_public_functions() -> None:
    assert eigen_pid_package.__all__ == ["eigen_pid", "conditional_covariance", "debias_eigen_pid"]
    assert eigen_pid_package.eigen_pid is eigen_pid
    assert eigen_pid_package.conditional_covariance is conditional_covariance


def test_function_returns_documented_dictionary_surface() -> None:
    assert list(eigen_pid(COVARIANCE, 1, 1, 1)) == RESULT_KEYS
    assert list(eigen_pid(COVARIANCE, 1, 1, 1,
                          compute_optimal_covariance=True)) == RESULT_KEYS + OPTIMAL_KEYS


def test_basic_example_runs(capsys) -> None:
    example = Path(__file__).resolve().parents[1] / "basic_usage" / "basic_usage.py"
    runpy.run_path(str(example), run_name="__main__")
    output = capsys.readouterr().out
    assert "device:" in output
    assert "unique X:" in output
    assert "synergy:" in output
