"""Numerical tolerances for supported Torch dtypes."""

from __future__ import annotations

import torch


# Values are: symmetry tolerance, minimum reciprocal condition, identity rtol, identity atol.
DTYPE_TOLERANCES: dict[torch.dtype, tuple[float, float, float, float]] = {
    torch.float32: (1e-5, 1e-5, 1e-5, 1e-5),
    torch.float64: (1e-8, 1e-10, 1e-7, 1e-9)}
