from __future__ import annotations

import numpy as np


def swap_sources(
    covariance: np.ndarray, dm: int, dx: int, dy: int
) -> np.ndarray:
    indices = np.r_[
        np.arange(dm),
        np.arange(dm + dx, dm + dx + dy),
        np.arange(dm, dm + dx),
    ]
    return covariance[np.ix_(indices, indices)]


def merge_independent(
    cov1: np.ndarray,
    dims1: tuple[int, int, int],
    cov2: np.ndarray,
    dims2: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    dm1, dx1, dy1 = dims1
    dm2, dx2, dy2 = dims2
    raw = np.block(
        [
            [cov1, np.zeros((cov1.shape[0], cov2.shape[0]))],
            [np.zeros((cov2.shape[0], cov1.shape[0])), cov2],
        ]
    )
    m1 = np.arange(0, dm1)
    x1 = np.arange(dm1, dm1 + dx1)
    y1 = np.arange(dm1 + dx1, dm1 + dx1 + dy1)
    offset = cov1.shape[0]
    m2 = offset + np.arange(0, dm2)
    x2 = offset + np.arange(dm2, dm2 + dx2)
    y2 = offset + np.arange(dm2 + dx2, dm2 + dx2 + dy2)
    order = np.r_[m1, m2, x1, x2, y1, y2]
    return (
        raw[np.ix_(order, order)],
        (dm1 + dm2, dx1 + dx2, dy1 + dy2),
    )


def random_well_conditioned_transform(
    dimension: int, rng: np.random.Generator
) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    scales = np.linspace(0.7, 1.3, dimension)
    return q @ np.diag(scales)
