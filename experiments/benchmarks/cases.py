"""Deterministic population Gaussian covariance generators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

FloatMatrix = NDArray[np.float64]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    family: str
    dm: int
    dx: int
    dy: int
    seed: int
    condition_target: float = 1.0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GaussianCase:
    spec: CaseSpec
    covariance: FloatMatrix

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            covariance=self.covariance,
            metadata_json=np.array(json.dumps(asdict(self.spec), sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> "GaussianCase":
        with np.load(path, allow_pickle=False) as data:
            covariance = np.asarray(data["covariance"], dtype=np.float64)
            metadata = json.loads(str(data["metadata_json"].item()))
        return cls(CaseSpec(**metadata), covariance)


def rotation_2d(theta: float) -> FloatMatrix:
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )


def spd_matrix(
    dimension: int,
    condition_number: float,
    rng: np.random.Generator,
) -> FloatMatrix:
    """Generate an SPD matrix with a prescribed 2-norm condition number."""
    if dimension <= 0 or condition_number < 1.0:
        raise ValueError("invalid SPD matrix specification")
    if dimension == 1:
        return np.ones((1, 1), dtype=np.float64)
    if dimension > 256:
        eigenvalues = np.geomspace(1.0, 1.0 / condition_number, dimension)
        rng.shuffle(eigenvalues)
        return np.diag(eigenvalues).astype(np.float64)
    q, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    eigenvalues = np.geomspace(1.0, 1.0 / condition_number, dimension)
    result = (q * eigenvalues) @ q.T
    return np.asarray(0.5 * (result + result.T), dtype=np.float64)


def channel_covariance(
    sigma_m: FloatMatrix,
    hx: FloatMatrix,
    hy: FloatMatrix,
    noise_x: FloatMatrix,
    noise_y: FloatMatrix,
    noise_cross: FloatMatrix | None = None,
) -> FloatMatrix:
    """Assemble the exact covariance of ``M, X=Hx M+ex, Y=Hy M+ey``."""
    dm = sigma_m.shape[0]
    dx = noise_x.shape[0]
    dy = noise_y.shape[0]
    if sigma_m.shape != (dm, dm):
        raise ValueError("sigma_m must be square")
    if hx.shape != (dx, dm) or hy.shape != (dy, dm):
        raise ValueError("channel matrix dimensions do not match")
    if noise_y.shape != (dy, dy):
        raise ValueError("noise_y must be square")
    if noise_cross is None:
        noise_cross = np.zeros((dx, dy), dtype=np.float64)
    if noise_cross.shape != (dx, dy):
        raise ValueError("noise cross-covariance shape mismatch")
    sigma_mx = sigma_m @ hx.T
    sigma_my = sigma_m @ hy.T
    sigma_x = hx @ sigma_m @ hx.T + noise_x
    sigma_y = hy @ sigma_m @ hy.T + noise_y
    sigma_xy = hx @ sigma_m @ hy.T + noise_cross
    covariance = np.block(
        [
            [sigma_m, sigma_mx, sigma_my],
            [sigma_mx.T, sigma_x, sigma_xy],
            [sigma_my.T, sigma_xy.T, sigma_y],
        ]
    )
    return np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64)


def feasible_noise_cross(
    noise_x: FloatMatrix,
    noise_y: FloatMatrix,
    rng: np.random.Generator,
    strength: float,
) -> FloatMatrix:
    """Construct ``Lx K Ly.T`` with spectral norm of K strictly below one."""
    if not 0.0 <= strength < 1.0:
        raise ValueError("coupling strength must lie in [0, 1)")
    dx, dy = noise_x.shape[0], noise_y.shape[0]
    if strength == 0.0:
        return np.zeros((dx, dy), dtype=np.float64)
    k = rng.normal(size=(dx, dy))
    spectral_norm = float(linalg.svdvals(k)[0])
    k *= strength / spectral_norm
    lx = linalg.cholesky(noise_x, lower=True)
    ly = linalg.cholesky(noise_y, lower=True)
    return np.asarray(lx @ k @ ly.T, dtype=np.float64)


def _dense_case(spec: CaseSpec) -> FloatMatrix:
    rng = np.random.default_rng(spec.seed)
    sigma_m = spd_matrix(spec.dm, spec.condition_target, rng)
    noise_x = spd_matrix(spec.dx, spec.condition_target, rng)
    noise_y = spd_matrix(spec.dy, spec.condition_target, rng)
    gain = float(spec.parameters.get("gain", 0.8))
    hx = gain * rng.normal(size=(spec.dx, spec.dm)) / np.sqrt(spec.dm)
    hy = gain * rng.normal(size=(spec.dy, spec.dm)) / np.sqrt(spec.dm)
    coupling = float(spec.parameters.get("coupling", 0.0))
    cross = feasible_noise_cross(noise_x, noise_y, rng, coupling)
    return channel_covariance(sigma_m, hx, hy, noise_x, noise_y, cross)


def _structured_case(spec: CaseSpec) -> FloatMatrix:
    if not (spec.dm == spec.dx == spec.dy):
        return _dense_case(spec)
    dimension = spec.dm
    high = float(spec.parameters.get("high_gain", np.sqrt(3.0)))
    low = float(spec.parameters.get("low_gain", 0.5))
    hx_gain = np.where(np.arange(dimension) % 2 == 0, high, low)
    hy_gain = np.where(np.arange(dimension) % 2 == 0, low, high)
    return channel_covariance(
        np.eye(dimension),
        np.diag(hx_gain),
        np.diag(hy_gain),
        np.eye(dimension),
        np.eye(dimension),
    )


def _analytic_case(spec: CaseSpec) -> FloatMatrix:
    family = spec.family
    if family == "scalar":
        hx = np.array([[float(spec.parameters["gain_x"])]])
        hy = np.array([[float(spec.parameters["gain_y"])]])
        return channel_covariance(
            np.eye(1), hx, hy, np.eye(1), np.eye(1)
        )
    if family == "pure_uniqueness":
        return channel_covariance(
            np.eye(1),
            np.ones((1, 1)),
            np.zeros((1, 1)),
            np.eye(1),
            np.eye(1),
        )
    if family == "degraded":
        # X=M+Nx and Y=X+Nd, with Nx and Nd independent unit Gaussians.
        return channel_covariance(
            np.eye(1),
            np.ones((1, 1)),
            np.ones((1, 1)),
            np.eye(1),
            2.0 * np.eye(1),
            np.eye(1),
        )
    if family == "near_degraded":
        dimension = spec.dm
        delta = float(spec.parameters.get("delta", 1e-4))
        noise_x = np.eye(dimension)
        return channel_covariance(
            np.eye(dimension),
            np.eye(dimension),
            np.eye(dimension),
            noise_x,
            (1.0 + delta) * np.eye(dimension),
            noise_x,
        )
    if family == "opposed":
        return channel_covariance(
            np.eye(2),
            np.diag([np.sqrt(3.0), 0.0]),
            np.diag([0.0, np.sqrt(3.0)]),
            np.eye(2),
            np.eye(2),
        )
    if family == "venkatesh_example8":
        alpha = float(spec.parameters["alpha"])
        return channel_covariance(
            np.eye(2),
            np.diag([alpha, 1.0]),
            np.diag([1.0, 3.0]),
            np.eye(2),
            np.eye(2),
        )
    if family == "venkatesh_example9":
        theta = float(spec.parameters["theta"])
        gain = float(spec.parameters.get("gain", np.sqrt(2.0)))
        hx = np.diag([gain, 1.0]) @ rotation_2d(-theta)
        hy = np.diag([1.0, gain])
        return channel_covariance(
            np.eye(2), hx, hy, np.eye(2), np.eye(2)
        )
    if family == "near_neutral":
        epsilon = float(spec.parameters.get("epsilon", 1e-9))
        dimension = spec.dm
        hx = np.eye(dimension)
        hy = (1.0 + epsilon) * np.eye(dimension)
        return channel_covariance(
            np.eye(dimension), hx, hy, np.eye(dimension), np.eye(dimension)
        )
    raise ValueError(f"unknown analytic family {family!r}")


def build_case(spec: CaseSpec) -> GaussianCase:
    if spec.family in {
        "scalar",
        "pure_uniqueness",
        "degraded",
        "near_degraded",
        "opposed",
        "venkatesh_example8",
        "venkatesh_example9",
        "near_neutral",
    }:
        covariance = _analytic_case(spec)
    elif spec.family in {"structured", "repeated_block"}:
        covariance = _structured_case(spec)
    else:
        covariance = _dense_case(spec)
    expected = spec.dm + spec.dx + spec.dy
    if covariance.shape != (expected, expected):
        raise AssertionError("generator returned the wrong covariance shape")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise AssertionError("generator returned an asymmetric covariance")
    try:
        linalg.cholesky(covariance, lower=True, check_finite=True)
    except linalg.LinAlgError as exc:
        raise AssertionError(f"generator returned a non-SPD case {spec.case_id}") from exc
    return GaussianCase(spec, covariance)


def analytic_specs() -> list[CaseSpec]:
    specs = [
        CaseSpec("pure_uniqueness", "pure_uniqueness", 1, 1, 1, 0),
        CaseSpec("degraded", "degraded", 1, 1, 1, 0),
        CaseSpec("opposed", "opposed", 2, 2, 2, 0),
    ]
    for index, (gx, gy) in enumerate(((0.25, 2.0), (1.0, 1.0), (3.0, 0.5))):
        specs.append(
            CaseSpec(
                f"scalar_{index}",
                "scalar",
                1,
                1,
                1,
                index,
                parameters={"gain_x": gx, "gain_y": gy},
            )
        )
    for index, alpha in enumerate((0.0, 0.5, 0.99, 1.0, 1.5, 2.0, 3.0)):
        specs.append(
            CaseSpec(
                f"example8_alpha_{index}",
                "venkatesh_example8",
                2,
                2,
                2,
                index,
                parameters={"alpha": alpha},
            )
        )
    angles = np.r_[0.0, np.logspace(-1, np.log10(np.pi / 2.0), 9)]
    for index, theta in enumerate(angles):
        specs.append(
            CaseSpec(
                f"example9_theta_{index}",
                "venkatesh_example9",
                2,
                2,
                2,
                index,
                parameters={"theta": float(theta), "gain": np.sqrt(2.0)},
            )
        )
    for index, epsilon in enumerate((0.0, 1e-13, -1e-13, 1e-9, -1e-9)):
        specs.append(
            CaseSpec(
                f"near_neutral_{index}",
                "near_neutral",
                4,
                4,
                4,
                index,
                parameters={"epsilon": epsilon},
            )
        )
    return specs


def equivalence_specs(profile: str) -> list[CaseSpec]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    specs = analytic_specs()
    quick_dims = (1, 2, 4, 8, 16)
    conditions = (1.0, 10.0, 1e3)
    for index in range(50):
        dimension = quick_dims[index % len(quick_dims)]
        condition = conditions[index % len(conditions)]
        specs.append(
            CaseSpec(
                f"quick_dense_{index:03d}",
                "dense",
                dimension,
                dimension,
                dimension,
                10_000 + index,
                condition,
                {"coupling": 0.5 if index % 4 == 0 else 0.0},
            )
        )
    for index in range(10):
        dimension = 32 if index < 5 else 64
        specs.append(
            CaseSpec(
                f"quick_medium_{index:02d}",
                "structured" if index % 2 == 0 else "dense",
                dimension,
                dimension,
                dimension,
                20_000 + index,
                conditions[index % len(conditions)],
            )
        )
    if profile == "quick":
        return specs

    for index in range(200):
        dimension = 1 + index % 8
        specs.append(
            CaseSpec(
                f"full_small_{index:03d}",
                "dense",
                dimension,
                dimension,
                dimension,
                30_000 + index,
                conditions[index % len(conditions)],
                {"coupling": (index % 8) / 10.0},
            )
        )
    for index in range(100):
        dimension = 9 + index % 24
        specs.append(
            CaseSpec(
                f"full_mid_{index:03d}",
                "dense",
                dimension,
                dimension,
                dimension,
                40_000 + index,
                conditions[index % len(conditions)],
            )
        )
    for index in range(30):
        specs.append(
            CaseSpec(
                f"full_64_{index:02d}",
                "dense" if index % 2 else "structured",
                64,
                64,
                64,
                50_000 + index,
                conditions[index % len(conditions)],
            )
        )
    for dimension in (128, 256, 512, 768, 1000):
        specs.append(
            CaseSpec(
                f"full_structured_{dimension}",
                "structured",
                dimension,
                dimension,
                dimension,
                60_000 + dimension,
            )
        )
        if dimension <= 256:
            specs.append(
                CaseSpec(
                    f"full_dense_{dimension}",
                    "dense",
                    dimension,
                    dimension,
                    dimension,
                    70_000 + dimension,
                    10.0,
                )
            )
    for dm, dx, dy in (
        (100, 1000, 100),
        (100, 100, 1000),
        (1000, 100, 100),
        (100, 1000, 10),
        (100, 10, 1000),
        (10, 1000, 1000),
    ):
        specs.append(
            CaseSpec(
                f"full_imbalanced_{dm}_{dx}_{dy}",
                "imbalanced",
                dm,
                dx,
                dy,
                80_000 + dm + dx + dy,
                10.0,
            )
        )
    for index, strength in enumerate((0.0, 0.2, 0.5, 0.8)):
        specs.append(
            CaseSpec(
                f"coupling_sweep_{index}",
                "coupling_sweep",
                8, 8, 8, 140_000, 10.0,
                {"coupling": strength},
            )
        )
    for index, delta in enumerate((1e-2, 1e-4, 1e-6)):
        specs.append(
            CaseSpec(
                f"near_degraded_{index}",
                "near_degraded",
                4, 4, 4, 150_000 + index,
                parameters={"delta": delta},
            )
        )
    for dimension in (8, 16, 32):
        specs.append(
            CaseSpec(
                f"repeated_block_{dimension}",
                "repeated_block",
                dimension, dimension, dimension, 160_000 + dimension,
            )
        )
    for condition in (1e6, 1e9):
        specs.append(
            CaseSpec(
                f"stress_condition_{condition:.0e}",
                "conditioning_stress",
                16,
                16,
                16,
                90_000 + int(np.log10(condition)),
                condition,
            )
        )
    return specs


def benchmark_specs(profile: str) -> list[CaseSpec]:
    balanced = (1, 2, 4, 8, 16, 32, 64) if profile == "quick" else (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        768,
        1000,
    )
    specs = [
        CaseSpec(
            f"benchmark_balanced_{dimension}",
            "structured",
            dimension,
            dimension,
            dimension,
            100_000 + dimension,
        )
        for dimension in balanced
    ]
    if profile == "quick":
        specs.extend(
            [
                CaseSpec("benchmark_source_heavy", "source_heavy", 10, 64, 64, 101_001, 10.0),
                CaseSpec("benchmark_target_heavy", "target_heavy", 64, 10, 10, 101_002, 10.0),
                CaseSpec("benchmark_imbalanced", "imbalanced", 32, 64, 8, 101_003, 10.0),
            ]
        )
        return specs
    for dm in (2, 10, 50, 100):
        for source_dimension in (32, 64, 128, 256, 512, 768, 1000):
            specs.append(
                CaseSpec(
                    f"benchmark_source_heavy_{dm}_{source_dimension}",
                    "source_heavy",
                    dm,
                    source_dimension,
                    source_dimension,
                    110_000 + dm + source_dimension,
                    10.0,
                )
            )
    for target_dimension in (32, 64, 128, 256, 512, 768, 1000):
        for source_dimension in (2, 10, 50, 100):
            specs.append(
                CaseSpec(
                    f"benchmark_target_heavy_{target_dimension}_{source_dimension}",
                    "target_heavy",
                    target_dimension,
                    source_dimension,
                    source_dimension,
                    120_000 + target_dimension + source_dimension,
                    10.0,
                )
            )
    for dm, dx, dy in (
        (100, 1000, 100),
        (100, 100, 1000),
        (1000, 100, 100),
        (100, 1000, 10),
        (100, 10, 1000),
        (10, 1000, 1000),
    ):
        specs.append(
            CaseSpec(
                f"benchmark_imbalanced_{dm}_{dx}_{dy}",
                "imbalanced",
                dm,
                dx,
                dy,
                130_000 + dm + dx + dy,
                10.0,
            )
        )
    return specs


def write_manifest(specs: Iterable[CaseSpec], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "family",
                "seed",
                "dm",
                "dx",
                "dy",
                "condition_target",
                "parameters_json",
            ),
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "case_id": spec.case_id,
                    "family": spec.family,
                    "seed": spec.seed,
                    "dm": spec.dm,
                    "dx": spec.dx,
                    "dy": spec.dy,
                    "condition_target": spec.condition_target,
                    "parameters_json": json.dumps(
                        spec.parameters, sort_keys=True
                    ),
                }
            )
