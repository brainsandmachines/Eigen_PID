"""In-process adapters and the shared three-method PID runner."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from eigen_pid import eigen_pid
import torch

ROOT = Path(__file__).resolve().parents[1]
VENKATESH_ROOT = ROOT / "external" / "gpid"
ZHAO_ROOT = ROOT / "external" / "flow-pid"
COMMITS = {
    "eigen_math": "local",
    "flow_tilde": "4a30b1a3ff3eb9d181fc49dff023930a617a86b3",
    "venkatesh_tilde": "6a2b15e637681f329590755464ee01c11dd66d27",
    "zhao_thin": "4a30b1a3ff3eb9d181fc49dff023930a617a86b3",
}
_LOADED: dict[str, tuple[Callable[..., Any], types.ModuleType, str]] = {}
_VALUE_FIELDS = (
    ("mutual_information_x", "imx"),
    ("mutual_information_y", "imy"),
    ("mutual_information_xy", "imxy"),
    ("union_information", "union_info"),
    ("objective", "obj"),
    ("unique_x", "uix"),
    ("unique_y", "uiy"),
    ("redundancy", "redundancy"),
    ("synergy", "synergy"),
)
_PID_ATOMS = ("redundancy", "unique_x", "unique_y", "synergy")
_REGULAR_TILDE = {
    "flow": "flow_tilde",
    "venkatesh": "venkatesh_tilde",
}
_OPTIMAL_PLAN = (
    ("analytical", "eigen_math", "default", True),
    ("thin", "zhao_thin", "optimal_cov", False),
    ("tilde", "venkatesh_tilde", "optimal_cov", False),
)



def _load_file_module(name: str, path: Path) -> types.ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_venkatesh() -> tuple[Callable[..., Any], types.ModuleType, str]:
    """Load only the pinned Venkatesh package."""
    if "venkatesh_tilde" in _LOADED:
        return _LOADED["venkatesh_tilde"]
    source = VENKATESH_ROOT / "src"
    if not source.exists():
        raise FileNotFoundError(
            f"Bundled Tilde-PID source is missing from {VENKATESH_ROOT}"
        )
    sys.path.insert(0, str(source))
    module = importlib.import_module("gpid.tilde_pid")
    loaded = (module.exact_gauss_tilde_pid, module, "direct")
    _LOADED["venkatesh_tilde"] = loaded
    return loaded


def load_zhao() -> tuple[Callable[..., Any], types.ModuleType, str]:
    """Load Zhao's Thin-PID without executing its flow-importing initializer."""
    if "zhao_thin" in _LOADED:
        return _LOADED["zhao_thin"]
    if not (ZHAO_ROOT / "pid" / "thin_pid.py").exists():
        raise FileNotFoundError(
            f"Bundled Thin-PID source is missing from {ZHAO_ROOT}"
        )

    package = types.ModuleType("pid")
    package.__path__ = [str(ZHAO_ROOT / "pid")]  # type: ignore[attr-defined]
    package.__package__ = "pid"
    sys.modules["pid"] = package

    # estimate_channel imports cvxpy for unrelated estimators. A placeholder is
    # sufficient when cvxpy is absent because the repository's whiten function
    # does not access it. No flow or neural dependency is imported.
    if "cvxpy" not in sys.modules and importlib.util.find_spec("cvxpy") is None:
        sys.modules["cvxpy"] = types.ModuleType("cvxpy")
    estimate = _load_file_module(
        "_zhao_gaussian_estimate_channel",
        ZHAO_ROOT / "utils" / "estimate_channel.py",
    )
    linalg_module = _load_file_module(
        "_zhao_gaussian_linalg", ZHAO_ROOT / "utils" / "linalg.py"
    )
    utility_shim = types.ModuleType("utils")
    utility_shim.whiten = estimate.whiten  # type: ignore[attr-defined]
    utility_shim.pinv = linalg_module.pinv  # type: ignore[attr-defined]
    utility_shim.__doc__ = (
        "Gaussian-only adapter exposing Zhao repository implementations of "
        "whiten and pinv."
    )
    sys.modules["utils"] = utility_shim

    module = importlib.import_module("pid.thin_pid")
    loaded = (module.exact_gauss_thin_pid, module, "gaussian_utility_shim")
    _LOADED["zhao_thin"] = loaded
    return loaded


def _prepare_method(method: str) -> None:
    """Import method-specific code before benchmark timing starts."""
    if method == "venkatesh_tilde":
        load_venkatesh()
    elif method == "flow_tilde":
        load_zhao()
        importlib.import_module("pid.tilde_pid")
    elif method == "zhao_thin":
        load_zhao()


def _run_method(
    method: str,
    covariance: np.ndarray,
    dm: int,
    dx: int,
    dy: int,
    init_cov: np.ndarray | None = None,
    *,
    include_optimal_cov: bool = False,
    add_reg: bool = False,
) -> tuple[dict[str, float], dict[str, Any], float]:
    """Execute one complete method and return values, diagnostics, and seconds.

    ``include_optimal_cov`` exposes Eigen's already-computed optimal covariance
    to in-process callers without adding it to normal benchmark diagnostics.
    """
    if include_optimal_cov and method != "eigen_math":
        raise ValueError("include_optimal_cov is available only for eigen_math")
    if method == "flow_tilde" and init_cov is not None:
        raise ValueError("init_cov is unavailable for flow_tilde")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        started = time.perf_counter_ns()
        if method == "eigen_math":
            result = eigen_pid(
                covariance,
                dm,
                dx,
                dy,
                compute_optimal_covariance=include_optimal_cov,
                add_reg=add_reg,
                
            )
            values = {
                "imx": float(result["mix"].cpu()),
                "imy": float(result["miy"].cpu()),
                "imxy": float(result["mixy"].cpu()),
                "union_info": float(result["union_info"].cpu()),
                "obj": float(result["union_info"].cpu()),
                "uix": float(result["uix"].cpu()),
                "uiy": float(result["uiy"].cpu()),
                "redundancy": float(result["red"].cpu()),
                "synergy": float(result["syn"].cpu()),
            }
            details = {
                "adapter_mode": "direct",
                "eigenvalues": np.asarray(result["gen_ev"].cpu()).tolist(),
                "log2_eigenvalues": np.asarray(result["log2_gen_ev"].cpu()).tolist(),
            }
            if include_optimal_cov:
                details["optimal_cov"] = result["optimal_cov"].detach().cpu().numpy()
        else:
            if method == "venkatesh_tilde":
                function, _, adapter_mode = load_venkatesh()
            elif method == "flow_tilde":
                load_zhao()
                function = importlib.import_module("pid.tilde_pid").exact_gauss_tilde_pid
                adapter_mode = "gaussian_utility_shim"
            elif method == "zhao_thin":
                function, _, adapter_mode = load_zhao()
            else:
                raise ValueError(method)
            method_kwargs = {} if method == "flow_tilde" else {"init_cov": init_cov}
            result = function(
                covariance,
                dm,
                dx,
                dy,
                verbose=False,
                ret_t_sigt=False,
                plot=False,
                unbiased=False,
                sample_size=None,
                **method_kwargs,
            )
        if method != "eigen_math":
            if len(result) != 9:
                raise ValueError(f"expected a nine-element result, got {len(result)}")
            values = dict(
                zip(
                    ("imx", "imy", "imxy", "union_info", "obj", "uix", "uiy",
                     "redundancy", "synergy"),
                    map(float, result),
                    strict=True,
                )
            )
            details = {"adapter_mode": adapter_mode}
        elapsed = (time.perf_counter_ns() - started) / 1e9
    warning_counts: dict[str, int] = {}
    for item in caught:
        message = f"{item.category.__name__}: {item.message}"
        warning_counts[message] = warning_counts.get(message, 0) + 1
    details["warnings"] = [
        f"{message} (repeated {count} times)" if count > 1 else message
        for message, count in list(warning_counts.items())[:20]
    ]
    if len(warning_counts) > 20:
        details["warnings"].append(
            f"{len(warning_counts) - 20} additional distinct warnings omitted"
        )
    return values, details, elapsed


def _failed_row(
    name: str,
    method: str,
    initialization: str,
    status: str,
    error: str,
    elapsed: float = float("nan"),
) -> dict[str, object]:
    return {
        "method": name,
        "adapter_method": method,
        "initialization": initialization,
        **{field: float("nan") for field, _ in _VALUE_FIELDS},
        "runtime_seconds": elapsed,
        "status": status,
        "warnings": "",
        "error": error,
        "maximum_pid_identity_residual": float("nan"),
        "minimum_pid_atom": float("nan"),
    }


def _pid_diagnostics(values: dict[str, float]) -> tuple[str, float, float]:
    if not all(np.isfinite(value) for value in values.values()):
        return "NONFINITE", float("nan"), float("nan")
    residuals = (
        values["mutual_information_x"] - values["redundancy"] - values["unique_x"],
        values["mutual_information_y"] - values["redundancy"] - values["unique_y"],
        values["mutual_information_xy"] - sum(values[atom] for atom in _PID_ATOMS),
    )
    maximum_residual = max(map(abs, residuals))
    minimum_atom = min(values[atom] for atom in _PID_ATOMS)
    if maximum_residual > 1e-7:
        return "IDENTITY_VIOLATION", maximum_residual, minimum_atom
    if minimum_atom < -1e-7:
        return "NEGATIVE_ATOM", maximum_residual, minimum_atom
    return "SUCCESS", maximum_residual, minimum_atom


def _run_row(
    specification: tuple[str, str, str, bool],
    covariance: np.ndarray,
    dm: int,
    dx: int,
    dy: int,
    optimal_cov: np.ndarray | None,
    add_reg: bool = False,
) -> tuple[dict[str, object], np.ndarray | None]:
    name, method, initialization, capture_optimal = specification
    if initialization == "optimal_cov" and optimal_cov is None:
        return _failed_row(
            name,
            method,
            initialization,
            "SKIPPED",
            "Eigen-PID did not return optimal_cov",
        ), None

    started: float | None = None
    details: dict[str, Any] = {}
    if name == "analytical_gpu":
        #Change covarianace to GPU array
        cov = torch.from_numpy(covariance).to("cuda")
    else:
        cov = covariance
    try:
        _prepare_method(method)
        started = time.perf_counter()
        raw, details, elapsed = _run_method(
            method,
            cov,
            dm,
            dx,
            dy,
            init_cov=optimal_cov if initialization == "optimal_cov" else None,
            include_optimal_cov=capture_optimal,
            add_reg=add_reg
        )
        values = {field: raw[source] for field, source in _VALUE_FIELDS}
        status, maximum_residual, minimum_atom = _pid_diagnostics(values)
    except Exception as error:  # Preserve failures without changing solvers.
        elapsed = float("nan") if started is None else time.perf_counter() - started
        return _failed_row(
            name,
            method,
            initialization,
            "EXCEPTION",
            f"{type(error).__name__}: {error}",
            elapsed,
        ), optimal_cov

    return {
        "method": name,
        "adapter_method": method,
        "initialization": initialization,
        **values,
        "runtime_seconds": elapsed,
        "status": status,
        "warnings": " | ".join(details.get("warnings", [])),
        "error": "",
        "maximum_pid_identity_residual": maximum_residual,
        "minimum_pid_atom": minimum_atom,
    }, details.get("optimal_cov", optimal_cov)


def _add_analytical_differences(rows: list[dict[str, object]]) -> None:
    analytical = rows[0]
    for row in rows:
        for atom in _PID_ATOMS:
            difference = float(row[atom]) - float(analytical[atom])
            row[f"{atom}_difference_vs_analytical"] = difference
            row[f"{atom}_absolute_error_vs_analytical"] = abs(difference)


def run_pid_methods(
    covariance: np.ndarray,
    dm: int,
    dx: int,
    dy: int,
    *,
    mode: str = "regular",
    regular_tilde: str = "venkatesh",
    methods = ('analytical', 'thin', 'tilde'),
    add_reg: bool = False,
) -> list[dict[str, object]]:
    """Run Eigen, Thin, and Tilde PID and return uniform result rows."""
    if mode not in {"regular", "optimal"}:
        raise ValueError("mode must be 'regular' or 'optimal'")
    if regular_tilde not in _REGULAR_TILDE:
        raise ValueError("regular_tilde must be 'flow' or 'venkatesh'")

    plan_anlytical = ("analytical", "eigen_math", "default", False)
    plan_analytical_gpu = ("analytical_gpu", "eigen_math", "default", False)
    plan_thin = ("thin", "zhao_thin", "default", False)
    plan_tilde = ("tilde", _REGULAR_TILDE[regular_tilde], "default", False)
    regular_plan = ()

    for method in methods:
        if method == 'analytical':
            regular_plan += (plan_anlytical,)
            if torch.cuda.is_available():
                regular_plan += (plan_analytical_gpu,)
        elif method == 'thin':
            regular_plan += (plan_thin,)
        elif method == 'tilde':
            regular_plan += (plan_tilde,)
        else:
            raise ValueError(f"Unknown method: {method}")

    plan = {"regular": regular_plan, "optimal": _OPTIMAL_PLAN}[mode]
    rows: list[dict[str, object]] = []
    optimal_cov = None
    for specification in plan:
        row, optimal_cov = _run_row(
            specification, covariance, dm, dx, dy, optimal_cov,add_reg=add_reg
        )
        rows.append(row)
    _add_analytical_differences(rows)
    return rows
