"""Load pinned upstream solvers in isolated benchmark workers; no solver is vendored."""

import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import types


COMMITS = {"tilde": "179fd78ef426c34837e23d95c31db0293e74585d",
           "thin": "4a30b1a3ff3eb9d181fc49dff023930a617a86b3"}


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_solver(solver, checkout):
    """Import upstream whitening and PID functions without importing flow models."""
    checkout = Path(checkout).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    assert commit == COMMITS[solver], f"Use the documented {solver} commit: {COMMITS[solver]}"
    if solver == "tilde":
        sys.path.insert(0, str(checkout / "src"))
        return importlib.import_module("gpid.tilde_pid").exact_gauss_tilde_pid, "direct"
    assert solver == "thin"
    # Thin-PID's initializer imports neural-flow dependencies not used by Gaussian PID.
    package = types.ModuleType("pid")
    package.__path__ = [str(checkout / "pid")]
    package.__package__ = "pid"
    sys.modules["pid"] = package
    estimate = load_file("_thin_estimate_channel", checkout / "utils/estimate_channel.py")
    linalg = load_file("_thin_linalg", checkout / "utils/linalg.py")
    utilities = types.ModuleType("utils")
    utilities.whiten, utilities.pinv = estimate.whiten, linalg.pinv
    sys.modules["utils"] = utilities
    return importlib.import_module("pid.thin_pid").exact_gauss_thin_pid, "gaussian_utility_shim"
