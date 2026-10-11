from __future__ import annotations

import numpy as np
import pytest

from benchmarks.adapters import ZHAO_ROOT, _run_method, load_zhao
from benchmarks.cases import CaseSpec, build_case
from eigen_pid import eigen_pid

pytestmark = pytest.mark.external


def _available() -> None:
    if not (ZHAO_ROOT / "pid" / "thin_pid.py").exists():
        pytest.skip("bundled Thin-PID source is absent")


def test_zhao_adapter_is_gaussian_only_shim() -> None:
    _available()
    function, _, adapter_mode = load_zhao()
    assert callable(function)
    assert adapter_mode == "gaussian_utility_shim"


@pytest.mark.parametrize(
    "spec",
    [
        CaseSpec("scalar", "scalar", 1, 1, 1, 0, parameters={"gain_x": 2.0, "gain_y": 0.5}),
        CaseSpec("degraded", "degraded", 1, 1, 1, 0),
        CaseSpec("rotation", "venkatesh_example9", 2, 2, 2, 0, parameters={"theta": 0.4}),
    ],
)
def test_external_zhao_matches_eigen_integration(spec: CaseSpec) -> None:
    _available()
    case = build_case(spec)
    expected = eigen_pid(case.covariance, spec.dm, spec.dx, spec.dy)
    actual, details, elapsed = _run_method(
        "zhao_thin", case.covariance, spec.dm, spec.dx, spec.dy
    )
    for field, expected_field in (
        ("imx", "mix"),
        ("imy", "miy"),
        ("imxy", "mixy"),
        ("union_info", "union_info"),
        ("uix", "uix"),
        ("uiy", "uiy"),
        ("redundancy", "red"),
        ("synergy", "syn"),
    ):
        assert np.isclose(
            actual[field], expected[expected_field], rtol=1e-5, atol=1e-7
        ), (spec.case_id, field, actual[field], expected[expected_field])
    assert details["adapter_mode"] == "gaussian_utility_shim"
    assert elapsed >= 0.0
