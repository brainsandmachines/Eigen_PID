from __future__ import annotations

import inspect

import numpy as np
import pytest

from benchmarks.adapters import VENKATESH_ROOT, load_venkatesh
from benchmarks.cases import CaseSpec, build_case
from eigen_pid import eigen_pid

pytestmark = pytest.mark.external


def _available() -> None:
    if not (VENKATESH_ROOT / "src" / "gpid" / "tilde_pid.py").exists():
        pytest.skip("bundled Tilde-PID source is absent")


def test_external_signature_has_optional_init_cov() -> None:
    _available()
    external, _, _ = load_venkatesh()
    parameters = tuple(inspect.signature(external).parameters.values())
    assert tuple(parameter.name for parameter in parameters) == (
        "cov",
        "dm",
        "dx",
        "dy",
        "verbose",
        "ret_t_sigt",
        "plot",
        "unbiased",
        "sample_size",
        "init_cov",
    )
    assert parameters[-1].name == "init_cov" and parameters[-1].default is None


@pytest.mark.parametrize(
    "spec",
    [
        CaseSpec("scalar", "scalar", 1, 1, 1, 0, parameters={"gain_x": 2.0, "gain_y": 0.5}),
        CaseSpec("degraded", "degraded", 1, 1, 1, 0),
        CaseSpec("rotation", "venkatesh_example9", 2, 2, 2, 0, parameters={"theta": 0.4}),
    ],
)
def test_external_venkatesh_matches_eigen_integration(
    spec: CaseSpec,
) -> None:
    _available()
    case = build_case(spec)
    external, _, _ = load_venkatesh()
    expected = eigen_pid(case.covariance, spec.dm, spec.dx, spec.dy)
    actual = external(case.covariance, spec.dm, spec.dx, spec.dy)
    for index, field in (
        (0, "mix"),
        (1, "miy"),
        (2, "mixy"),
        (3, "union_info"),
        (5, "uix"),
        (6, "uiy"),
        (7, "red"),
        (8, "syn"),
    ):
        assert np.isclose(
            actual[index], expected[field], rtol=1e-5, atol=1e-7
        ), (spec.case_id, field, actual[index], expected[field])
