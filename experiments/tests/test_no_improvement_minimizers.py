"""Focused checks for the external optimizers' minimal initialization hook."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.adapters import VENKATESH_ROOT, load_venkatesh, load_zhao
from benchmarks.cases import channel_covariance
from eigen_pid import eigen_pid

pytestmark = pytest.mark.external


def _extract_s_star(module, optimal_cov, dm, dx, dy):
    conditional_sources = module.whiten(
        optimal_cov, dm, dx, dy, ret_channel_params=True
    )[-1]
    return conditional_sources[:dx, dx:]


@pytest.mark.parametrize("method", ("venkatesh", "zhao"))
def test_init_cov_preserves_default_and_sets_exact_first_state(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if method == "venkatesh" and not (VENKATESH_ROOT / "src" / "gpid").exists():
        pytest.skip("pinned Venkatesh repository is unavailable")
    cov = channel_covariance(
        np.eye(2), np.array([[1.6, .2], [.1, .5]]),
        np.array([[.4, -.3], [.2, 1.4]]), np.eye(2), np.eye(2))
    function, module, _ = (load_venkatesh() if method == "venkatesh"
                           else load_zhao())
    legacy = function(cov, 2, 2, 2)
    explicit_none = function(cov, 2, 2, 2, init_cov=None)
    np.testing.assert_array_equal(legacy, explicit_none)

    optimal_cov = np.asarray(
        eigen_pid(cov, 2, 2, 2, compute_optimal_covariance=True)["optimal_cov"]
    )
    s_star = _extract_s_star(module, optimal_cov, 2, 2, 2)
    first_state = []
    original_objective = module.objective

    def capture(sig, *args, **kwargs):
        if not first_state:
            first_state.append(np.asarray(sig).copy())
        return original_objective(sig, *args, **kwargs)

    monkeypatch.setattr(module, "objective", capture)
    function(cov, 2, 2, 2, init_cov=optimal_cov)
    np.testing.assert_array_equal(first_state[0], s_star)

    invalid = optimal_cov.copy()
    invalid[0, 0] += 1e-4
    with pytest.raises(ValueError, match="preserve"):
        function(cov, 2, 2, 2, init_cov=invalid)


def test_zhao_transposes_supplied_initialization_for_internal_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cov = channel_covariance(
        np.eye(2), np.array([[1., .2]]),
        np.array([[.4, -.3], [.2, 1.4]]), np.eye(1), np.eye(2))
    function, module, _ = load_zhao()
    optimal_cov = np.asarray(
        eigen_pid(cov, 2, 1, 2, compute_optimal_covariance=True)["optimal_cov"]
    )
    s_star = _extract_s_star(module, optimal_cov, 2, 1, 2)
    first_state = []
    original_objective = module.objective

    def capture(sig, *args, **kwargs):
        if not first_state:
            first_state.append(np.asarray(sig).copy())
        return original_objective(sig, *args, **kwargs)

    monkeypatch.setattr(module, "objective", capture)
    function(cov, 2, 1, 2, init_cov=optimal_cov)
    np.testing.assert_array_equal(first_state[0], s_star.T)
