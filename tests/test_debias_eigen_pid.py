"""Numerical checks against independent NumPy/SciPy calculations."""

import numpy as np
import pytest
import torch
from scipy.special import digamma

from eigen_pid import debias_eigen_pid
from eigen_pid.debias_eigen_pid import components, entropy_bias, estimate_covariance, pid_from_covariance


def sample():
    rng = np.random.default_rng(17)
    a = rng.normal(size=(6, 6))
    return torch.from_numpy(rng.normal(size=(100, 6)) @ np.linalg.cholesky(a @ a.T + np.eye(6)).T)


@pytest.mark.parametrize("d,n", [(1, 20), (7, 100), (3704, 9841)])
def test_entropy_bias_matches_scipy(d, n):
    expected = (d * np.log(2 / (n - 1)) + digamma((n - np.arange(1, d + 1)) / 2).sum()) / (2 * np.log(2))
    assert entropy_bias(d, n) == pytest.approx(expected, abs=1e-10)


def test_analytical_bias_matches_wishart_simulation():
    samples = np.random.default_rng(42).normal(size=(2000, 100, 3))
    centered = samples - samples.mean(axis=1, keepdims=True)
    cov = centered.transpose(0, 2, 1) @ centered / 99
    entropies = np.linalg.slogdet(cov)[1] / (2 * np.log(2))
    assert abs(entropies.mean() - entropy_bias(3, 100)) < 5 * entropies.std(ddof=1) / np.sqrt(len(entropies))


def test_covariance_and_source_swap():
    data = sample()
    cov = estimate_covariance(data)
    np.testing.assert_allclose(cov.numpy(), np.cov(data.numpy(), rowvar=False), rtol=1e-12, atol=1e-12)
    raw = pid_from_covariance(cov, 2, 3, 1)
    order = [0, 1, 5, 2, 3, 4]
    swapped = pid_from_covariance(cov[order][:, order], 2, 1, 3)
    assert raw["uix"] == pytest.approx(swapped["uiy"])
    assert raw["uiy"] == pytest.approx(swapped["uix"])
    assert raw["syn"] == pytest.approx(swapped["syn"])


def test_resampling_and_correction_identities():
    data = sample()
    state = torch.random.get_rng_state()
    result = debias_eigen_pid(data, 2, 3, 1, n_shuffles=3, n_resamples=2, seed=5)
    assert torch.equal(state, torch.random.get_rng_state())
    repeated = debias_eigen_pid(data, 2, 3, 1, n_shuffles=3, n_resamples=2, seed=5)
    assert result["surrogate_unions"] == repeated["surrogate_unions"]

    # Replay the draws using NumPy covariance, independently of Torch's estimator.
    rng = torch.Generator().manual_seed(5)
    factor = torch.linalg.cholesky(torch.from_numpy(np.cov(data.numpy(), rowvar=False)))
    for kind, count in (("shuffle", 3), ("bootstrap", 2)):
        reference = []
        for _ in range(count):
            if kind == "shuffle":
                surrogate = data.clone()
                surrogate[:, :2] = data[torch.randperm(len(data), generator=rng), :2]
            else:
                surrogate = torch.randn(data.shape, dtype=data.dtype, generator=rng) @ factor.T
            reference.append(pid_from_covariance(torch.from_numpy(np.cov(surrogate.numpy(), rowvar=False)), 2, 3, 1)["union_info"])
        np.testing.assert_allclose(result["surrogate_unions"][kind], reference, atol=1e-10)
    raw = result["raw"]
    shuffled = np.mean(result["surrogate_unions"]["shuffle"])
    bootstrapped = np.mean(result["surrogate_unions"]["bootstrap"]) - raw["union_info"]
    assert raw["union_info"] - result["combined"]["union_info"] == pytest.approx((shuffled + bootstrapped) / 2)
    assert raw["union_info"] - result["permutation"]["union_info"] == pytest.approx(shuffled)
    for method in ("raw", "combined", "permutation"):
        v = result[method]
        assert v["uix"] + v["red"] == pytest.approx(v["mix"])
        assert v["uiy"] + v["red"] == pytest.approx(v["miy"])
        assert sum(v[k] for k in ("uix", "uiy", "red", "syn")) == pytest.approx(v["mixy"])
    assert components(0, 0, 0, 1)["red"] == -1


def test_shrinkage():
    from sklearn.covariance import LedoitWolf
    data = sample()
    np.testing.assert_allclose(estimate_covariance(data, "ledoit-wolf"), LedoitWolf().fit(data).covariance_)
    result = debias_eigen_pid(data, 2, 3, 1, covariance="ledoit-wolf", n_shuffles=1, n_resamples=1)
    assert result["analytical_correction_approximate"]


@pytest.mark.parametrize("invalid", [torch.ones(6, 6), torch.ones(100, 5), torch.full((100, 6), float("nan"))])
def test_invalid_observations(invalid):
    with pytest.raises(AssertionError):
        debias_eigen_pid(invalid, 2, 3, 1)


def test_singular_covariance_fails():
    data = sample()
    data[:, 1] = data[:, 0]
    with pytest.raises((torch.linalg.LinAlgError, ValueError)):
        debias_eigen_pid(data, 2, 3, 1, n_shuffles=1, n_resamples=1)
