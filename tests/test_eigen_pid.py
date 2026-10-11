from __future__ import annotations

import pytest
import torch

from eigen_pid import conditional_covariance, eigen_pid
from eigen_pid.linalg import condition_num_from_eigvals, condition_sym_mat
from eigen_pid.linalg import conditional_covariance_spd, factor_spd, generalized_eigh
from eigen_pid.linalg import validate_psd, validate_symmetric
from eigen_pid.tolerances import DTYPE_TOLERANCES


COVARIANCE = torch.tensor([[1.0, 0.6, 0.4],
                           [0.6, 1.0, 0.2],
                           [0.4, 0.2, 1.0]], dtype=torch.float64)
REDUNDANT_SOURCES = torch.tensor([[2.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0],
                                  [1.0, 1.0, 1.0]], dtype=torch.float64)
TWO_DIMENSIONAL = torch.tensor([[1.0, 0.0, 0.6, 0.0, 0.4, 0.0],
                                [0.0, 1.0, 0.0, 0.4, 0.0, 0.6],
                                [0.6, 0.0, 1.0, 0.0, 0.2, 0.0],
                                [0.0, 0.4, 0.0, 1.0, 0.0, 0.2],
                                [0.4, 0.0, 0.2, 0.0, 1.0, 0.0],
                                [0.0, 0.6, 0.0, 0.2, 0.0, 1.0]], dtype=torch.float64)
INFORMATION_KEYS = ("mix", "miy", "mixy", "union_info", "uix", "uiy", "red", "syn")
CUDA = pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                     reason="CUDA is unavailable"))


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
def test_condition_sym_mat_matches_cond(device, dtype):
    matrices = torch.stack([
        COVARIANCE, -COVARIANCE,
        torch.diag(torch.tensor([-2.0, -1.0, 0.5], dtype=torch.float64)),
        torch.diag(torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)),
        torch.zeros_like(COVARIANCE),
    ]).reshape(1, 5, 3, 3).to(device=device, dtype=dtype)
    if dtype.is_complex:
        phase = torch.tensor([1, 1j, -1j], device=device, dtype=dtype)
        matrices = matrices * phase[:, None] * phase.conj()[None, :]
    rtol = 1e-4 if dtype in (torch.float32, torch.complex64) else 1e-10
    expected = torch.linalg.cond(matrices)
    # Reverse the spectrum to also exercise unsorted, precomputed eigenvalues.
    eigenvalues = torch.linalg.eigvalsh(matrices).flip(-1)
    for actual in (condition_sym_mat(matrices), condition_num_from_eigvals(eigenvalues)):
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_scalar_regression(dtype: torch.dtype) -> None:
    result = eigen_pid(COVARIANCE.to(dtype), 1, 1, 1)
    expected = torch.tensor([0.3219280948873623, 0.12576938349798222,
                             0.42040070257537293, 0.3219280948873623,
                             0.19615871138938004, 0.0,
                             0.12576938349798222, 0.09847260768801064], dtype=dtype)
    _, _, rtol, atol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(torch.stack([result[key] for key in INFORMATION_KEYS]),
                               expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(result["gen_ev"], torch.tensor([1.3125], dtype=dtype),
                               rtol=rtol, atol=atol)


def test_fixed_independent_and_redundant_regressions() -> None:
    independent = eigen_pid(torch.eye(3, dtype=torch.float64), 1, 1, 1)
    torch.testing.assert_close(torch.stack([independent[key] for key in INFORMATION_KEYS]),
                               torch.zeros(8, dtype=torch.float64), rtol=0, atol=1e-14)
    redundant = eigen_pid(REDUNDANT_SOURCES, 1, 1, 1)
    expected = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0],
                            dtype=torch.float64)
    torch.testing.assert_close(torch.stack([redundant[key] for key in INFORMATION_KEYS]),
                               expected, rtol=1e-12, atol=1e-14)


def test_unique_information_uses_zero_subgradient_at_equal_strength() -> None:
    covariance = torch.tensor([[1.25, 0.5, 0.5],
                               [0.5, 1.0, 0.0],
                               [0.5, 0.0, 1.0]], dtype=torch.float64, requires_grad=True)
    result = eigen_pid(covariance, 1, 1, 1)

    torch.testing.assert_close(result["gen_ev"], torch.ones(1, dtype=torch.float64))
    for key in ("uix", "uiy"):
        gradient, = torch.autograd.grad(result[key], covariance, retain_graph=True)
        torch.testing.assert_close(gradient, torch.zeros_like(covariance), rtol=0, atol=0)


def test_fixed_multidimensional_regression() -> None:
    result = eigen_pid(TWO_DIMENSIONAL, 2, 2, 2)
    expected = torch.tensor([0.4476974783853445, 0.4476974783853445,
                             0.8408014051507459, 0.6438561897747246,
                             0.19615871138938004, 0.19615871138938004,
                             0.25153876699596444, 0.19694521537602128],
                            dtype=torch.float64)
    torch.testing.assert_close(torch.stack([result[key] for key in INFORMATION_KEYS]),
                               expected, rtol=1e-12, atol=1e-14)
    expected_eigenvalues = torch.tensor([0.7619047619047619, 1.3125], dtype=torch.float64)
    torch.testing.assert_close(result["gen_ev"], expected_eigenvalues,
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_generalized_eigenvalue_identities(dtype: torch.dtype, device: str) -> None:
    matrix_a = torch.tensor([[1.4, 0.2], [0.2, 0.9]], dtype=dtype, device=device)
    matrix_b = torch.tensor([[2.0, 0.3], [0.3, 1.2]], dtype=dtype, device=device)
    symmetry_tol, condition_tol, rtol, atol = DTYPE_TOLERANCES[dtype]
    factor = factor_spd(matrix_b, "B", symmetry_tol, condition_tol)
    eigenvalues, vectors = generalized_eigh(matrix_a, factor, eigenvectors=True)
    torch.testing.assert_close(matrix_a @ vectors,
                               (matrix_b @ vectors) * eigenvalues.unsqueeze(-2),
                               rtol=rtol, atol=atol)
    torch.testing.assert_close(vectors.mT @ matrix_b @ vectors,
                               torch.eye(2, dtype=dtype, device=device),
                               rtol=rtol, atol=atol)


def test_conditional_covariance_spd_supports_arbitrary_batch_dimensions() -> None:
    sigma_source = torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    sigma_source = sigma_source.expand(2, 3, -1, -1)
    sigma_m_source = torch.arange(12, dtype=torch.float64).reshape(2, 3, 1, 2) / 20
    sigma_m = torch.full((2, 3, 1, 1), 2.0, dtype=torch.float64)
    symmetry_tol, condition_tol, _, _ = DTYPE_TOLERANCES[torch.float64]

    result = conditional_covariance_spd(
        sigma_m, sigma_m_source, sigma_source, "Sigma(S)",
        symmetry_tol=symmetry_tol, reciprocal_condition_tol=condition_tol)
    expected = sigma_m - sigma_m_source @ torch.linalg.solve(
        sigma_source, sigma_m_source.mT)

    assert result.shape == (2, 3, 1, 1)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_cpu_gpu_parity(dtype: torch.dtype) -> None:
    cpu_result = eigen_pid(TWO_DIMENSIONAL.to(dtype), 2, 2, 2,
                           compute_optimal_covariance=True)
    gpu_result = eigen_pid(TWO_DIMENSIONAL.to(dtype).cuda(), 2, 2, 2,
                           compute_optimal_covariance=True)
    _, _, rtol, atol = DTYPE_TOLERANCES[dtype]
    assert cpu_result.keys() == gpu_result.keys()
    for key in cpu_result:
        torch.testing.assert_close(cpu_result[key], gpu_result[key].cpu(),
                                   rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_batch_matches_individual_calls(dtype: torch.dtype, device: str) -> None:
    row = torch.stack((COVARIANCE, torch.eye(3), REDUNDANT_SOURCES))
    batch = torch.stack((row, row * 0.25)).to(dtype=dtype, device=device)
    batched = eigen_pid(batch, 1, 1, 1, compute_optimal_covariance=True)
    individual = [eigen_pid(item, 1, 1, 1, compute_optimal_covariance=True)
                  for item in batch.reshape(-1, 3, 3)]
    _, _, rtol, atol = DTYPE_TOLERANCES[dtype]
    for key, values in batched.items():
        expected = torch.stack([result[key] for result in individual])
        expected = expected.reshape(*batch.shape[:-2], *expected.shape[1:])
        torch.testing.assert_close(values, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_optimal_covariance_is_reported_valid_without_repair(dtype: torch.dtype,
                                                             device: str) -> None:
    covariance = COVARIANCE.to(dtype=dtype, device=device)
    original = covariance.clone()
    result = eigen_pid(covariance, 1, 1, 1, compute_optimal_covariance=True)
    optimal = result["optimal_cov"]
    symmetry_tol, condition_tol, rtol, atol = DTYPE_TOLERANCES[dtype]
    scale = torch.abs(optimal).max()
    relative_asymmetry = torch.abs(optimal - optimal.mT).max() / scale
    minimum_eigenvalue = torch.linalg.eigvalsh(optimal)[0]
    assert relative_asymmetry <= symmetry_tol, f"reported asymmetry: {relative_asymmetry.item():.3e}"
    assert minimum_eigenvalue >= -condition_tol * scale, \
        f"reported minimum eigenvalue: {minimum_eigenvalue.item():.3e}"
    torch.testing.assert_close(covariance, original, rtol=0, atol=0)
    torch.testing.assert_close(optimal[:2, :2], covariance[:2, :2], rtol=0, atol=0)
    torch.testing.assert_close(optimal[(0, 2), :][:, (0, 2)],
                               covariance[(0, 2), :][:, (0, 2)], rtol=0, atol=0)
    reconstructed = conditional_covariance(optimal[:1, :1], optimal[:1, 1:],
                                           optimal[1:, 1:], rtol=condition_tol)
    torch.testing.assert_close(reconstructed, result["gamma_star"], rtol=rtol, atol=atol)
    attained_union = 0.5 * torch.log2(optimal[0, 0] / reconstructed[0, 0])
    torch.testing.assert_close(attained_union, result["union_info"], rtol=rtol, atol=atol)


def test_optimal_covariance_accepts_roundoff_in_zero_loewner_residual() -> None:
    generator = torch.Generator().manual_seed(4)
    basis = torch.randn(6, 6, generator=generator, dtype=torch.float64)
    covariance = basis @ basis.mT + 0.5 * torch.eye(6, dtype=torch.float64)

    result = eigen_pid(covariance, 2, 2, 2, compute_optimal_covariance=True)

    assert (result["gen_ev"] > 1).all()
    assert torch.isfinite(result["optimal_cov"]).all()


@pytest.mark.parametrize(("dimensions", "message"),
                         [((0, 1, 1), "positive"), ((1, -1, 1), "positive"),
                          ((1.0, 1, 1), "integers"), ((True, 1, 1), "integers")])
def test_invalid_dimensions(dimensions: tuple[object, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        eigen_pid(COVARIANCE, *dimensions)  # type: ignore[arg-type]


def test_invalid_input_type_dtype_shape_and_option() -> None:
    converted = eigen_pid(COVARIANCE.tolist(), 1, 1, 1)  # type: ignore[arg-type]
    assert list(converted) == [
        "mix", "miy", "mixy", "union_info", "uix", "uiy", "red", "syn",
        "gen_ev", "log2_gen_ev",
    ]
    for dtype in (torch.int64, torch.complex128):
        with pytest.raises(ValueError, match="dtype"):
            eigen_pid(COVARIANCE.to(dtype), 1, 1, 1)
    with pytest.raises(ValueError, match="trailing shape"):
        eigen_pid(torch.eye(4), 1, 1, 1)
    with pytest.raises(ValueError, match="boolean"):
        eigen_pid(COVARIANCE, 1, 1, 1, compute_optimal_covariance=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty"):
        eigen_pid(torch.empty((0, 3, 3)), 1, 1, 1)


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_input(invalid: float) -> None:
    covariance = COVARIANCE.clone()
    covariance[0, 0] = invalid
    with pytest.raises(ValueError, match="NaN or infinity"):
        eigen_pid(covariance, 1, 1, 1)


def test_asymmetric_indefinite_and_deterministic_inputs_are_reported() -> None:
    asymmetric = COVARIANCE.clone()
    asymmetric[0, 1] += 1e-3
    with pytest.raises(ValueError, match="asymmetric"):
        eigen_pid(asymmetric, 1, 1, 1)
    indefinite = torch.tensor([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 2.0],
                               [0.0, 2.0, 1.0]], dtype=torch.float64)
    with pytest.raises(torch.linalg.LinAlgError, match="positive semidefinite"):
        eigen_pid(indefinite, 1, 1, 1)
    deterministic = torch.tensor([[1.0, 1.0, 0.0],
                                  [1.0, 1.0, 0.0],
                                  [0.0, 0.0, 1.0]], dtype=torch.float64)
    with pytest.raises(torch.linalg.LinAlgError, match=r"Sigma\(M\|X,Y\)"):
        eigen_pid(deterministic, 1, 1, 1)


def test_validation_is_relative_to_each_matrix_scale_and_zero_is_psd() -> None:
    unit = torch.eye(2, dtype=torch.float64)
    small_asymmetric = torch.tensor([[1e-10, 1e-13], [0.0, 1e-10]],
                                    dtype=torch.float64)
    with pytest.raises(ValueError, match="asymmetric"):
        validate_symmetric(torch.stack((unit, small_asymmetric)), "batch", 1e-10)

    small_indefinite = torch.diag(torch.tensor([1e-10, -1e-13], dtype=torch.float64))
    with pytest.raises(torch.linalg.LinAlgError, match="positive semidefinite"):
        validate_psd(torch.stack((unit, small_indefinite)), "batch", 1e-12)

    validate_symmetric(torch.zeros((2, 3, 3), dtype=torch.float64), "zero", 1e-10)
    validate_psd(torch.zeros((2, 3, 3), dtype=torch.float64), "zero", 1e-12)


def test_singular_individual_source_is_rejected() -> None:
    covariance = torch.diag(torch.tensor([1.0, 1.0, 0.0, 1.0],
                                         dtype=torch.float64))

    with pytest.raises(torch.linalg.LinAlgError, match=r"Sigma\(X\) is not positive definite"):
        eigen_pid(covariance, 1, 2, 1)


def test_ill_conditioned_spd_source_is_rejected_instead_of_truncated() -> None:
    epsilon = 1e-13
    covariance = torch.tensor(
        [[1.0, 0.0, 0.5 * epsilon ** 0.5, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [0.5 * epsilon ** 0.5, 0.0, epsilon, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)

    with pytest.raises(torch.linalg.LinAlgError, match=r"Sigma\(X\) is ill-conditioned"):
        eigen_pid(covariance, 1, 2, 1)


def test_singular_joint_sources_are_accepted_and_input_is_not_repaired() -> None:
    covariance = REDUNDANT_SOURCES.clone()
    original = covariance.clone()
    result = eigen_pid(covariance, 1, 1, 1)
    assert all(torch.isfinite(result[key]).all() for key in INFORMATION_KEYS)
    torch.testing.assert_close(covariance, original, rtol=0, atol=0)
