"""Torch linear algebra for the Gaussian Eigen-PID calculation."""

from __future__ import annotations

import torch


def condition_num_from_eigvals(eigenvalues: torch.Tensor) -> torch.Tensor:
    """Return the 2-norm condition number from symmetric/Hermitian eigenvalues.

    Args:
        eigenvalues: Tensor of shape (..., n); ordering does not matter.
    Returns:
        Tensor of shape (...). A zero eigenvalue gives infinity; all-zero
        eigenvalues give NaN, as with torch.linalg.cond.
    """
    singular_values = eigenvalues.abs()
    return singular_values.amax(dim=-1) / singular_values.amin(dim=-1)


def condition_sym_mat(matrix: torch.Tensor) -> torch.Tensor:
    """Return the 2-norm condition number of symmetric/Hermitian matrices.

    Args:
        matrix: Tensor of shape (..., n, n). Symmetry is assumed; only the
            lower triangle is used.
    Returns:
        Tensor of shape (...) on the input device. A zero eigenvalue gives
        infinity; an all-zero matrix gives NaN, as with torch.linalg.cond.
    """
    return condition_num_from_eigvals(torch.linalg.eigvalsh(matrix))


def _matrix_scale(matrix: torch.Tensor) -> torch.Tensor:
    return torch.amax(torch.abs(matrix), dim=(-2, -1))


def validate_symmetric(matrix: torch.Tensor, name: str, symmetry_tol: float) -> None:
    """Report material asymmetry without modifying the matrix."""
    scale = _matrix_scale(matrix)
    absolute_error = torch.amax(torch.abs(matrix - matrix.mT), dim=(-2, -1))
    denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
    error = absolute_error / denominator
    if (error > symmetry_tol).any().item():
        raise ValueError(f"{name} is asymmetric: relative max error {error.max().item():.3e} "
                         f"exceeds {symmetry_tol:.3e}")


def validate_psd(matrix: torch.Tensor, name: str, eigenvalue_tol: float) -> None:
    """Report material indefiniteness without modifying the matrix."""
    try:
        minimum = torch.linalg.eigvalsh(matrix)[..., 0]
    except RuntimeError as exc:
        raise torch.linalg.LinAlgError(f"could not calculate eigenvalues of {name}") from exc
    if (minimum < -eigenvalue_tol * _matrix_scale(matrix)).any().item():
        raise torch.linalg.LinAlgError(f"{name} is not positive semidefinite")


def _pinvh(matrix: torch.Tensor, rtol: float) -> torch.Tensor:
    try:
        return torch.linalg.pinv(matrix, rtol=rtol, hermitian=True)
    except RuntimeError as exc:
        raise torch.linalg.LinAlgError("the symmetric pseudoinverse failed") from exc


def conditional_covariance(sigma_m: torch.Tensor, sigma_m_source: torch.Tensor,
                           sigma_source: torch.Tensor, *, rtol: float) -> torch.Tensor:
    """Calculate batched Sigma(M|S) using a symmetric pseudoinverse."""
    return sigma_m - sigma_m_source @ _pinvh(sigma_source, rtol) @ sigma_m_source.mT


def factor_spd(matrix: torch.Tensor, name: str, symmetry_tol: float,
               reciprocal_condition_tol: float, *,
               condition_number: torch.Tensor | None = None,
               disable_numerical_checks: bool = False) -> torch.Tensor:
    """Validate and Cholesky-factor SPD matrices, reusing condition_number if given."""
    if not disable_numerical_checks:
        validate_symmetric(matrix, name, symmetry_tol)
    try:
        factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
    except RuntimeError as exc:
        raise torch.linalg.LinAlgError(f"{name} is not positive definite") from exc
    if (info != 0).any().item():
        raise torch.linalg.LinAlgError(f"{name} is not positive definite")
    if disable_numerical_checks:
        return factor
    try:
        condition = (condition_sym_mat(matrix.detach()) if condition_number is None
                     else condition_number)
    except RuntimeError as exc:
        raise torch.linalg.LinAlgError(f"could not estimate the condition of {name}") from exc
    reciprocal = torch.where(torch.isfinite(condition), condition.reciprocal(), torch.zeros_like(condition))
    if (reciprocal < reciprocal_condition_tol).any().item():
        raise torch.linalg.LinAlgError(f"{name} is ill-conditioned: estimated condition={condition.max().item():.3e}, "
                                       f"required reciprocal condition >= {reciprocal_condition_tol:.3e}")
    return factor


def conditional_covariance_spd(
    sigma_m: torch.Tensor,
    sigma_m_source: torch.Tensor,
    sigma_source: torch.Tensor,
    source_name: str,
    *,
    symmetry_tol: float,
    reciprocal_condition_tol: float,
    disable_numerical_checks: bool = False,
) -> torch.Tensor:
    """Calculate batched Sigma(M|S) when Sigma(S) is required to be SPD."""
    factor = factor_spd(sigma_source, source_name, symmetry_tol,
                        reciprocal_condition_tol,
                        disable_numerical_checks=disable_numerical_checks)
    try:
        solved = torch.cholesky_solve(sigma_m_source.mT, factor)
    except RuntimeError as exc:
        raise torch.linalg.LinAlgError(f"the SPD solve for {source_name} failed") from exc
    return sigma_m - sigma_m_source @ solved


def logdet_from_cholesky(factor: torch.Tensor) -> torch.Tensor:
    """Calculate log-determinants from lower Cholesky factors."""
    return 2 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(dim=-1)


def generalized_eigh(
    matrix_a: torch.Tensor,
    matrix_b_factor: torch.Tensor,
    *,
    eigenvectors: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Solve A v = lambda B v from a lower Cholesky factor of B."""
    try:
        whitened = torch.linalg.solve_triangular(matrix_b_factor, matrix_a, upper=False)
        whitened = torch.linalg.solve_triangular(matrix_b_factor, whitened.mT, upper=False).mT
        if not eigenvectors:
            return torch.linalg.eigvalsh(whitened)
        eigenvalues, vectors = torch.linalg.eigh(whitened)
        vectors = torch.linalg.solve_triangular(matrix_b_factor.mT, vectors, upper=True)
        return eigenvalues, vectors
    except RuntimeError as exc:
        raise torch.linalg.LinAlgError("the generalized symmetric-definite eigenproblem failed") from exc


def construct_gamma_star(sigma_m_given_x: torch.Tensor, sigma_m_given_y: torch.Tensor,
                         eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, *,
                         symmetry_tol: float, reciprocal_condition_tol: float,
                         disable_numerical_checks: bool = False) -> torch.Tensor:
    """Construct and validate the batched closed-form covariance optimizer."""
    metric_vectors = sigma_m_given_x @ eigenvectors
    gamma_star = (metric_vectors * eigenvalues.clamp_max(1).unsqueeze(-2)) @ metric_vectors.mT
    if disable_numerical_checks:
        return gamma_star
    if not torch.isfinite(gamma_star).all().item():
        raise ArithmeticError("Gamma_star contains NaN or infinity")
    validate_symmetric(gamma_star, "Gamma_star", symmetry_tol)
    bounds = ((sigma_m_given_x - gamma_star, "Sigma(M|X)", sigma_m_given_x),
              (sigma_m_given_y - gamma_star, "Sigma(M|Y)", sigma_m_given_y))
    for difference, bound_name, bound in bounds:
        scale = torch.maximum(_matrix_scale(bound), _matrix_scale(gamma_star))
        minimum = torch.linalg.eigvalsh(difference)[..., 0]
        if (minimum < -reciprocal_condition_tol * scale).any().item():
            raise ArithmeticError(f"Gamma_star is not Loewner-bounded by {bound_name}")
    return gamma_star


def construct_optimal_covariance(
    cov: torch.Tensor,
    gamma_star: torch.Tensor,
    dm: int,
    dx: int,
    dy: int,
    *,
    symmetry_tol: float,
    reciprocal_condition_tol: float,
    identity_rtol: float,
    identity_atol: float,
    disable_numerical_checks: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct and validate batched covariances attaining Gamma_star."""
    target, source_x = slice(0, dm), slice(dm, dm + dx)
    source_y, sources_xy = slice(dm + dx, dm + dx + dy), slice(dm, dm + dx + dy)
    sigma_m, sigma_mx = cov[..., target, target], cov[..., target, source_x]
    sigma_my = cov[..., target, source_y]
    c_star = sigma_m - gamma_star
    if not disable_numerical_checks:
        validate_symmetric(c_star, "C_star", symmetry_tol)
        validate_psd(c_star, "C_star", reciprocal_condition_tol)
    c_star_pinv = _pinvh(c_star, reciprocal_condition_tol)
    if not disable_numerical_checks:
        for cross_covariance, source_name in ((sigma_mx.mT, "X"), (sigma_my.mT, "Y")):
            reconstructed = cross_covariance @ c_star_pinv @ c_star
            if not torch.allclose(reconstructed, cross_covariance, rtol=identity_rtol, atol=identity_atol):
                raise ArithmeticError(f"{source_name} range condition failed")
    sigma_xy_star = sigma_mx.mT @ c_star_pinv @ sigma_my
    optimal_cov = cov.clone()
    optimal_cov[..., source_x, source_y] = sigma_xy_star
    optimal_cov[..., source_y, source_x] = sigma_xy_star.mT
    if disable_numerical_checks:
        return sigma_xy_star, optimal_cov
    if not torch.isfinite(optimal_cov).all().item():
        raise ValueError("optimal_cov contains NaN or infinity")
    validate_symmetric(optimal_cov, "optimal_cov", symmetry_tol)
    validate_psd(optimal_cov, "optimal_cov", reciprocal_condition_tol)
    reconstructed = conditional_covariance(sigma_m, optimal_cov[..., target, sources_xy],
                                            optimal_cov[..., sources_xy, sources_xy],
                                            rtol=reciprocal_condition_tol)
    if not torch.allclose(reconstructed, gamma_star, rtol=identity_rtol, atol=identity_atol):
        raise ArithmeticError("constructed optimal covariance does not reproduce Gamma_star")
    return sigma_xy_star, optimal_cov
