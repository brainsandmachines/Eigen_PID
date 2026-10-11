"""Torch implementation of the closed-form generalized-eigenvalue Gaussian PID."""

from __future__ import annotations

from math import log
from numbers import Integral

import torch

from .linalg import conditional_covariance, conditional_covariance_spd, construct_gamma_star
from .linalg import construct_optimal_covariance, factor_spd, generalized_eigh
from .linalg import logdet_from_cholesky, validate_psd, validate_symmetric
from .linalg import condition_num_from_eigvals
from .tolerances import DTYPE_TOLERANCES


LOG_TWO = log(2)


def eigen_pid(cov: torch.Tensor, dm: int, dx: int, dy: int,
              compute_optimal_covariance: bool = False, add_reg=False,
              *,
              disable_numerical_checks: bool = False) -> dict[str, torch.Tensor]:
    """Compute the bivariate population-Gaussian BROJA PID from a covariance matrix.

    The covariance must be ordered as [M, X, Y], where M is the target and
    X and Y are the two sources. Information quantities are returned in bits.

    Args:
        cov:
            Covariance tensor with shape
            (..., dm + dx + dy, dm + dx + dy). Leading dimensions are
            treated as batch dimensions. The tensor must use torch.float32
            or torch.float64 and may be on CPU or CUDA.
            Array-like inputs are converted to float64 Torch tensors.
        dm:
            Number of target dimensions in M.
        dx:
            Number of dimensions in source X.
        dy:
            Number of dimensions in source Y.
        compute_optimal_covariance:
            If True, also construct the covariance that attains the BROJA
            union information. This is optional because it requires additional
            matrix construction and validation.
        add_reg:
            If True, add 1e-7 times the identity to Cov(X, Y) when calculating
            I(M; X, Y). This option exists to reproduce the regularization used
            in the Venkatesh and Zhao calculations.
        disable_numerical_checks:
            If True, skip symmetry, PSD, conditioning, and output diagnostics.
            Use only for numerically suitable inputs. Input validation and
            Cholesky failure detection remain active. By default, checks run;
            condition numbers use detached symmetric eigenvalues on the input
            device without affecting solver gradients.

    Returns:
        A dictionary of Torch tensors containing:

        - ``mix``: Mutual information I(M; X).
        - ``miy``: Mutual information I(M; Y).
        - ``mixy``: Joint mutual information I(M; X, Y).
        - ``union_info``: BROJA union information.
        - ``uix``: Information about M unique to X.
        - ``uiy``: Information about M unique to Y.
        - ``red``: Redundant information shared by X and Y about M.
        - ``syn``: Synergistic information about M available jointly from X
          and Y.
        - ``gen_ev``: Generalized eigenvalues of Cov(M|Y) relative to
          Cov(M|X).
        - ``log2_gen_ev``: Base-2 logarithms of ``gen_ev``.

        If ``compute_optimal_covariance=True``, the dictionary also contains:

        - ``gamma_star``: Optimal conditional covariance Cov(M|X,Y) attaining
          the BROJA union information.
        - ``optimal_cov``: Complete covariance attaining ``gamma_star`` while
          preserving the M-X and M-Y marginals.
        - ``optimal_cov_xy``: X-Y cross-covariance used in ``optimal_cov``.

        Scalar information quantities have the input batch shape.
        ``gen_ev`` and ``log2_gen_ev`` have a final dimension of size ``dm``.

    Raises:
        ValueError:
            If an input, dimension, option, shape, dtype, or device is invalid.
        torch.linalg.LinAlgError:
            If a required covariance is not positive definite, is too
            ill-conditioned, or an eigendecomposition fails.
        ArithmeticError:
            If the calculated optimizer or PID quantities fail their
            mathematical consistency checks.
    """
    if not isinstance(cov, torch.Tensor):
        cov = torch.tensor(cov, dtype=torch.float64)
    dimensions = (dm, dx, dy)
    if any(isinstance(dimension, bool) or not isinstance(dimension, Integral) for dimension in dimensions):
        raise ValueError("dm, dx, and dy must be integers")
    if dm <= 0 or dx <= 0 or dy <= 0:
        raise ValueError("dm, dx, and dy must be positive")
    if not isinstance(compute_optimal_covariance, bool):
        raise ValueError("compute_optimal_covariance must be a boolean")
    if not isinstance(disable_numerical_checks, bool):
        raise ValueError("disable_numerical_checks must be a boolean")
    if cov.dtype not in DTYPE_TOLERANCES:
        raise ValueError("cov must have dtype torch.float32 or torch.float64")
    total_dimension = dm + dx + dy
    expected_shape = (total_dimension, total_dimension)
    if cov.ndim < 2 or tuple(cov.shape[-2:]) != expected_shape:
        raise ValueError(f"cov must have trailing shape {expected_shape}, got {tuple(cov.shape)}")
    if any(size == 0 for size in cov.shape[:-2]):
        raise ValueError("cov batch dimensions must be nonempty")
    if not torch.isfinite(cov).all().item():
        raise ValueError("cov contains NaN or infinity")
    symmetry_tol, reciprocal_condition_tol, identity_rtol, identity_atol = DTYPE_TOLERANCES[cov.dtype]
    if not disable_numerical_checks:
        validate_symmetric(cov, "cov", symmetry_tol)
        validate_psd(cov, "Sigma(M,X,Y)", reciprocal_condition_tol)

    target, source_x = slice(0, dm), slice(dm, dm + dx)
    source_y, sources_xy = slice(dm + dx, total_dimension), slice(dm, total_dimension)
    sigma_m, sigma_x = cov[..., target, target], cov[..., source_x, source_x]
    sigma_y, sigma_xy = cov[..., source_y, source_y], cov[..., sources_xy, sources_xy]
    sigma_mx, sigma_my = cov[..., target, source_x], cov[..., target, source_y]
    sigma_mxy = cov[..., target, sources_xy]

    #Reg term to match Venkatesh and Zhao calulations
    if add_reg:
        reg = 1e-7
        sigma_xy = sigma_xy + reg * torch.eye(sigma_xy.shape[-1], dtype=sigma_xy.dtype, device=sigma_xy.device)
    sigma_m_given_x = conditional_covariance_spd(
        sigma_m, sigma_mx, sigma_x, "Sigma(X)", symmetry_tol=symmetry_tol,
        reciprocal_condition_tol=reciprocal_condition_tol,
        disable_numerical_checks=disable_numerical_checks)
    sigma_m_given_y = conditional_covariance_spd(
        sigma_m, sigma_my, sigma_y, "Sigma(Y)", symmetry_tol=symmetry_tol,
        reciprocal_condition_tol=reciprocal_condition_tol,
        disable_numerical_checks=disable_numerical_checks)
    sigma_m_given_xy = conditional_covariance(sigma_m, sigma_mxy, sigma_xy,
                                              rtol=reciprocal_condition_tol)
    condition_m = condition_m_given_xy = None
    if not disable_numerical_checks:
        conditional_xy_eigenvalues = torch.linalg.eigvalsh(sigma_m_given_xy.detach())
        target_eigenvalues = torch.linalg.eigvalsh(sigma_m.detach())
        target_scale = torch.amax(torch.abs(target_eigenvalues), dim=-1)
        invalid_conditional = conditional_xy_eigenvalues[..., 0] <= reciprocal_condition_tol * target_scale
        if invalid_conditional.any().item():
            raise torch.linalg.LinAlgError("Sigma(M|X,Y) is not positive definite; "
                                           "this causes infinite mutual information")
        condition_m_given_xy = condition_num_from_eigvals(conditional_xy_eigenvalues)
        condition_m = condition_num_from_eigvals(target_eigenvalues)
    factor_m_given_xy = factor_spd(sigma_m_given_xy, "Sigma(M|X,Y)", symmetry_tol,
                                   reciprocal_condition_tol,
                                   condition_number=condition_m_given_xy,
                                   disable_numerical_checks=disable_numerical_checks)
    factor_m = factor_spd(sigma_m, "Sigma(M)", symmetry_tol, reciprocal_condition_tol,
                         condition_number=condition_m,
                         disable_numerical_checks=disable_numerical_checks)
    factor_m_given_x = factor_spd(sigma_m_given_x, "Sigma(M|X)", symmetry_tol,
                                  reciprocal_condition_tol,
                                  disable_numerical_checks=disable_numerical_checks)
    factor_m_given_y = factor_spd(sigma_m_given_y, "Sigma(M|Y)", symmetry_tol,
                                  reciprocal_condition_tol,
                                  disable_numerical_checks=disable_numerical_checks)

    logdet_m = logdet_from_cholesky(factor_m)
    logdet_m_given_x = logdet_from_cholesky(factor_m_given_x)
    logdet_m_given_y = logdet_from_cholesky(factor_m_given_y)
    logdet_m_given_xy = logdet_from_cholesky(factor_m_given_xy)
    mutual_information_x = (logdet_m - logdet_m_given_x) / (2 * LOG_TWO)
    mutual_information_y = (logdet_m - logdet_m_given_y) / (2 * LOG_TWO)
    mutual_information_xy = (logdet_m - logdet_m_given_xy) / (2 * LOG_TWO)

    eigensystem = generalized_eigh(sigma_m_given_y, factor_m_given_x,
                                   eigenvectors=compute_optimal_covariance)
    if compute_optimal_covariance:
        generalized_eigenvalues, generalized_eigenvectors = eigensystem
    else:
        generalized_eigenvalues = eigensystem
    if not disable_numerical_checks:
        invalid_eigenvalues = ~torch.isfinite(generalized_eigenvalues) | (generalized_eigenvalues <= 0)
        if invalid_eigenvalues.any().item():
            raise torch.linalg.LinAlgError("generalized eigenvalues must be finite and positive")
    log2_generalized_eigenvalues = torch.log2(generalized_eigenvalues)
    unique_information_x = 0.5 * torch.relu(log2_generalized_eigenvalues).sum(dim=-1)
    unique_information_y = 0.5 * torch.relu(-log2_generalized_eigenvalues).sum(dim=-1)
    redundancy_from_x = mutual_information_x - unique_information_x
    if not disable_numerical_checks:
        redundancy_from_y = mutual_information_y - unique_information_y
        if not torch.allclose(redundancy_from_x, redundancy_from_y,
                              rtol=identity_rtol, atol=identity_atol):
            raise ArithmeticError("directional redundancies disagree")
    redundancy = redundancy_from_x
    union_information = unique_information_x + unique_information_y + redundancy
    synergy = mutual_information_xy - union_information
    information = {"mix": mutual_information_x, "miy": mutual_information_y,
                   "mixy": mutual_information_xy, "union_info": union_information,
                   "uix": unique_information_x, "uiy": unique_information_y,
                   "red": redundancy, "syn": synergy}
    if not disable_numerical_checks:
        information_values = torch.stack(tuple(information.values()), dim=-1)
        if not torch.isfinite(information_values).all().item():
            raise ArithmeticError("an information quantity is not finite")
        if (information_values < -identity_atol).any().item():
            raise ArithmeticError("an information quantity is materially negative")
    result = {**information, "gen_ev": generalized_eigenvalues,
              "log2_gen_ev": log2_generalized_eigenvalues}
    if compute_optimal_covariance:
        gamma_star = construct_gamma_star(sigma_m_given_x, sigma_m_given_y,
                                          generalized_eigenvalues, generalized_eigenvectors,
                                          symmetry_tol=symmetry_tol,
                                          reciprocal_condition_tol=reciprocal_condition_tol,
                                          disable_numerical_checks=disable_numerical_checks)
        sigma_xy_star, optimal_cov = construct_optimal_covariance(cov, gamma_star, dm, dx, dy,
                                                                    symmetry_tol=symmetry_tol,
            reciprocal_condition_tol=reciprocal_condition_tol,
            identity_rtol=identity_rtol, identity_atol=identity_atol,
            disable_numerical_checks=disable_numerical_checks)
        result.update(gamma_star=gamma_star, optimal_cov=optimal_cov,
                      optimal_cov_xy=sigma_xy_star)
    return result
