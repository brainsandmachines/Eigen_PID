"""Reimplementation of Lorenz et al.'s debiasing procedure using Eigen-PID.

The finite-sample correction procedure is from Lorenz et al. (2026):
https://doi.org/10.1016/j.patter.2026.101619.
This module reimplements that procedure, using Eigen-PID to evaluate the
original covariance and every surrogate covariance. Eigen-PID is described in
"Gaussian BROJA Partial Information Decomposition in Closed Form" (under review).

The analytical MI correction is exact for iid Gaussian observations and the
ordinary sample covariance. Union-information bias is estimated by target-row
permutations and Gaussian parametric bootstraps. Negative estimates are retained.
"""

from math import isfinite, log
from numbers import Integral
from time import perf_counter

import torch

from .eigen_pid import eigen_pid


def estimate_covariance(data, method="empirical"):
    """Fit the same estimator to the observations and every surrogate."""
    assert method in {"empirical", "ledoit-wolf"}, method
    if method == "empirical":
        return torch.cov(data.T)
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as error:
        raise ImportError("Ledoit–Wolf requires scikit-learn: pip install scikit-learn") from error
    fitted = LedoitWolf(store_precision=False).fit(data.cpu().numpy()).covariance_
    return torch.from_numpy(fitted).to(data.device)


def components(mix, miy, mixy, union_info, *, disable_numerical_checks=False):
    """Derive the four atoms in bits, without clipping negative estimates."""
    values = dict(mix=mix, miy=miy, mixy=mixy, union_info=union_info,
                  uix=union_info - miy, uiy=union_info - mix,
                  red=mix + miy - union_info, syn=mixy - union_info)
    if not disable_numerical_checks:
        assert all(isfinite(value) for value in values.values())
    return values


def pid_from_covariance(covariance, dm, dx, dy, *, disable_numerical_checks=False):
    """Return scalar information quantities, omitting optimizer diagnostics."""
    result = eigen_pid(covariance, dm, dx, dy,
                       disable_numerical_checks=disable_numerical_checks)
    return components(*(result[key].item() for key in ("mix", "miy", "mixy", "union_info")),
                      disable_numerical_checks=disable_numerical_checks)


def entropy_bias(dimension, n):
    """Gaussian sample-entropy bias in bits, from the Wishart log determinant."""
    assert 0 < dimension < n
    k = torch.arange(1, dimension + 1, dtype=torch.float64)
    return (dimension * log(2 / (n - 1)) + torch.special.digamma((n - k) / 2).sum().item()) / (2 * log(2))


@torch.no_grad()
def debias_eigen_pid(data, dm, dx, dy, *, covariance="empirical", n_shuffles=20,
                     n_resamples=20, seed=None, progress=False,
                     disable_numerical_checks=False):
    """Estimate Gaussian PID from observations, with finite-sample corrections.

    Returns the uncorrected estimate and two corrected estimates of the
    information that sources X and Y carry about target M. All information
    quantities are in bits.

    Args:
        data:
            Torch tensor or array-like object of shape (n, dm + dx + dy).
            Rows are observations; columns contain M, then X, then Y.
            Values must be finite, n must exceed dm + dx + dy, and the fitted
            covariance must be positive definite.
        dm:
            Number of target M columns. Must be a positive integer.
        dx:
            Number of source X columns. Must be a positive integer.
        dy:
            Number of source Y columns. Must be a positive integer.
        covariance:
            Covariance estimator: "empirical" (default) for ordinary sample
            covariance, or "ledoit-wolf" for shrinkage, which requires
            scikit-learn. The same estimator is used for every surrogate.
        n_shuffles:
            Number of target-row permutations used to estimate union-information
            bias. X and Y remain paired. Positive integer; default 20.
        n_resamples:
            Number of Gaussian bootstrap samples drawn from the fitted
            covariance to estimate union-information bias. Each sample has
            n rows. Positive integer; default 20.
        seed:
            Integer seed for a local Torch random generator, or None (default)
            to use and advance the caller's RNG without reseeding it. An integer
            seed leaves the caller's RNG unchanged. CPU and CUDA draws may
            differ even with the same seed.
        progress:
            If True, display progress bars for permutations and bootstraps.
            Requires tqdm. Default False.
        disable_numerical_checks:
            If True, skip symmetry, PSD, conditioning, and output diagnostics
            for the original covariance and every surrogate. Use only for
            numerically suitable inputs. Input validation and Cholesky failure
            detection remain active. Default False keeps the checks enabled.

    Returns:
        A dictionary with the following entries. Numerical estimates are Python
        floats, not Torch tensors.

        raw:
            Uncorrected PID from the fitted covariance.
        permutation:
            PID with analytical mutual-information bias removed and union
            information corrected using the mean shuffled-target estimate.
        combined:
            PID with the same analytical mutual-information correction, but
            union information corrected using the average of permutation and
            bootstrap bias estimates. Bootstrap bias is the mean bootstrap
            union information minus the raw union information.

            Each of raw, permutation, and combined is a dictionary containing:
            ``mix`` = I(M; X), ``miy`` = I(M; Y), ``mixy`` = I(M; X, Y),
            ``union_info`` = BROJA union information, ``uix`` and ``uiy`` =
            information unique to X and Y, ``red`` = redundancy, and
            ``syn`` = synergy. All values are in bits.
        mi_bias:
            Dictionary with keys ``mix``, ``miy``, and ``mixy``: analytical
            biases in bits subtracted from raw MI in both corrected estimates.
        union_bias:
            Dictionary with keys ``permutation`` and ``combined``: estimated
            biases in bits subtracted from raw union information.
        surrogate_unions:
            Dictionary containing ``shuffle`` and ``bootstrap`` lists of union
            information estimates in bits, of lengths n_shuffles and
            n_resamples, respectively.
        timings:
            Wall-clock seconds by stage: ``covariance`` (fit the original
            covariance), ``raw_pid`` (original PID), ``bootstrap_factor``
            (factor the covariance for sampling), ``shuffle`` and ``bootstrap``
            (generate and evaluate each set of surrogates), and ``total``
            (the complete function call).
        iteration_seconds:
            Dictionary containing ``shuffle`` and ``bootstrap`` lists of
            per-surrogate wall-clock seconds, in the same order as
            surrogate_unions. Each includes sample generation, covariance
            fitting, and PID calculation.
        covariance:
            Name of the covariance estimator used.
        analytical_correction_approximate:
            Boolean: True for Ledoit–Wolf shrinkage, False for empirical
            covariance.

    Notes:
        The analytical correction assumes independent, identically distributed
        Gaussian observations and ordinary sample covariance. Using it with
        Ledoit–Wolf shrinkage is approximate. Negative corrected estimates are
        retained rather than clipped.

        Computation uses float64 and preserves the input tensor's device;
        array-like inputs normally use CPU. Ledoit–Wolf fits on CPU before
        transferring the covariance back. Condition checks use detached
        eigenvalues on the input device. GPU timing boundaries synchronize that
        device. This function does not support autograd.

    Example:
        # data contains target (2 columns), X (3 columns), and Y (4 columns).
        result = debias_eigen_pid(data, dm=2, dx=3, dy=4, seed=0)
        synergy_bits = result["combined"]["syn"]
    """
    start = perf_counter()
    data = torch.as_tensor(data, dtype=torch.float64)

    def clock():
        if data.is_cuda:
            torch.cuda.synchronize(data.device)
        return perf_counter()
    for value in (dm, dx, dy, n_shuffles, n_resamples):
        assert isinstance(value, Integral) and not isinstance(value, bool) and value > 0
    assert data.ndim == 2 and data.shape[1] == dm + dx + dy
    n = data.shape[0]
    assert n > data.shape[1], "Analytical correction requires n > dm + dx + dy"
    assert torch.isfinite(data).all(), "Observations contain NaN or infinity"
    iterator = range
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as error:
            raise ImportError("Progress reporting requires tqdm: pip install tqdm") from error
        iterator = lambda count: tqdm(range(count), desc="PID surrogates")

    tick = clock()
    fitted = estimate_covariance(data, covariance)
    times = {"covariance": clock() - tick}
    tick = clock()
    raw = pid_from_covariance(fitted, dm, dx, dy,
                              disable_numerical_checks=disable_numerical_checks)
    times["raw_pid"] = clock() - tick
    tick = clock()
    factor = torch.linalg.cholesky(fitted) # used later to generate Gaussian bootstraps
    times["bootstrap_factor"] = clock() - tick
    rng = None if seed is None else torch.Generator(device=data.device).manual_seed(seed)
    unions, iteration_times = {}, {}

    # Keep X and Y paired when destroying target dependence. Bootstrap from the
    # fitted joint distribution, whose population union is the raw estimate.
    for kind, count in (("shuffle", n_shuffles), ("bootstrap", n_resamples)):
        tick = clock()
        unions[kind], iteration_times[kind] = [], []
        for _ in iterator(count):
            iteration_start = clock()
            if kind == "shuffle":  # randomly permute the target rows, keeping X and Y paired
                surrogate = data.clone()
                surrogate[:, :dm] = data[torch.randperm(n, device=data.device, generator=rng), :dm]
            else: # draw a Gaussian bootstrap sample from the fitted covariance, using the Cholesky factor
                surrogate = torch.randn(data.shape, dtype=data.dtype, device=data.device, generator=rng) @ factor.T
            estimate = pid_from_covariance(estimate_covariance(surrogate, covariance), dm, dx, dy,
                                           disable_numerical_checks=disable_numerical_checks)
            unions[kind].append(estimate["union_info"])
            iteration_times[kind].append(clock() - iteration_start)
        times[kind] = clock() - tick

    target_bias = entropy_bias(dm, n)
    mi_bias = {key: target_bias + entropy_bias(d, n) - entropy_bias(dm + d, n)
               for key, d in (("mix", dx), ("miy", dy), ("mixy", dx + dy))}
    shuffle_bias = sum(unions["shuffle"]) / n_shuffles
    bootstrap_bias = sum(unions["bootstrap"]) / n_resamples - raw["union_info"]
    union_bias = {"permutation": shuffle_bias, "combined": (shuffle_bias + bootstrap_bias) / 2}
    corrected = {name: components(*(raw[key] - mi_bias[key] for key in ("mix", "miy", "mixy")),
                                  raw["union_info"] - bias,
                                  disable_numerical_checks=disable_numerical_checks)
                 for name, bias in union_bias.items()}
    times["total"] = clock() - start
    return dict(raw=raw, **corrected, mi_bias=mi_bias, union_bias=union_bias,
                surrogate_unions=unions, timings=times, iteration_seconds=iteration_times,
                covariance=covariance, analytical_correction_approximate=covariance != "empirical")
