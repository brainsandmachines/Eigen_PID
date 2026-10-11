# Thin PID core algorithm
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import warnings
import numpy as np
import scipy.linalg as la
import numpy.linalg as npla

from .tilde_pid import project as tilde_project
from utils import whiten, pinv


# Suppress the specific warning: linAlgWarning: Ill-conditioned matrix since we can just resolve it or approximate it
# warnings.filterwarnings("ignore", category=la.LinAlgWarning)

def objective(sig, hx, hy, dm, dx, dy, reg):  # the objective function
    dx, dy = sig.shape

    H = np.concatenate((hx, hy), axis=0)
    sig_all = np.block([[np.eye(dx), sig],
                        [sig.T, np.eye(dy)]])

    obj = 0.5 / np.log(2) * (npla.slogdet(H @ H.T + sig_all)[1] - npla.slogdet(sig_all)[1])
    return obj


def gradient(sig, hx, hy, dm, dx, dy, reg):
    H = np.concatenate((hx, hy), axis=0)
    sig_all = np.block([[np.eye(dx), sig],
                        [sig.T, np.eye(dy)]])

    G = H @ H.T + sig_all
    G_11 = G[0:dx, 0:dx]
    G_12 = G[0:dx, dx:dx + dy]
    G_22 = G[dx:dx + dy, dx:dx + dy]
    G11_inv_G12 = la.solve(G_11, G_12)

    # Matrix computation intermediate steps to compute the gradient
    A = G_22 - G_12.T @ G11_inv_G12 + reg * np.eye(dy)
    A1 = np.eye(dy) - sig.T @ sig + reg * np.eye(dy)

    sol_AB = npla.solve(A, G11_inv_G12.T)
    sol_AB1 = npla.solve(A1, sig.T)
    g_sig = -sol_AB + sol_AB1
    g_sig = g_sig.T
    return g_sig


def thin_project(sig_temp):  # project the matrix onto the PSD cone, but only need to work with the upper triangular part

    U, S, VT = npla.svd(sig_temp, full_matrices=False)
    S_clamped = np.clip(S, 0, 0.99999999999999999)  # clamp to [1e-10,0.99999999999999999] to avoid numerical issues
    S_clamped_matrix = np.diag(S_clamped)
    sig_proj = U @ S_clamped_matrix @ VT

    return sig_proj, True


def exact_thin_pid_minimizer(hx, hy, plot=False, ret_obj=False, reg=1e-7, max_iters=20000, verbose=False, init_sig=None):
    dx, dm = hx.shape
    dy, dm_ = hy.shape
    if dm != dm_:
        raise ValueError('Incompatible shapes for Hx and Hy')

    swap = False
    if dx < dy:  # Swap if necessary, since we assume dx >= dy
        dx, dy = dy, dx
        hy, hx = hx, hy
        swap = True

    # Gradient descent
    eta_sig = 1e-3 * np.ones((dx, dy))
    beta = 0.9  # Factor to increase or decrease LR for Rprop
    alpha = 0.999  # Slow decay of overall learning rate

    stop_threshold = reg
    max_iterations = max_iters  # Maximum number of iterations
    patience = 20  # Num iters with small gradient before stopping (min=1)
    extra_iters = 0  # Num of extra iters after stop criterion is attained

    minima = None
    g_sig_prev = None
    running_obj = []
    i = 1
    extra = 0

    if init_sig is None:
        sig_temp = hx @ pinv(hy)
        try:
            sig_temp_proj, _ = thin_project(sig_temp)
        except npla.LinAlgError:
            warnings.warn('Thin projection failed, falling back to tilde projection.')
            sig_temp_proj, _ = tilde_project(sig_temp)
        sig = sig_temp_proj.copy()
    else:
        sig = np.asarray(init_sig).T.copy() if swap else np.asarray(init_sig).copy()
        if sig.shape != (dx, dy):
            raise ValueError(f'init_sig has shape {sig.shape}, expected {(dx, dy)}')

    obj_hist = np.array([])
    while True:
        # Evaluate the objective
        obj = objective(sig, hx, hy, dm, dx, dy, reg)

        if minima is None or obj < min(running_obj):
            minima = (sig.copy(), obj)

        if len(running_obj) >= patience:
            if extra == 0:
                if (np.abs(np.array(running_obj[-patience:]) - obj) < stop_threshold).all() or i >= max_iterations:
                    if i >= max_iterations:

                        warnings.warn('Exceeded maximum number of iterations. May not have converged.')
                    if extra_iters == 0: break
                    extra += 1
            elif extra > extra_iters:
                break
            else:
                extra += 1

        if np.isnan(obj):
            running_obj.append(np.inf)
        else:
            running_obj.append(obj)
        i += 1

        g_sig = gradient(sig, hx, hy, dm, dx, dy, reg)
        g_sig = np.sign(g_sig).astype(int)

        # gradient descent
        sig_plus = sig - alpha ** i * eta_sig * g_sig
        # project sig back onto the PSD cone, but only need to work with the upper triangular part
        try:
            sig_proj, _ = thin_project(sig_plus)
        except npla.LinAlgError:
            warnings.warn('Thin projection failed, falling back to tilde projection.')
            sig_proj, _ = tilde_project(sig_plus)

        # Learning rate update
        if g_sig_prev is not None:
            sign_changed = - g_sig * g_sig_prev  # -1 if sign did not change, +1 if sign changed
            eta_sig *= beta ** sign_changed

        g_sig_prev = g_sig
        sig[:, :] = sig_proj

        if verbose:
            obj_hist = np.append(obj_hist, obj)

    sig, obj = minima

    if swap:
        sig = sig.T

    if ret_obj:
        return sig, obj, i, obj_hist
    return sig


def bias(d, n):
    return sum(np.log(1 - k / n) for k in range(1, d+1)) / np.log(2) / 2


def compute_bias(du, dv, n):
    """
    Compute the bias in the mutual information estimate based on the work of
    Cai et al. (J. Mult. Anal., 2015).

    This value needs to be subtracted from the mutual information estimate to
    recover the unbiased mutual information.
    """
    # Bias of differential entropy estimate
    return bias(du, n) + bias(dv, n) - bias(du + dv, n)


def debias(imxy, bias_):
    """Remove bias while ensuring non-negativity."""

    return np.maximum(imxy - bias_, 0)


def exact_gauss_thin_pid(cov, dm, dx, dy, verbose=False, ret_t_sigt=False,
                          plot=False, unbiased=False, sample_size=None,
                          init_cov=None):

    # XXX: Debiasing has not been thoroughly tested.
    # Right now, we assume that the proportion of bias in the union information
    # is the same as the proportion of bias in I(M ; (X, Y)).

    # Regularization
    reg = 1e-7

    if unbiased == True and sample_size is None:
        raise ValueError('Must supply sample_size when requesting unbiased estimates')

    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    init_sig = None
    if init_cov is not None:
        init_cov = np.asarray(init_cov)
        mask = np.ones(cov.shape, dtype=bool)
        mask[dm:dm+dx, dm+dx:] = mask[dm+dx:, dm:dm+dx] = False
        if init_cov.shape != cov.shape or not np.allclose(init_cov[mask], cov[mask], rtol=1e-7, atol=1e-9):
            raise ValueError('init_cov must preserve Cov(M), Cov(M,X), Cov(X), Cov(M,Y), and Cov(Y)')
        init_sig = whiten(init_cov, dm, dx, dy, ret_channel_params=True)[-1][:dx, dx:]

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(sigxy + reg * np.eye(*sigxy.shape), hxy))[1] / np.log(2)

    if unbiased:
        imx = debias(imx, compute_bias(dm, dx, sample_size))
        imy = debias(imy, compute_bias(dm, dy, sample_size))
        imxy_debiased = debias(imxy, compute_bias(dm, dx + dy, sample_size))

        # But ensure that the debiased imxy does not go below the debiased imx
        # or the debiased imy, as this will make PID values negative
        imxy_debiased = max(imxy_debiased, imx, imy)
    else:
        imxy_debiased = imxy

    debias_factor = imxy_debiased / imxy

    sig, obj, _, obj_hist = exact_thin_pid_minimizer(hx, hy, plot=plot, ret_obj=True, reg=reg, verbose=verbose, init_sig=init_sig)

    union_info = objective(sig, hx, hy, dm, dx, dy, reg=reg)
    union_info *= debias_factor

    # Union info is lower bounded by max{I(M; X), I(M; Y)} and upper bounded by
    # min{I(M; X) + I(M; Y), I(M; (X, Y))}: imposing this ensures positivity of
    # the PID terms
    union_info = max(union_info, imx, imy)
    union_info = min(union_info, imx + imy, imxy_debiased)

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy_debiased - union_info

    # Return union_info and None in place of deficiency values to keep return signature consistent
    ret = (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    if verbose:
        return ret, obj_hist

    return ret
