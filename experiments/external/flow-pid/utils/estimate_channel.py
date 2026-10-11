#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cvxpy as cp
import scipy.linalg as la
import numpy.linalg as npla
import warnings


def solve(a, b):
    return la.solve(a, b, assume_a='pos')


def whiten(cov, dm, dx, dy, ret_channel_params=False):
    """
    Whiten the X- and Y-channel noise covariances, and return a new joint
    covariance matrix between M, X and Y.

    Also standardizes the covariance of M to an identity matrix.

    Assumes invertibility of matrices where required:
        sig_m, sig_x__m, sig_y__m
    """

    # Variable name convention
    # - No separator implies joint auto-covariance between variables
    # - One underscore refers to cross-covariance between variables
    # - Two underscores refers to conditioning
    #
    # Examples
    # - sig_mxy: joint (auto-)covariance between M, X and Y
    # - sig_xy_m: cross-covariance between the stacked vector (X, Y) and M
    # - sig_x_m__y: conditional cross-covariance matrix between X and M given Y

    # First standardize M
    sig_mxy = cov.copy()
    sig_m = cov[:dm, :dm]
    sig_mxy[:, :dm] = solve(la.sqrtm(sig_m).real, sig_mxy[:, :dm].T).T
    sig_mxy[:dm, :] = solve(la.sqrtm(sig_m).real, sig_mxy[:dm, :])
    sig_m = sig_mxy[:dm, :dm]  # Redefine sig_m

    # Extract necessary parameters
    sig_x = sig_mxy[dm:dm+dx, dm:dm+dx]
    sig_y = sig_mxy[dm+dx:, dm+dx:]
    sig_x_m = sig_mxy[dm:dm+dx, :dm]  # Also equal to hx pre-whitening
    sig_y_m = sig_mxy[dm+dx:, :dm]    # Also equal to hy pre-whitening

    # Compute channel noise covariance matrices (TODO: Skip the solve here because sig_m = I?)
    sig_x__m = sig_x - sig_x_m @ solve(sig_m, sig_x_m.T)
    sig_y__m = sig_y - sig_y_m @ solve(sig_m, sig_y_m.T)

    # Whiten the X-channel
    sig_mxy[:, dm:dm+dx] = solve(la.sqrtm(sig_x__m).real, sig_mxy[:, dm:dm+dx].T).T
    sig_mxy[dm:dm+dx, :] = solve(la.sqrtm(sig_x__m).real, sig_mxy[dm:dm+dx, :])

    # Whiten the Y-channel
    sig_mxy[:, dm+dx:] = solve(la.sqrtm(sig_y__m).real, sig_mxy[:, dm+dx:].T).T
    sig_mxy[dm+dx:, :] = solve(la.sqrtm(sig_y__m).real, sig_mxy[dm+dx:, :])

    # Extract the final joint covariance of (X, Y) given M
    sig_xy = sig_mxy[dm:, dm:]
    sig_xy_m = sig_mxy[dm:, :dm]
    sig_xy__m = sig_xy - sig_xy_m @ solve(sig_m, sig_xy_m.T) # TODO: Skip solve?

    if ret_channel_params:
        return sig_mxy, sig_x_m, sig_y_m, sig_xy_m, sig_xy__m
    return sig_mxy


def robust_whiten(cov, dm, dx, dy, ret_channel_params=True, epsilon=1e-6):
    """
    Robust implementation of matrix whitening that handles near-singular matrices.

    Parameters:
    -----------
    cov : numpy.ndarray
        Covariance matrix to whiten
    dm : int
        Dimension of the first part of the matrix
    dx : int
        Input dimension
    dy : int
        Output dimension
    ret_channel_params : bool, optional
        Whether to return channel parameters
    epsilon : float, optional
        Small constant for numerical stability

    Returns:
    --------
    tuple
        Whitened matrix and channel parameters if ret_channel_params=True
    """

    def robust_solve(a, b):
        """
        Robust matrix solve that handles near-singular matrices using
        regularization and pseudoinverse when needed.
        """
        try:
            # First try regular solve with positive definite assumption
            return la.solve(a, b, assume_a='pos')
        except la.LinAlgError:
            # If that fails, try pseudoinverse approach
            a_reg = a + epsilon * np.eye(a.shape[0])
            return np.linalg.pinv(a_reg) @ b

    def robust_sqrtm(matrix):
        """
        Robust matrix square root that handles numerical issues.
        """
        # Add small regularization to ensure positive definiteness
        matrix_reg = matrix + epsilon * np.eye(matrix.shape[0])

        # Compute eigendecomposition
        eigvals, eigvecs = npla.eigh(matrix_reg)

        # Set any negative eigenvalues to small positive value
        eigvals = np.maximum(eigvals, epsilon)

        # Compute matrix square root
        sqrt_eigvals = np.sqrt(eigvals)
        return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

    # Copy input matrix
    sig_mxy = cov.copy()
    sig_m = cov[:dm, :dm]

    # Compute square root matrix robustly
    sqrt_sig_m = robust_sqrtm(sig_m)

    # Perform whitening transformation
    sig_mxy[:, :dm] = robust_solve(sqrt_sig_m, sig_mxy[:, :dm].T).T
    sig_mxy[:dm, :] = robust_solve(sqrt_sig_m, sig_mxy[:dm, :])

    # Update sig_m
    sig_m = sig_mxy[:dm, :dm]

    # Extract necessary parameters
    sig_x = sig_mxy[dm:dm + dx, dm:dm + dx]
    sig_y = sig_mxy[dm + dx:, dm + dx:]
    sig_x_m = sig_mxy[dm:dm + dx, :dm]  # Also equal to hx pre-whitening
    sig_y_m = sig_mxy[dm + dx:, :dm]  # Also equal to hy pre-whitening

    # Compute channel noise covariance matrices (TODO: Skip the solve here because sig_m = I?)
    sig_x__m = sig_x - sig_x_m @ robust_solve(sig_m, sig_x_m.T)
    sig_y__m = sig_y - sig_y_m @ robust_solve(sig_m, sig_y_m.T)

    # Whiten the X-channel
    sig_mxy[:, dm:dm + dx] = robust_solve(robust_sqrtm(sig_x__m).real, sig_mxy[:, dm:dm + dx].T).T
    sig_mxy[dm:dm + dx, :] = robust_solve(robust_sqrtm(sig_x__m).real, sig_mxy[dm:dm + dx, :])

    # Whiten the Y-channel
    sig_mxy[:, dm + dx:] = robust_solve(robust_sqrtm(sig_y__m).real, sig_mxy[:, dm + dx:].T).T
    sig_mxy[dm + dx:, :] = robust_solve(robust_sqrtm(sig_y__m).real, sig_mxy[dm + dx:, :])

    # Extract the final joint covariance of (X, Y) given M
    sig_xy = sig_mxy[dm:, dm:]
    sig_xy_m = sig_mxy[dm:, :dm]
    sig_xy__m = sig_xy - sig_xy_m @ robust_solve(sig_m, sig_xy_m.T)  # TODO: Skip solve?

    if ret_channel_params:
        return sig_mxy, sig_x_m, sig_y_m, sig_xy_m, sig_xy__m
    return sig_mxy


def recondition(x, max_cond=1e10, return_tf=False):
    """
    Utility function to remove correlated elements from a vector `x`, so that
    its covariance matrix has a condition number no greater than `max_cond`.

    The smallest eigenvalues are removed and the covariance matrix is reduced
    in size.

    Returns the transposed truncated eigenvector matrix (the effective
    transformation that removes said eigenvalues) if `return_tf` is True.
    """

    # Rows of x represent variables, columns represent realizations
    cov = np.cov(x)

    w, v = la.eigh(cov)
    w = w.real
    v = v.real  # w and v should already be real, but this step ensures it

    # Isolate indices causing large condition number
    bad_indices = np.where(w < w[-1] / max_cond)[0]
    start_index = bad_indices.max() + 1

    wsub = w[start_index:]
    vsub = v[:, start_index:]

    x_new = vsub.T @ x

    if return_tf:
        return x_new, vsub.T
    return x_new


def lin_tf_params_from_cov(cov, dm, dx, dy):
    """
    Utility function to extract linear transform parameters from a covariance
    matrix.

    Parameters
    ----------
    cov : np.ndarray [dm+dx+dy, dm+dx+dy]
            Covariance matrix to extract parameters from
    dm : int
            Dimension of M
    dx : int
            Dimension of X
    dy : int
            Dimension of Y

    Returns
    -------
    hx : np.ndarray [dx,dm]
            channel gain matrix for M->X (i.e. X|M has mean hx.dot(M))
    hy : np.ndarray [dy,dm]
            channel gain matrix for M->Y
    hxy : np.ndarray [dx+dy,m]
            channel gain matrix for M->(X,Y)
    sigx : np.ndarray [dx,dx]
            channel covariance matrix for M->X (i.e. X|M has cov sigx)
    sigy : np.ndarray [dy,dy]
            channel covariance matrix for M->Y
    sigxy : np.ndarray [dx+dy,dx+dy]
            channel covariance matrix for M->(X,Y)
    covxy : np.ndarray [dx+dy,dx+dy]
            covariance matrix for (X,Y)
    sigm : np.ndarray [dm,dm]
            covariance matrix for M
    """
    covm = cov[:dm, :dm]
    covx = cov[dm:dm+dx, dm:dm+dx]
    covy = cov[dm+dx:, dm+dx:]
    covxy = cov[dm:, dm:]
    covx_m = cov[:dm, dm:dm+dx].T
    covy_m = cov[:dm, dm+dx:].T
    covxy_m = cov[:dm, dm:].T

    # channel gain matrices (using multivariate conditional mean)
    # Assumes that covm is invertible
    hx = covx_m.dot(la.inv(covm))
    hy = covy_m.dot(la.inv(covm))
    hxy = covxy_m.dot(la.inv(covm))

    # channel covariance matrices (using mvar conditional covariance)
    sigm = covm
    sigx = covx - covx_m.dot(la.inv(covm)).dot(covx_m.T)
    sigy = covy - covy_m.dot(la.inv(covm)).dot(covy_m.T)
    sigxy = covxy - covxy_m.dot(la.inv(covm)).dot(covxy_m.T)

    # whiten
    # Assumes that sigx and sigy are invertible
    hx = la.sqrtm(la.inv(sigx)).dot(hx)
    sigx = np.eye(dx)
    hy = la.sqrtm(la.inv(sigy)).dot(hy)
    sigy = np.eye(dy)

    # XXX: sigxy is *not* whitened here!!!

    return hx, hy, hxy, sigx, sigy, sigxy, covxy, sigm


def lin_tf_params_ip_dfncy(cov, dm, dx, dy):
    covm = cov[:dm, :dm]
    covx = cov[dm:dm+dx, dm:dm+dx]
    covy = cov[dm+dx:, dm+dx:]
    covxy = cov[dm:, dm:]
    covx_m = cov[:dm, dm:dm+dx].T
    covy_m = cov[:dm, dm+dx:].T
    covxy_m = cov[:dm, dm:].T

    sigm = covm
    gx = covx_m @ la.sqrtm(la.inv(covy))
    gy = covy_m @ la.inv(covy)
    sigx = covx

    # return gx, gy, sigm, sigx, sigy, sigx_m, sigy_m
    return gx, gy, sigm, sigx


def lin_tf_params_bert(cov, dm, dx, dy):
    # XXX: Inconsistency: sigx, sigx_y, sigxy etc. refer to the respective auto-
    # or cross-covariance matrices in this function, *not* the conditional
    # covariance matrices (given M) as in lin_tf_params_from_cov.

    covm = cov[:dm, :dm]
    covx = cov[dm:dm+dx, dm:dm+dx]
    covy = cov[dm+dx:, dm+dx:]
    covxy = cov[dm:, dm:]
    covx_m = cov[:dm, dm:dm+dx].T
    covy_m = cov[:dm, dm+dx:].T
    covxy_m = cov[:dm, dm:].T
    covx_y = cov[dm:dm+dx, dm+dx:]

    ## Standardize M, X and Y and re-compute cross-covariances

    covm_sqrt = la.sqrtm(covm)  # TODO: Check that this is symmetric-sqrt
    covx_sqrt = la.sqrtm(covx)
    covy_sqrt = la.sqrtm(covy)

    # Just think of hx as the (standardized) cross-covariance between X and M
    hx = la.solve(covx_sqrt, la.solve(covm_sqrt, covx_m.T).T)
    hy = la.solve(covy_sqrt, la.solve(covm_sqrt, covy_m.T).T)
    # Standardized cross-covariance between X and Y
    sigx_y = la.solve(covx_sqrt, la.solve(covy_sqrt, covx_y.T).T)

    # Here, sigx and sigy are also just referring to the standardized
    # auto-covariance matrices, not the conditional covariance matrices
    sigm = np.eye(dm)
    sigx = np.eye(dx)
    sigy = np.eye(dy)

    # Also pass on some block matrices
    hxy = np.vstack((hx, hy))
    sigxy = np.block([[sigx, sigx_y], [sigx_y.T, sigy]])

    return hx, hy, sigx_y, sigm, sigx, sigy, hxy, sigxy


def remove_lin_dep_comps(cov, dm, dx, dy):
    # XXX: Untested
    """
    Utility function to remove linearly dependent components from the
    covariance matrix for M, in order to make it invertible.

    Parameters
    ----------
    cov : np.ndarray [dm+dx+dy, dm+dx+dy]
            Covariance matrix of M, X and Y
    dm : int
            Dimension of M
    dx : int
            Dimension of X
    dy : int
            Dimension of Y

    Returns
    -------
    cov_new : np.ndarray [dm+dx+dy, dm+dx+dy]
            New joint covariance matrix for M, X and Y
    dm_new : int
            New dimension of M
    """

    covm = cov[:dm, :dm]
    #covx = cov[dm:dm+dx, dm:dm+dx]
    #covy = cov[dm+dx:, dm+dx:]

    wm, vm = la.eigh(covm)
    wm = wm[::-1]
    vm = vm[:, ::-1]
    #wx, vx = la.eigh(covx)
    #wy, vy = la.eigh(covy)

    wm_thresholded_mask = (wm > 1e-10)
    wm_cumsum = np.cumsum(wm) / wm.sum()
    wm_var_capture_index = np.where(wm_cumsum < 1 - 1e-3)[0].max() + 1
    wm_var_mask = (np.arange(wm.size) <= wm_var_capture_index)
    mask = wm_thresholded_mask & wm_var_mask

    wm_new = wm[mask]
    vm_new = vm[:, mask]
    dm_new = wm_new.size

    transform = np.tile([[vm_new.T, np.zeros((dm, dx + dy))],
                         [np.zeros((dx + dy, dm)), np.eye(dx + dy)]])
    cov_new = transform @ cov @ transform.T

    return cov_new, dm_new


def make_cov_m_equals_xy(covxy, dx, dy, epsilon=0.0):
    """
    Utility function to make a joint covariance matrix for M, X and Y
    where M = (X, Y), given the covariance matrix for (X, Y).

    Parameters
    ----------
    covxy : np.ndarray [dx+dy, dx+dy]
            Covariance matrix of X and Y
    dx : int
            Dimension of X
    dy : int
            Dimension of Y

    Returns
    -------
    cov : np.ndarray [dm+dx+dy, dm+dx+dy]
            Covariance matrix of M, X and Y
    """
    covx = covxy[:dx, :dx]
    covy = covxy[dx:, dx:]
    covx_y = covxy[:dx, dx:]

    covm_x = np.vstack((covx, covx_y.T))
    covm_y = np.vstack((covx_y, covy))
    cov = np.block([[covxy, covm_x, covm_y],
                    [covm_x.T, covx + epsilon * np.eye(dx), covx_y],
                    [covm_y.T, covx_y.T, covy + epsilon * np.eye(dy)]])

    return cov


def heuristic_channel(di, do, hi, ho, sigi, sigo, sigm,
                      verbose=False, maxiter=5000, eps=1e-10):
    """
    Approximate the deficiency-minimizing channel from a input to
    a output conditioned on M. For example, when considering
    \delta(M:X\Y), we are estimating a channel from Y to X, so we call
    Y the input and X the output

    Param
    -----------
    di : int
            dimension of input
    do : int
            dimension of output
    hi : np.ndarray [di,dm]
            channel gain matrix for M->input
    ho : np.ndarray [do,dm]
            channel gain matrix for M->output
    sigi : np.ndarray [di,di]
            channel covariance matrix for M->input
    sigo : np.ndarray [do,do]
            channel covariance matrix for M->output
    sigm : np.ndarray [dm,dm]
            covariance matrix for M
    verbose : bool
            whether or not to print

    Returns
    -------
    t : np.ndarray [do,di]
            channel gain matrix for input->output
    sigt : np.ndarray [do,do]
            channel covariance matrix for input->output
    """

    t = cp.Variable((do, di))
    A = sigo + ho @ sigm @ ho.T
    C = np.linalg.inv(sigi + hi @ sigm @ hi.T)
    B = t
    M = cp.bmat([[A, B], [B.T, C]])
    Ai = la.sqrtm(np.linalg.inv(sigo + ho @ sigm @ ho.T))
    #Ai = np.eye(do, do)
    sqrtm = la.sqrtm(sigm)
    objective = cp.Minimize(cp.norm(Ai @ t @ hi @ sqrtm - Ai @ ho @ sqrtm, 'fro'))
    constraints = [M >> 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver='SCS', verbose=verbose,
                        alpha=1, max_iters=maxiter, eps=eps)

    if 'unbounded' in prob.status or 'infeasible' in prob.status:
        raise ValueError('Convex problem was %s' % prob.status)
    if 'inaccurate' in prob.status:
        warnings.warn('Inaccurate solution')

    t = t.value

    #sigt = sigo + ho.dot(sigm).dot(ho.T) - t.dot(sigi + hi.dot(sigm).dot(hi.T)).dot(t.T)
    sigt = sigo + ho @ sigm @ ho.T - t @ (sigi + hi @ sigm @ hi.T) @ t.T

    w, v = np.linalg.eigh(sigt)
    sigt = v.dot(np.diag(w)).dot(v.T)
    wclip = w.clip(min=0)
    sigt = v.dot(np.diag(wclip)).dot(v.T)
    if verbose:
        if((wclip != w).any()):
            print('Negative eigenvalues:')
            print(w)

    return t, sigt


def exp_mvar_kl(h1, sig1, h2, sig2, sigm):
    """
    Compute the KL-divergence between two multivariate
    Gaussian channel outputs, taking an expectation w.r.t. the channel
    input

    Param
    -----------
    h1 : np.ndarray
            KLD first argument channel gain matrix
    sig1 : np.ndarray
            KLD first argument channel covariance matrix
    h2 : np.ndarray
            KLD second argument channel gain matrix
    sig2 : np.ndarray
            KLD secondt argument channel covariance matrix
    sigm : np.ndarray
            covariance matrix for channel input

    Returns
    -------
    expected kl divergence : float
    """
    #t1 = np.log(np.linalg.det(sig2)/np.linalg.det(sig1))
    t1 = np.linalg.slogdet(sig2)[1] - np.linalg.slogdet(sig1)[1]
    t2 = h1.shape[0]
    #t3 = np.trace(np.linalg.inv(sig2).dot(sig1))
    t3 = np.trace(la.solve(sig2, sig1, assume_a='pos'))
    #t4 = np.trace(sigm.dot((h2-h1).T).dot(np.linalg.inv(sig2)).dot(h2-h1))
    t4 = np.trace(sigm @ (h2-h1).T @ la.solve(sig2, h2-h1, assume_a='pos'))
    #print('\n\n\n%g\n\n\n' % (0.5 * (t1 + t3 + t4 - t2) / np.log(2)))
    return 0.5 * (t1 - t2 + t3 + t4) / np.log(2)


def approx_pid(hx, hy, hxy, sigx, sigy, sigxy, covxy, sigm, maxiter=5000, eps=1e-10, verbose=True, ret_t_sigt=False):
    """
    Approximate a PID using the outputs from the generate function

    Param
    -----------
    hx : np.ndarray [dx,dm]
            channel gain matrix for M->X (i.e. X|M has mean hx.dot(M))
    hy : np.ndarray [dy,dm]
            channel gain matrix for M->Y
    hxy : np.ndarray [dx+dy,m]
            channel gain matrix for M->(X,Y)
    sigx : np.ndarray [dx,dx]
            channel covariance matrix for M->X (i.e. X|M has cov sigx)
    sigy : np.ndarray [dy,dy]
            channel covariance matrix for M->Y
    sigxy : np.ndarray [dx+dy,dx+dy]
            channel covariance matrix for M->(X,Y)
    covxy : np.ndarray [dx+dy,dx+dy]
            covariance matrix for (X,Y)
    sigm : np.ndarray [dm,dm]
            covariance matrix for M

    Returns
    -------
    imx : float
            mutual information between M and X
    imy : float
            mutual information between M and Y
    imxy : float
            mutual information between M and (X,Y)
    defx : float
            deficiency of X w.r.t. Y
    defy : float
            deficiency of Y w.r.t. X
    uix : float
            unique information in X (about M, unique w.r.t Y)
    uiy : float
            unique information in Y
    ri : float
            redundant information in X and Y
    si : float
            synergistic information in X and Y
    """

    dx = hx.shape[0]
    dy = hy.shape[0]

    # compute mutual informations
    imx = exp_mvar_kl(hx, sigx, np.zeros_like(hx),
                      hx.dot(sigm).dot(hx.T) + sigx, sigm)
    imy = exp_mvar_kl(hy, sigy, np.zeros_like(hy),
                      hy.dot(sigm).dot(hy.T) + sigy, sigm)
    imxy = exp_mvar_kl(hxy, sigxy, np.zeros_like(hxy), covxy, sigm)

    # approximate channel Y-> X (unique info in X)
    t, sigt = heuristic_channel(dy, dx, hy, hx, sigy, sigx, sigm,
                                verbose=verbose, maxiter=maxiter, eps=eps)
    # get deficiency of Y w.r.t. X
    def_x_minus_y = exp_mvar_kl(hx, sigx, t @ hy, t @ sigy @ t.T + sigt, sigm)
    tx, sigtx = t.copy(), sigt.copy()

    # approximate channel X->Y
    t, sigt = heuristic_channel(dx, dy, hx, hy, sigx, sigy, sigm,
                                verbose=verbose, maxiter=maxiter, eps=eps)
    # get deficiency of X w.r.t. Y (unique info in Y)
    def_y_minus_x = exp_mvar_kl(hy, sigy, t @ hx, t @ sigx @ t.T + sigt, sigm)
    ty, sigty = t.copy(), sigt.copy()

    # compute PID
    ri = min(imx - def_x_minus_y, imy - def_y_minus_x)
    uix = imx - ri
    uiy = imy - ri
    si = imxy - uix - uiy - ri

    ret = (imx, imy, imxy, def_y_minus_x, def_x_minus_y, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, tx, sigtx, ty, sigty)

    return ret


def approx_pid_from_cov(cov, dm, dx, dy, verbose=False, ret_t_sigt=False):
    """
    Compute the approximate Gaussian PID from a covariance matrix.
    """

    params = lin_tf_params_from_cov(cov, dm, dx, dy)
    return approx_pid(*params, verbose=verbose, ret_t_sigt=ret_t_sigt)