#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import scipy.linalg as la


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

    return gx, gy, sigm, sigx, sigy, sigx_m, sigy_m


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
