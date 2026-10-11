import numpy as np
import scipy.linalg as la


def pinv(a):
    """
    la.pinv sometimes raises an la.LinAlgError: SVD did not converge. This
    appears to be some kind of BLAS/LAPACK bug:
    https://github.com/numpy/numpy/issues/12941.

    The solution appears to be to use the gesvd driver for the SVD, which is
    slower, but does not appear to cause this problem. In our case, this
    means we need a manual implementation of pinv that uses this driver.

    This function is a simplified version of the scipy implementation:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/linalg/_basic.py#L1319-L1464
    """

    # NOTE: We could use a try-except block to try using the default la.pinv
    # first, and only use this function if that fails, but since we do not use
    # pinv in any time-critical loops, it might be better to simply have a
    # slower but more accurate result.
    u, s, vh = la.svd(a, lapack_driver='gesvd')
    t = u.dtype.char.lower()
    maxS = np.max(s)

    atol = 0.
    rtol = max(a.shape) * np.finfo(t).eps

    val = atol + maxS * rtol
    rank = np.sum(s > val)

    u = u[:, :rank]
    u /= s[:rank]
    B = (u @ vh[:rank]).conj().T

    return B