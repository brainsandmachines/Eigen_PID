# Gaussian reference implementations

This directory contains the Gaussian solver and utility files used by the
synthetic comparisons. No Git checkout or solver download is needed.

- **Thin-PID:** Zhao et al.'s [Flow-PID implementation](https://github.com/warrenzha/flow-pid),
  based on upstream revision `4a30b1a3ff3eb9d181fc49dff023930a617a86b3`.
- **Tilde-PID:** Venkatesh et al.'s [GPID implementation](https://github.com/praveenv253/gpid),
  based on upstream revision `6a2b15e637681f329590755464ee01c11dd66d27`.

Both implementations include a small modification allowing an optional
`init_cov` argument: it is checked against the original target–source marginals
and whitened to initialize the optimizer. Without this argument, the original
initialization is retained. Thin-PID also transposes the supplied initialization
when it swaps the sources internally. Tilde-PID also evaluates final union information without regularization.
Regularization during optimization and in the joint mutual information is
retained. The optimization procedures are unchanged.

The original attribution is retained. GPID declares the MIT license in its
supplied `setup.py`; Flow-PID did not specify a software license at the time of
access. These files are separate from the Eigen-PID package and are excluded
from its wheel and source distribution.

The optional fMRI reference benchmarks use different, unmodified upstream
revisions documented in [the fMRI README](../../fmri/README.md).
