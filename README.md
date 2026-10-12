<p align="center">
  <img src="assets/eigen-pid.webp" alt="Eigen-PID logo" width="480">
</p>

Eigen-PID calculates the bivariate population-Gaussian BROJA partial
information decomposition (PID). It separates the information that two
sources, `X` and `Y`, carry about a target, `M`, into:

- information unique to `X`;
- information unique to `Y`;
- redundant information shared by both sources; and
- synergistic information available only from the sources together.

The implementation uses PyTorch and supports CPU and CUDA tensors. Information
values are returned in bits.

## Paper and citation

If you use Eigen-PID in your research, please cite:

Sheelo, O., Golan, T., & Pakman, A. (2026). *Gaussian BROJA Partial Information
Decomposition in Closed Form*. arXiv:2610.13464.
https://arxiv.org/abs/2610.13464

```bibtex
@misc{sheelo2026gaussianbroja,
  title         = {Gaussian {BROJA} Partial Information Decomposition in Closed Form},
  author        = {Sheelo, Ohad and Golan, Tal and Pakman, Ari},
  year          = {2026},
  eprint        = {2610.13464},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT},
  url           = {https://arxiv.org/abs/2610.13464}
}
```

## Install Eigen-PID for your own calculations

Install directly from GitHub:

```bash
python -m pip install "git+https://github.com/brainsandmachines/Eigen_PID.git"
```

This installs the `eigen_pid` Python package and its required dependency,
PyTorch. Neither `experiments/` nor `fmri/` is included in the wheel or source
distribution; their plotting and comparison dependencies are optional.

If you have cloned the repository and want only the Eigen-PID package, use the
minimal package requirements file:

```bash
python -m pip install -r requirements.txt
```

The lightweight examples in [`basic_usage/`](basic_usage/) are available in the
repository for reference. They are not copied into your Python environment by
pip.

To run the core and debiasing tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Usage: Compute Eigen-PID from population joint-covariance

The covariance matrix must be ordered as `[M, X, Y]`:

```python
import torch

from eigen_pid import eigen_pid

covariance = torch.tensor(
    [
        [1.0, 0.6, 0.4],
        [0.6, 1.0, 0.2],
        [0.4, 0.2, 1.0],
    ],
    dtype=torch.float64,
)

result = eigen_pid(covariance, dm=1, dx=1, dy=1)

print("Unique information from X:", result["uix"])
print("Unique information from Y:", result["uiy"])
print("Redundant information:", result["red"])
print("Synergistic information:", result["syn"])
```

`dm`, `dx`, and `dy` are the dimensions of `M`, `X`, and `Y`. Batched
covariance tensors are supported by adding leading dimensions.
Array-like covariance inputs are converted to float64 Torch tensors.

The main result fields are:

| Key | Quantity |
| --- | --- |
| `mix` | Mutual information `I(M; X)` |
| `miy` | Mutual information `I(M; Y)` |
| `mixy` | Joint mutual information `I(M; X, Y)` |
| `uix` | Information about `M` unique to `X` |
| `uiy` | Information about `M` unique to `Y` |
| `red` | Redundant information |
| `syn` | Synergistic information |
| `union_info` | BROJA union information |

Pass `compute_optimal_covariance=True` to also return `gamma_star`,
`optimal_cov`, and `optimal_cov_xy`.

### Numerical requirements

- Tensor inputs must be finite PyTorch `float32` or `float64` covariance matrices.
  Array-like inputs are converted to float64 Torch tensors.
- Covariance blocks are ordered `[M, X, Y]`.
- `Cov(X)` and `Cov(Y)` must be positive definite and sufficiently
  conditioned.
- Joint `Cov(X, Y)` may be singular and uses a symmetric Moore-Penrose
  pseudoinverse.
- `Cov(M | X, Y)` must be positive definite and sufficiently conditioned.
- Validation reports invalid inputs; it does not symmetrize, clip, or add
  numerical jitter.

Numerical checks are enabled by default and preserve solver gradients.
CPU and CUDA condition estimates can differ slightly near a rejection threshold.

Pass `disable_numerical_checks=True` to skip symmetry, PSD, conditioning, and
output consistency diagnostics, including those for the optional optimal
covariance. This saves computation but assumes numerically suitable inputs.
Input shape, type, device, and finiteness checks and Cholesky failure detection
remain active. The flag does not change the PID formulas or regularization.

For more package examples, see
[`basic_usage/basic_usage.py`](basic_usage/basic_usage.py) and
[`basic_usage/compare_cpu_gpu.py`](basic_usage/compare_cpu_gpu.py).

### GPU computation

Pass a CUDA covariance tensor to compute Eigen-PID on the GPU. PyTorch >=2.12
is recommended for its [improved CUDA eigensolver performance](https://github.com/pytorch/pytorch/issues/178979)
(`torch.linalg.eigvalsh`).

If Torch CUDA execution reports `Failed to find C compiler`, install a host C
compiler and ensure it is on `PATH` (or set `CC`). Some Torch operations use
[Triton kernels](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/build.py)
even without `torch.compile`; this occurred with batched scalar inputs on
Torch 2.14.1+cu130. Eigen-PID itself has no compiled extension.

## Usage: Compute Eigen-PID from sample joint-covariance

`debias_eigen_pid` reimplements the correction procedure of
[Lorenz et al. (2026)](https://doi.org/10.1016/j.patter.2026.101619), using
Eigen-PID for the original covariance and every surrogate.

```python
from eigen_pid import debias_eigen_pid

# data: observations in rows; columns ordered as target (2), X (3), Y (4).
result = debias_eigen_pid(data, dm=2, dx=3, dy=4, seed=0)
print(result["combined"]["syn"])  # Corrected synergy in bits.
```

The result contains `raw`, `permutation`, and `combined` estimates, plus bias
estimates, surrogate union information, and timings. Defaults use sample
covariance, 20 target permutations, and 20 Gaussian bootstraps. Both corrected
methods remove analytical Gaussian MI bias; `combined` averages permutation and
bootstrap estimates of union-information bias. Negative estimates are retained.

Data must have more observations than joint dimensions and a positive-definite
fitted covariance. The analytical correction assumes iid Gaussian observations.
Computation uses float64 on the input tensor's device (CPU for array-like inputs).
Pass CUDA data for Torch GPU computation.
Numerical checks remain enabled unless `disable_numerical_checks=True`.
Debiasing does not support autograd; GPU timings synchronize the input device.

The default `seed=None` advances the caller's Torch RNG without reseeding it.
An explicit seed uses a local generator; CPU and CUDA draws can differ.
Optional `covariance="ledoit-wolf"` requires scikit-learn and makes the analytical
correction approximate; `progress=True` requires tqdm. See
`help(debias_eigen_pid)` for all arguments and returned fields.

For an application to neural data, see the [fMRI demonstration](fmri/README.md).

## Reproduce the research experiments

The repository contains synthetic comparisons in [`experiments/`](experiments/)
and the paper's fMRI appendix analysis in [`fmri/`](fmri/README.md).

### Required setup

**Setup required:** Before running synthetic or fMRI experiments, including
regenerating plots from saved results, install the corresponding dependencies
below. Experiment imports assume Eigen-PID is installed as a package and will
fail otherwise.

Clone the repository and run the installation commands from its root directory:

```bash
git clone https://github.com/brainsandmachines/Eigen_PID.git
cd Eigen_PID
```

**Synthetic experiments and plotting:**

```bash
python -m pip install -r experiments/requirements-experiments.txt
```

This installs Eigen-PID in editable mode together with the synthetic experiment
and plotting dependencies.

**fMRI experiments and plotting:**

```bash
python -m pip install -e .
python -m pip install -r fmri/requirements.txt
```

### Synthetic experiments

The synthetic experiments include the Gaussian comparison code for Thin-PID
and Tilde-PID, with the initialization modifications described in
[`experiments/external/`](experiments/external/README.md).

The active experiments are:

| Experiment | Purpose |
| --- | --- |
| Gain sweep | Compare Eigen-PID, Thin-PID, and Tilde-PID over increasing gain in `X` |
| Runtime sweep | Measure Eigen-PID runtime while varying one variable's dimension |
| Optimal initialization comparison | Compare optimizer results initialized with Eigen-PID's optimal covariance |

Run these commands from the `experiments/` directory:

```bash
cd experiments

python -m examples.thin_pid_comparison.compare_gain_sweep
python -m examples.thin_pid_comparison.compare_runtime
python -m examples.thin_pid_comparison.compare_runtime_optimal_covariance
```

#### Gain-sweep figure

The gain sweep runs all three PID methods on the same covariance matrices. It
creates:

```text
experiments/examples/thin_pid_comparison/results/gain_sweep/
├── gain_sweep_results.csv
└── gain_sweep.pdf
```

The PDF contains the PID values from each method and the signed differences
between Eigen-PID and the numerical methods.

#### Runtime and optimal initialization comparisons

Edit
[`runtime_config.yaml`](experiments/examples/thin_pid_comparison/runtime_config.yaml)
before running the runtime sweep or optimal initialization comparison. The main
settings are:

- `dimensions`: dimensions included in the sweep;
- `vary`: the variable to vary (`M`, `X`, or `Y`);
- `dm`, `dx`, and `dy`: the fixed/default dimensions;
- `trials`: independent trials per dimension;
- `seed_start`: first random seed;
- `percentiles`: lower and upper bounds of the plotted runtime band; and
- `resume`: whether to continue a partially completed regular runtime sweep.

The regular runtime command writes the raw trials, a copy of the configuration,
and `runtime_scaling.pdf` under:

```text
experiments/examples/thin_pid_comparison/results/runtime/<configuration>/
```

The optimal initialization comparison requires `vary: X`. It writes the raw
results, copied configuration, and `union_information_differences.pdf` under:

```text
experiments/examples/thin_pid_comparison/results/runtime_optimal_covariance/<configuration>/
```

### fMRI appendix: Eigen-PID applied to visual fMRI responses

The [`fmri/`](fmri/README.md) directory contains the code, saved measurements, and
figures for the paper's appendix, “Eigen-PID applied to visual fMRI responses.”
The analysis applies Eigen-PID with finite-sample correction to visual cortical
responses from eight NSD participants. Follow its README to regenerate PID figures and
tables from saved results without downloading fMRI data or running PID estimation,
or to rerun the analysis from data.

## Repository layout

```text
.
├── pyproject.toml
├── src/eigen_pid/          # Installable Eigen-PID package
├── tests/                  # Package tests
├── basic_usage/            # Lightweight package examples
├── fmri/                   # fMRI appendix code, saved results, and plots
└── experiments/
    ├── benchmarks/         # Shared comparison adapters and cases
    ├── examples/           # Reported research experiments
    ├── external/
    │   ├── flow-pid/       # Included Gaussian Thin-PID comparison code
    │   └── gpid/           # Included Gaussian Tilde-PID comparison code
    └── tests/              # Experiment tests
```

Only `src/eigen_pid/` becomes an importable package when Eigen-PID is installed.
The experiment tree remains repository source used for research reproduction.

## License and attribution

Eigen-PID’s own code is licensed under the [MIT license](LICENSE). Bundled
reference implementations and external datasets have separate terms; see
[third-party notices](THIRD_PARTY_NOTICES.md).
