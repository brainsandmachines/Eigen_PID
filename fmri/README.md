# Visual fMRI demonstration

This example decomposes the information that left and right V1 responses carry
about left or right hV4 responses in eight NSD participants. It includes data
loading, bias correction, saved estimates, figure generation, and runtime comparisons.
The installed Eigen-PID package does not include this directory or its dependencies.

## Reproduce figures and tables from saved results

The supplied `results/` and `plots/` contain the completed measurements and
figures. Regenerating the two paper PID figures and two tables requires no fMRI data
download, PID estimation, or reference-solver installation. The cortical
illustration is supplied separately; regenerating it requires surface meshes and
ROI masks as described below.

Run all commands from the repository root, the directory containing
`pyproject.toml`. Use Python 3.10 or newer and a suitable PyTorch build; a CPU build
is sufficient for this demonstration. The scripts use Unix filesystem features
and have been tested on Linux.

Before running experiments or regenerating plots, complete the
[required setup](../README.md#required-setup) for fMRI in the repository README.

**Plotting prerequisite:** Arial must be installed and discoverable by Matplotlib.
It is not included in the Python dependencies. Install it through your operating
system's font manager before plotting, then check availability with:

```bash
python -c "from matplotlib import font_manager; print(font_manager.findfont('Arial', fallback_to_default=False))"
```

The following commands copy the supplied results to a new `reproduced/results/`
directory, rebuild the exports, and write figures and tables under `reproduced/`.
Choose a new destination if that results directory already exists.

```bash
python -c "import shutil; shutil.copytree('fmri/results', 'reproduced/results')"
python fmri/code/run_experiment.py --output reproduced/results --export-only
python fmri/code/summarize_results.py --results reproduced/results
python fmri/code/plot_results.py --results reproduced/results --output reproduced/plots
python fmri/code/export_numbers.py --results reproduced/results --output reproduced/tables
```

Expected outputs are:

- `reproduced/plots/pid_raw.{pdf,png}` and `pid_combined.{pdf,png}`: the two paper PID figures.
- `reproduced/tables/dimensions_table.tex`, `runtime_table.tex`, and `numbers.tex`:
  the two LaTeX tables and numerical macros.
- `reproduced/results/`: regenerated estimate, timing, dimension, and plot-data
  exports alongside the copied case records and reference measurements.

## Rerun the fMRI analysis from data

Complete the [required setup](../README.md#required-setup) for fMRI first.
`DATA_DIR`, `RESULTS_DIR`, and `PLOTS_DIR` below are directory paths to replace
with your own choices. Use a data directory outside the repository and a fresh
results directory for a new analysis.

Obtain access through the [Algonauts 2023 data request](https://algonautsproject.com/2023/challenge.html).
The downloader skips images by default, resumes downloads, and validates existing
NPY files before skipping them. Responses and surface meshes are not redistributed.

```bash
python fmri/code/download_algonauts.py DATA_DIR
python fmri/code/run_experiment.py --data DATA_DIR --output RESULTS_DIR --progress --disable-numerical-checks
python fmri/code/summarize_results.py --results RESULTS_DIR
python fmri/code/plot_results.py --results RESULTS_DIR --output PLOTS_DIR
```

The analysis writes 32 case records plus `estimates.csv` and `timings.csv`.
Matching completed cases are skipped on restart. The plot command requires the
complete eight-participant analysis.

CPU is the default. To reproduce the GPU timing analysis (real-data panels A--B),
add `--device cuda --conditions real` and use a separate results directory.
PyTorch >=2.12 is recommended for GPU computation.
Use separate output directories for different backends. The runner passes explicit
resampling seeds (`--seed 2026` by default); CPU and CUDA draws can differ even with
the same seed. The reusable `debias_eigen_pid` function defaults to `seed=None`,
which uses the caller's Torch RNG without reseeding it.

Numerical checks are enabled by default; condition numbers use symmetric
eigenvalues on the input device. For numerically suitable inputs,
`--disable-numerical-checks` skips symmetry, PSD, conditioning, and output
diagnostics while retaining input validation and Cholesky failure detection.
This setting is recorded in the run configuration; use a separate output directory
when changing it. The paper measurements and commands above disable numerical
checks; this does not change the PID formulas or regularization.

A new analysis does not produce the Thin-PID runtime measurements used by the
table exporter. Summaries work without them and omit reference comparisons.
To generate those comparisons for a new run, follow the optional benchmark
workflow below before running `summarize_results.py` and `export_numbers.py`.
To reproduce the reported tables using the supplied
measurements, use the saved-results workflow above.

## Analysis

Algonauts supplies NSD responses standardized within scan session and averaged
across image repetitions. Each row corresponds to a distinct image. V1 combines
V1v and V1d; exact duplicate response columns within each region are removed.
No PCA or additional scaling is applied.

The experiment uses ordinary sample covariance and 20 target permutations plus
20 Gaussian parametric bootstraps per case. Both targets are analyzed in real data
and in one shuffled-target control per participant: 32 cases and 1,312 Eigen-PID
evaluations. The real-data analysis alone comprises 656 evaluations. Shuffling
preserves paired sources and each region's multivariate response distribution.

Following [Lorenz et al. (2026)](https://doi.org/10.1016/j.patter.2026.101619),
the permutation estimate of union-information bias is its mean across target
permutations. The bootstrap estimate is its mean across samples from the fitted
Gaussian, minus the fitted population value. The combined correction averages
these biases. Both correction methods use analytical Gaussian sample-covariance
biases for the three mutual informations, then reconstruct the four atoms.
Negative estimates are retained. The supplied analysis uses CPU Torch computation
and the upstream solver default `add_reg=False`; no additional regularization
is applied.

The analytical correction assumes independent Gaussian observations and ordinary
sample covariance. Optional `--covariance ledoit-wolf` applies shrinkage consistently
to observations and surrogates but makes the analytical correction approximate.
Shuffled controls assess finite-sample effects; they do not establish unbiasedness
for informative distributions or causal relationships between regions.

## Results and timings

`results/cases/` contains 32 authoritative JSON records with estimates, individual
surrogate union values, seeds, dimensions, duplicate counts, and timings.
`estimates.csv` and `timings.csv` are derived exports; regenerate them with:

```bash
python fmri/code/run_experiment.py --output RESULTS_DIR --export-only
```

The summary script regenerates dimensions, descriptive summaries, workload timings,
and, when reference measurements are present, all-participant Thin-PID projections. Its reference exports use the
`reference_runtime/` directory within the selected results directory. The table
exporter requires all 16 completed Thin-PID measurements and generates both LaTeX
tables, numerical macros, and `eigen_participants_runtime.json`.

`source_hashes.json` records source paths and code hashes at measurement time;
these identify the measured implementation, not the current source files.
`run.json` and `hardware.json` record analysis settings and hardware provenance.
Historical configuration fields are retained as recorded.

`results/gpu/cases/` contains the 16 Torch CUDA real-data cases supporting the
paper's GPU timing claim. GPU draws can differ from CPU draws; the paper's PID
figures use the CPU records. The table exporter also derives `GPUMinutes` in
`numbers.tex` and `gpu/runtime_summary.json` when GPU records are present.

Timing uses wall-clock seconds. `raw_pid` excludes covariance fitting. `total`
covers covariance fitting, the original PID, Gaussian factorization, surrogate
generation and evaluation, and analytical correction. `case_including_assembly`
also includes target shuffling, construction of the joint response matrix, and
transfer to the selected device. GPU timing boundaries synchronize that device.
`shared_four_roi_loading` is shared across both targets and must not be counted twice.
Per-surrogate times include generation, covariance fitting, and PID. These are
recorded workload times, not isolated hardware microbenchmarks.

Plots use one point per participant and component. Colors and jitter are
reproducible; black marks show participant means and 95% percentile bootstrap
intervals (10,000 participant resamples, seed 2026), computed with Seaborn.
These resamples are independent of the PID bias-correction procedure. All four
panels share a bits axis. PDFs are 6.75 × 5 inches and use Arial.
`plotted_values.csv` and `plot_summary.csv` retain the displayed values.

## Cortical illustration and validation

```bash
python fmri/code/plot_cortex.py --data DATA_DIR --output PLOTS_DIR
python fmri/code/validate_debiasing.py --output VALIDATION_DIR
# Optional full-size validation, substantially more expensive:
python fmri/code/validate_debiasing.py --nsd --data DATA_DIR --output VALIDATION_DIR
```

The cortical script uses Nilearn's inflated fsaverage meshes and participant 1's
ROI masks. Nilearn manages the cache unless `--cache` is supplied. No FreeSurfer
installation is required. The view and lighting settings are saved with the figure.
The validation driver samples informative and independent Gaussian populations,
compares estimates against their known population PID, and saves every result.
Validation results are not distributed; choose an output directory outside the
repository. Only the runs reported in the paper are supplied.

## Optional Thin-PID and Tilde-PID benchmarks

These optional fMRI benchmarks require separate upstream checkouts. Clone the
repositories into external directories of your choice and check out the exact
commits below:

| Solver | Repository | Commit |
| --- | --- | --- |
| Thin-PID | https://github.com/warrenzha/flow-pid | `4a30b1a3ff3eb9d181fc49dff023930a617a86b3` |
| Tilde-PID | https://github.com/praveenv253/gpid | `179fd78ef426c34837e23d95c31db0293e74585d` |

These fMRI revisions differ from the bundled code under `experiments/external/`.
Keep separate checkouts at the commits above to preserve the provenance of the
supplied measurements. The fMRI code does not use the bundled synthetic solvers.

Install `fmri/benchmark_requirements.txt` in the worker Python environment.
`reference_solvers.py` loads upstream Gaussian functions; for Thin-PID it exposes
upstream whitening and pseudoinverse functions without importing neural-flow models.
The optimizer code is unchanged. Solver imports and input loading precede timing.

To time Thin-PID for all participants and targets after a fresh fMRI analysis,
use that analysis's `RESULTS_DIR`. `COVARIANCE_DIR` is an external directory for
generated inputs, and `THIN_CHECKOUT` is the pinned Thin-PID checkout. Store the
measurements under `RESULTS_DIR/reference_runtime` so the summary and table
exporters can find them:

```bash
python fmri/code/run_thin_benchmarks.py --data DATA_DIR --inputs COVARIANCE_DIR \
  --thin-repo THIN_CHECKOUT --results RESULTS_DIR --output RESULTS_DIR/reference_runtime
```

Use `--python PYTHON_EXECUTABLE` for a separate worker environment. Inputs must live
outside the repository. The queue runs serially with a six-hour limit per case,
20 CPU threads, and a one-minute heartbeat. Use the same thread count as the fMRI
analysis. Completed cases are skipped; failures and timeouts remain explicit.
The supplied measurements are in `results/reference_runtime/subjXX/`.

Once all 16 measurements are completed, regenerate the summaries and tables;
`TABLES_DIR` is your chosen output directory:

```bash
python fmri/code/summarize_results.py --results RESULTS_DIR
python fmri/code/export_numbers.py --results RESULTS_DIR --output TABLES_DIR
```

A single Thin-PID or Tilde-PID call can also be timed on a saved covariance.
Here `TILDE_CHECKOUT` is the pinned Tilde-PID checkout, `COVARIANCE_FILE` is an NPZ
input, and `BENCHMARK_DIR` is the output directory:

```bash
python fmri/code/benchmark_reference_pid.py --worker tilde --tilde-repo TILDE_CHECKOUT \
  --input COVARIANCE_FILE --target lh_hV4 --output BENCHMARK_DIR
```

NPZ inputs contain `covariance` and `dimensions`, ordered `[M, X, Y]`; they are
produced by the queue and not distributed. Direct `--worker` calls have no external
timeout; the queue enforces its limit. Running the reference script without
`--worker`, with both checkout arguments, evaluates Thin-PID then Tilde-PID for
both targets using inputs named `lh_hV4_covariance.npz` and `rh_hV4_covariance.npz`
in its output directory.

The supplied 16 completed Thin-PID measurements support the reported 41-call
projections per participant and target. These exclude surrogate generation and
covariance fitting; surrogate covariances were not benchmarked. Long benchmarks
are never part of the default analysis or test suite.
