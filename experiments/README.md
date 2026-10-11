# Eigen-PID research experiments

This directory contains the experiments and numerical comparisons used for the
Eigen-PID research project. It is repository source and is not installed as
part of the `eigen_pid` Python package.

Before running experiments or regenerating plots, complete the
[required setup](../README.md#required-setup) for synthetic experiments in the
repository README.

The Gaussian reference solvers are included under
[`external/`](external/README.md).

## Active experiments

Run commands from this `experiments/` directory.

### Gain sweep

```bash
python -m examples.thin_pid_comparison.compare_gain_sweep
```

Writes `gain_sweep_results.csv` and `gain_sweep.pdf` under
`examples/thin_pid_comparison/results/gain_sweep/`.

### Runtime sweep

Edit `examples/thin_pid_comparison/runtime_config.yaml`, then run:

```bash
python -m examples.thin_pid_comparison.compare_runtime
```

The experiment saves every trial, the exact input configuration, and the
runtime figure in a configuration-specific results directory.

### Optimal initialization comparison

Set `vary: X` in `examples/thin_pid_comparison/runtime_config.yaml`, then run:

```bash
python -m examples.thin_pid_comparison.compare_runtime_optimal_covariance
```

The experiment initializes Thin-PID and Tilde-PID with Eigen-PID's optimal
covariance and plots their union-information differences.

## Directory map

- `benchmarks/`: shared adapters, covariance construction, and tolerances.
- `examples/thin_pid_comparison/`: active gain and runtime experiments.
- `external/flow-pid/`: Gaussian Thin-PID comparison code.
- `external/gpid/`: Gaussian Tilde-PID comparison code.
- `tests/`: checks for the experiment and comparison code.
