# Eigen-PID comparison experiments

This directory contains the three active research workflows:

| Script | Output |
| --- | --- |
| `compare_gain_sweep.py` | PID comparison CSV and five-panel gain figure |
| `compare_runtime.py` | Runtime CSV and scaling figure |
| `compare_runtime_optimal_covariance.py` | Optimal-covariance comparison CSV and union-information difference figure |

Install the repository's experiment dependencies first, then run the commands
from `experiments/`:

```bash
python -m examples.thin_pid_comparison.compare_gain_sweep
python -m examples.thin_pid_comparison.compare_runtime
python -m examples.thin_pid_comparison.compare_runtime_optimal_covariance
```

The gain sweep needs no configuration. Both runtime scripts read
`runtime_config.yaml`; the optimal-covariance runtime experiment requires
`vary: X`.

Generated CSV files, copied configurations, and PDF figures are written below
`results/`. See the [repository README](../../../README.md#reproduce-the-research-experiments)
for the full setup and output guide.
