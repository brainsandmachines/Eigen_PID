"""Compare end-to-end PID runtimes while varying one Gaussian RV dimension."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

if __package__:
    from ._common import RESULTS_DIR
    from .runtime_io import (
        _format_seed_ranges,
        _load_runtime_config,
        load_runtime_results,
        plot_runtime_results,
        save_runtime_results,
    )
else:
    from _common import RESULTS_DIR
    from runtime_io import (
        _format_seed_ranges,
        _load_runtime_config,
        load_runtime_results,
        plot_runtime_results,
        save_runtime_results,
    )

from benchmarks.adapters import run_pid_methods


CONFIG_PATH = Path(__file__).with_name("runtime_config.yaml")
METHODS = ["analytical", "thin","tilde"]  # analytical, thin, tilde


def generate_random_channels(
    dm: int, dx: int, dy: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate seeded standard-normal X and Y channel matrices."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dx, dm)), rng.standard_normal((dy, dm))


def covariance_from_channels(h_x: np.ndarray, h_y: np.ndarray) -> np.ndarray:
    """Build the identity-noise covariance represented by notebook channels."""
    sigma_m = np.eye(h_x.shape[1])
    return np.block(
        [
            [sigma_m, sigma_m @ h_x.T, sigma_m @ h_y.T],
            [
                h_x @ sigma_m,
                h_x @ sigma_m @ h_x.T + np.eye(h_x.shape[0]),
                h_x @ sigma_m @ h_y.T,
            ],
            [
                h_y @ sigma_m,
                h_y @ sigma_m @ h_x.T,
                h_y @ sigma_m @ h_y.T + np.eye(h_y.shape[0]),
            ],
        ]
    )


def main(config_path: str | Path | None = None) -> None:
    """Run or resume the runtime experiment configured by one YAML file."""
    config_path = CONFIG_PATH if config_path is None else Path(config_path)
    config = _load_runtime_config(config_path)
    sequence = config["dimensions"]
    varied_variable = config["vary"]
    dm, dx, dy = config["dm"], config["dx"], config["dy"]
    trials = int(config["trials"])
    seed_start = int(config["seed_start"])
    if varied_variable == "M":
        print(f"Fixed dimensions: X={dx}, Y={dy} and a varied M")
        experiment_directory = RESULTS_DIR / "runtime" / f"dx_{dx}_dy_{dy}"
    elif varied_variable == "X":
        print(f"Fixed dimensions: M={dm}, Y={dy} and a varied X")
        experiment_directory = RESULTS_DIR / "runtime" / f"dm_{dm}_dy_{dy}"
    else:
        print(f"Fixed dimensions: M={dm}, X={dx} and a varied Y")
        experiment_directory = RESULTS_DIR / "runtime" / f"dm_{dm}_dx_{dx}"

    fixed_dimensions = {"M": int(dm), "X": int(dx), "Y": int(dy)}
    experiment_directory.mkdir(parents=True, exist_ok=True)
    csv_path = experiment_directory / "runtime_results.csv"
    rows, existing_keys, resumed = load_runtime_results(csv_path, config)
    if resumed:
        print(f"Resuming with {len(rows)} existing results.")

    saved_config_path = experiment_directory / "inputs.yaml"
    if config_path.resolve() != saved_config_path.resolve():
        shutil.copyfile(config_path, saved_config_path)
    for dimension_index, dimension in enumerate(sequence.astype(int)):
        dimension = int(dimension)
        dimension_seed_start = seed_start + dimension_index * trials
        expected_seeds = list(
            range(dimension_seed_start, dimension_seed_start + trials)
        )
        missing_by_method = {
            method: [
                seed
                for seed in expected_seeds
                if (dimension, method, seed) not in existing_keys
            ]
            for method in METHODS
        }
        if resumed:
            for method, missing_seeds in missing_by_method.items():
                if missing_seeds:
                    print(
                        f"Seeds ({_format_seed_ranges(missing_seeds)}) are missing "
                        f"for method ({method}); evaluating them now."
                    )
        if not any(missing_by_method.values()):
            print(f"Skipping completed {varied_variable}={dimension}.")
            continue
        if not resumed:
            print(f"Running {trials} trials for {varied_variable}={dimension}...")
        current = {**fixed_dimensions, varied_variable: dimension}
        current_dm, current_dx, current_dy = (current["M"], current["X"], current["Y"])
        for trial, seed in enumerate(expected_seeds):
            missing_methods = [
                method
                for method in METHODS
                if (dimension, method, seed) not in existing_keys
            ]
            if not missing_methods:
                continue
            h_x, h_y = generate_random_channels(
                current_dm, current_dx, current_dy, seed
            )
            covariance = covariance_from_channels(h_x, h_y)
            for result in run_pid_methods(
                covariance, current_dm, current_dx, current_dy,
                methods=missing_methods,
            ):
                result_key = (dimension, str(result["method"]), seed)
                if result_key in existing_keys:
                    continue
                rows.append(
                    {
                        "trial": trial,
                        "seed": seed,
                        "varied_variable": varied_variable,
                        "dm": current_dm,
                        "dx": current_dx,
                        "dy": current_dy,
                        **result,
                    }
                )
                existing_keys.add(result_key)
            save_runtime_results(csv_path, rows)

    plot_runtime_results(csv_path)


if __name__ == "__main__":
    main()
