"""Configuration, result persistence, and plotting for runtime experiments."""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DIMENSION_COLUMNS = {"M": "dm", "X": "dx", "Y": "dy"}


def _format_seed_ranges(seeds: list[int]) -> str:
    """Format sorted seed numbers as compact consecutive ranges."""
    ranges = []
    for _, group in groupby(enumerate(seeds), lambda item: item[1] - item[0]):
        group_seeds = [seed for _, seed in group]
        start, end = group_seeds[0], group_seeds[-1]
        ranges.append(str(start) if start == end else f"{start}-{end}")
    return ", ".join(ranges)


def _load_runtime_config(config_path: str | Path) -> dict[str, object]:
    """Load and validate one path-free runtime experiment configuration."""
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("runtime config must be a YAML mapping")

    required_inputs = {
        "dimensions",
        "vary",
        "dm",
        "dx",
        "dy",
        "trials",
        "percentiles",
        "seed_start",
    }
    missing_inputs = required_inputs.difference(config)
    unknown_inputs = set(config).difference(required_inputs | {"resume"})
    if missing_inputs or unknown_inputs:
        problems = []
        if missing_inputs:
            problems.append(
                "missing inputs: "
                + ", ".join(sorted(str(name) for name in missing_inputs))
            )
        if unknown_inputs:
            problems.append(
                "unknown inputs: "
                + ", ".join(sorted(str(name) for name in unknown_inputs))
            )
        raise ValueError("runtime config " + "; ".join(problems))

    sequence = np.asarray(config["dimensions"])
    if sequence.ndim != 1 or sequence.size == 0:
        raise ValueError("dimensions must be a nonempty one-dimensional array")
    if not np.issubdtype(sequence.dtype, np.integer) or np.any(sequence < 1):
        raise ValueError("dimensions must contain positive integers")
    if np.unique(sequence).size != sequence.size:
        raise ValueError("dimensions must not contain duplicates")

    varied_variable = str(config["vary"]).upper()
    if varied_variable not in DIMENSION_COLUMNS:
        raise ValueError("vary must be one of M, X, or Y")

    dimensions = {}
    for name in DIMENSION_COLUMNS.values():
        value = config[name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
        dimensions[name] = int(value)

    trials = config["trials"]
    if (
        isinstance(trials, (bool, np.bool_))
        or not isinstance(trials, (int, np.integer))
        or trials < 2
    ):
        raise ValueError("trials must be an integer greater than one")

    seed_start = config["seed_start"]
    if (
        isinstance(seed_start, (bool, np.bool_))
        or not isinstance(seed_start, (int, np.integer))
        or seed_start < 0
    ):
        raise ValueError("seed_start must be a nonnegative integer")

    resume = config.get("resume", False)
    if not isinstance(resume, (bool, np.bool_)):
        raise ValueError("resume must be a boolean")

    percentiles = config["percentiles"]
    if (
        not isinstance(percentiles, list)
        or len(percentiles) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            for value in percentiles
        )
    ):
        raise ValueError("percentiles must contain two finite numbers")
    lower_percentile, upper_percentile = map(float, percentiles)
    if not 0 <= lower_percentile < 50 < upper_percentile <= 100:
        raise ValueError(
            "percentiles must satisfy 0 <= lower < 50 < upper <= 100"
        )

    return {
        "dimensions": sequence.astype(int),
        "vary": varied_variable,
        **dimensions,
        "trials": int(trials),
        "percentiles": (lower_percentile, upper_percentile),
        "seed_start": int(seed_start),
        "resume": bool(resume),
    }


def load_runtime_results(
    csv_path: str | Path,
    config: dict[str, object],
) -> tuple[list[dict[str, object]], set[tuple[int, str, int]], bool]:
    """Load and validate resumable rows and their dimension/method/seed keys."""
    csv_path = Path(csv_path)
    if not config["resume"] or not csv_path.is_file():
        return [], set(), False

    frame = pd.read_csv(csv_path)
    required_columns = {
        "method",
        "trial",
        "seed",
        "varied_variable",
        *DIMENSION_COLUMNS.values(),
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "cannot resume runtime CSV missing columns: "
            + ", ".join(missing_columns)
        )

    varied_variable = str(config["vary"])
    existing_varied_variables = set(
        frame["varied_variable"].dropna().astype(str).str.upper()
    )
    if existing_varied_variables != {varied_variable}:
        raise ValueError("cannot resume runtime CSV with a different varied_variable")

    sequence = np.asarray(config["dimensions"], dtype=int)
    varied_column = DIMENSION_COLUMNS[varied_variable]
    saved_dimensions = set(frame[varied_column].dropna().astype(int).unique())
    unexpected_dimensions = saved_dimensions.difference(sequence)
    if unexpected_dimensions:
        raise ValueError(
            "cannot resume after removing existing dimensions: "
            + ", ".join(map(str, sorted(unexpected_dimensions)))
        )

    for variable, column in DIMENSION_COLUMNS.items():
        if variable == varied_variable:
            continue
        saved_values = set(frame[column].dropna().astype(int).unique())
        if saved_values != {int(config[column])}:
            raise ValueError(
                f"cannot resume runtime CSV with different fixed {column}"
            )

    trials = int(config["trials"])
    trial_values = pd.to_numeric(frame["trial"], errors="coerce")
    seed_values = pd.to_numeric(frame["seed"], errors="coerce")
    if (
        trial_values.isna().any()
        or (trial_values < 0).any()
        or (trial_values >= trials).any()
        or (trial_values % 1 != 0).any()
        or seed_values.isna().any()
        or (seed_values < 0).any()
        or (seed_values % 1 != 0).any()
    ):
        raise ValueError("cannot resume runtime CSV with invalid trials or seeds")
    if frame["method"].isna().any():
        raise ValueError("cannot resume runtime CSV with missing methods")
    frame["trial"] = trial_values.astype(int)
    frame["seed"] = seed_values.astype(int)
    frame[varied_column] = frame[varied_column].astype(int)
    frame["method"] = frame["method"].astype(str)

    seed_start = int(config["seed_start"])
    dimension_indices = {
        int(dimension): index for index, dimension in enumerate(sequence)
    }
    for dimension, dimension_rows in frame.groupby(varied_column):
        expected_seeds = (
            seed_start
            + dimension_indices[int(dimension)] * trials
            + dimension_rows["trial"]
        )
        if not np.array_equal(dimension_rows["seed"], expected_seeds):
            raise ValueError(
                "cannot resume runtime CSV with seeds that do not match trials"
            )

    key_columns = [varied_column, "method", "seed"]
    if frame.duplicated(key_columns).any():
        raise ValueError("cannot resume runtime CSV with duplicate results")
    return (
        frame.to_dict(orient="records"),
        set(frame[key_columns].itertuples(index=False, name=None)),
        True,
    )


def save_runtime_results(csv_path: str | Path, rows: list[dict[str, object]]) -> None:
    """Checkpoint accumulated runtime result rows to CSV."""
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def plot_runtime_results(
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """Rebuild a runtime figure from a saved raw-trial CSV."""
    csv_path = Path(csv_path)
    config = _load_runtime_config(csv_path.with_name("inputs.yaml"))
    lower_percentile, upper_percentile = config["percentiles"]
    lower_quantile = lower_percentile / 100
    upper_quantile = upper_percentile / 100
    frame = pd.read_csv(csv_path)

    required_columns = {
        "method",
        "runtime_seconds",
        "varied_variable",
        *DIMENSION_COLUMNS.values(),
    }
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"runtime CSV is missing columns: {', '.join(missing_columns)}"
        )

    varied_variables = (
        frame["varied_variable"].dropna().astype(str).str.upper().unique()
    )
    if len(varied_variables) != 1 or varied_variables[0] not in DIMENSION_COLUMNS:
        raise ValueError(
            "runtime CSV must contain exactly one varied_variable: M, X, or Y"
        )
    varied_variable = varied_variables[0]

    for variable, column in DIMENSION_COLUMNS.items():
        if variable != varied_variable and len(frame[column].drop_duplicates()) != 1:
            raise ValueError(f"{column} must be fixed within one runtime CSV")

    output_path = (
        csv_path.with_name("runtime_scaling.pdf")
        if output_path is None
        else Path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.25, 2.45))
    method_labels = {
        "thin": "Thin-PID",
        "tilde": "Tilde-PID",
        "analytical": "Eigen-PID",
        "analytical_gpu": "Eigen-PID (GPU)",
    }
    methods_available = sorted(frame["method"].dropna().unique())
    methods_labels = {
        key: label
        for key, label in method_labels.items()
        if key in methods_available
    }

    for method, method_rows in frame.groupby("method", sort=False):
        summary = (
            method_rows.groupby(DIMENSION_COLUMNS[varied_variable], sort=False)[
                "runtime_seconds"
            ]
            .quantile([lower_quantile, 0.5, upper_quantile])
            .unstack()
        )
        (line,) = axis.plot(
            summary.index,
            summary[0.5],
            "-o",
            linewidth=1.3,
            markersize=3.5,
            label=method_labels[method],
        )
        axis.fill_between(
            summary.index,
            summary[lower_quantile],
            summary[upper_quantile],
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )

    axis.set_yscale("log")
    axis.set_xlabel(f"Source ({varied_variable}) dimension", fontsize=8)
    axis.set_ylabel("Runtime (seconds)", fontsize=8)
    axis.tick_params(axis="both", which="major", labelsize=7)
    axis.grid(True, linewidth=0.5, alpha=0.4)
    handles, labels = axis.get_legend_handles_labels()
    legend_order = methods_labels.values()
    handle_by_label = dict(zip(labels, handles))
    axis.legend(
        [handle_by_label[label] for label in legend_order],
        legend_order,
        ncol=1,
        fontsize=7,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )
    figure.tight_layout(pad=0.05)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
