"""Compare optimal-covariance starts on the runtime random-channel X sweep."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from ._common import COMPONENTS, RESULTS_DIR
    from .compare_gain_sweep import (
        AXIS_LABEL_SIZE,
        LINE_WIDTH,
        MAGNITUDE_SIZE,
        MARKER_SIZE,
        PANEL_LABEL_SIZE,
        PANEL_LABEL_X,
        SUBPLOT_TITLE_SIZE,
        TICK_LABEL_SIZE,
    )
    from .compare_runtime import (
        CONFIG_PATH,
        _load_runtime_config,
        covariance_from_channels,
        generate_random_channels,
    )
else:
    from _common import COMPONENTS, RESULTS_DIR
    from compare_gain_sweep import (
        AXIS_LABEL_SIZE,
        LINE_WIDTH,
        MAGNITUDE_SIZE,
        MARKER_SIZE,
        PANEL_LABEL_SIZE,
        PANEL_LABEL_X,
        SUBPLOT_TITLE_SIZE,
        TICK_LABEL_SIZE,
    )
    from compare_runtime import (
        CONFIG_PATH,
        _load_runtime_config,
        covariance_from_channels,
        generate_random_channels,
    )

from benchmarks.adapters import run_pid_methods


FIG_WIDTH = 6.75
FIG_HEIGHT = 2.3

COMPARISONS = (
    ("thin", "o", "Eigen-PID minus Thin-PID"),
    ("tilde", "o", "Eigen-PID minus Tilde-PID"),
)
RESULT_FIELDS = tuple(component for component, _, _ in COMPONENTS)
LEGACY_COLUMNS = {
    "imxy": "mutual_information_xy",
    "union_info": "union_information",
    "obj": "objective",
    "uix": "unique_x",
    "uiy": "unique_y",
}
LEGACY_METHODS = {
    "eigen_math": "analytical",
    "zhao_thin": "thin",
    "venkatesh_tilde": "tilde",
}


def plot_optimal_covariance_differences(
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """Plot median union-information differences with IQR bands."""
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    results = pd.read_csv(csv_path)
    results = results.rename(
        columns={
            old: new
            for old, new in LEGACY_COLUMNS.items()
            if new not in results.columns
        }
    )
    required = {
        "dx",
        "trial",
        "seed",
        "method",
        "initialization",
        "union_information",
    }
    missing = sorted(required.difference(results.columns))
    if results.empty or missing:
        raise ValueError(
            "invalid optimal-covariance runtime results; "
            f"missing={missing}, empty={results.empty}"
        )
    results["method"] = results["method"].replace(LEGACY_METHODS)

    methods = [method for method, _, _ in COMPARISONS]
    selected = results[
        ((results["method"] == "analytical") & (results["initialization"] == "default"))
        | (results["method"].isin(methods) & (results["initialization"] == "optimal_cov"))
    ]
    plots = (
        (
            "union_information",
            "Union information",
            csv_path.with_name("union_information_differences.pdf")
            if output_path is None
            else Path(output_path),
        ),
    )
    for field, ylabel, figure_path in plots:
        wide = selected.pivot(
            index=["dx", "trial", "seed"],
            columns="method",
            values=field,
        )
        missing_methods = {"analytical", *methods}.difference(wide.columns)
        if missing_methods:
            raise ValueError(f"missing paired methods: {sorted(missing_methods)}")
        dimensions = np.sort(wide.index.get_level_values("dx").unique())
        if dimensions.size == 0:
            raise ValueError("no paired optimal-covariance comparisons found")
        tick_positions = np.unique(
            np.linspace(0, len(dimensions) - 1, min(6, len(dimensions)), dtype=int)
        )
        figure, axes = plt.subplots(
            1,
            len(COMPARISONS),
            figsize=(FIG_WIDTH, FIG_HEIGHT),
            sharex=True,
            sharey=True,
        )
        for panel_index, (axis, (method, marker, title)) in enumerate(
            zip(axes, COMPARISONS)
        ):
            difference = (wide["analytical"] - wide[method]).dropna()
            summaries = []
            for position, dimension in enumerate(dimensions):
                values = difference[
                    difference.index.get_level_values("dx") == dimension
                ].to_numpy()
                values = values[np.isfinite(values)]
                if values.size:
                    summaries.append(
                        (position, np.median(values), *np.percentile(values, [25, 75]))
                    )
            if not summaries:
                raise ValueError(f"no finite {field} differences for {title}")
            x_values, medians, lower, upper = map(np.asarray, zip(*summaries))
            (line,) = axis.plot(
                x_values,
                medians,
                color="black",
                marker=marker,
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label="Median",
            )
            axis.fill_between(
                x_values,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.2,
                linewidth=0,
                label="25th–75th percentiles",
            )
            axis.axhline(0, color="0.5", linewidth=LINE_WIDTH, zorder=0.5)
            axis.set_title(title, fontsize=SUBPLOT_TITLE_SIZE)
            axis.set_title(
                chr(ord("A") + panel_index),
                loc="left",
                x=PANEL_LABEL_X,
                fontsize=PANEL_LABEL_SIZE,
                fontweight="bold",
                horizontalalignment="right",
            )
            axis.set_xticks(tick_positions, dimensions[tick_positions])
            axis.set_xlabel("Source (X) dimension", fontsize=AXIS_LABEL_SIZE)
            axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axis.yaxis.get_offset_text().set_fontsize(MAGNITUDE_SIZE)
            axis.yaxis.get_offset_text().set_visible(True)
            axis.margins(x=0.03)
            axis.grid(True)
        axes[0].set_ylim(-3e-14, 3e-14)
        axes[0].set_ylabel(f"{ylabel} difference\n(bits)", fontsize=AXIS_LABEL_SIZE)
        figure.tight_layout(pad=0, w_pad=2.0)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, metadata={"CreationDate": None})
        plt.close(figure)


def main(config_path: str | Path | None = None) -> None:
    """Sweep X with runtime covariances and Eigen-initialized optimizers."""
    config_path = CONFIG_PATH if config_path is None else Path(config_path)
    config = _load_runtime_config(config_path)
    if config["vary"] != "X":
        raise ValueError("optimal-covariance runtime comparison requires vary: X")

    sequence = np.asarray(config["dimensions"], dtype=int)
    dm, dy = int(config["dm"]), int(config["dy"])
    trials = int(config["trials"])
    seed_start = int(config["seed_start"])
    output_directory = RESULTS_DIR / "runtime_optimal_covariance" / f"dm_{dm}_dy_{dy}"
    output_directory.mkdir(parents=True, exist_ok=True)
    saved_config_path = output_directory / "inputs.yaml"
    if config_path.resolve() != saved_config_path.resolve():
        shutil.copyfile(config_path, saved_config_path)

    rows: list[dict[str, object]] = []
    csv_path = output_directory / "optimal_covariance_results.csv"
    for dimension_index, dx in enumerate(sequence):
        print(f"Running {trials} optimal-covariance trials for X={dx}...")
        for trial in range(trials):
            seed = seed_start + dimension_index * trials + trial
            h_x, h_y = generate_random_channels(dm, int(dx), dy, seed)
            covariance = covariance_from_channels(h_x, h_y)
            metadata = {
                "dm": dm,
                "dx": int(dx),
                "dy": dy,
                "trial": trial,
                "seed": seed,
            }
            rows.extend(
                {**metadata, **result}
                for result in run_pid_methods(
                    covariance, dm, int(dx), dy, mode="optimal"
                )
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    plot_optimal_covariance_differences(csv_path)


if __name__ == "__main__":
    main()
