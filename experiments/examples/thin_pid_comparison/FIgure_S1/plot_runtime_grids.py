"""Combine the saved runtime experiments into one X- and M-sweep grid."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


RESULTS_DIR = Path(__file__).resolve().parent
DIMENSION_COLUMNS = {"M": "dm", "X": "dx", "Y": "dy"}
METHOD_LABELS = {
    "thin": "Thin-PID",
    "analytical": "Eigen-PID",
}


@plt.rc_context(
    {
        "font.family": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    }
)
def _plot_grid(
    grouped_experiments: dict[
        str, list[tuple[Path, dict[str, object]]]
    ],
) -> Path:
    """Plot X sweeps above M sweeps in one labeled runtime grid."""
    column_count = max(map(len, grouped_experiments.values()))
    figure, axes = plt.subplots(
        len(grouped_experiments),
        column_count,
        figsize=(6.75, 3.67),
        squeeze=False,
    )
    legend_handles = {}

    for row, (varied_variable, experiments) in enumerate(
        grouped_experiments.items()
    ):
        for column, (experiment_dir, config) in enumerate(experiments):
            axis = axes[row, column]
            frame = pd.read_csv(experiment_dir / "runtime_results.csv")
            required_columns = {
                "method",
                "runtime_seconds",
                "varied_variable",
                *DIMENSION_COLUMNS.values(),
            }
            missing_columns = sorted(
                required_columns.difference(frame.columns)
            )
            if missing_columns:
                raise ValueError(
                    f"{experiment_dir.name}: runtime CSV is missing columns: "
                    + ", ".join(missing_columns)
                )

            csv_varied_variables = set(
                frame["varied_variable"].dropna().astype(str).str.upper()
            )
            if csv_varied_variables != {varied_variable}:
                raise ValueError(
                    f"{experiment_dir.name}: YAML and CSV varied variables "
                    "disagree"
                )

            fixed_variables = [
                variable
                for variable in DIMENSION_COLUMNS
                if variable != varied_variable
            ]
            for variable in fixed_variables:
                dimension_column = DIMENSION_COLUMNS[variable]
                expected = int(config[dimension_column])
                if set(frame[dimension_column].dropna().astype(int)) != {
                    expected
                }:
                    raise ValueError(
                        f"{experiment_dir.name}: YAML and CSV fixed "
                        f"{variable} dimensions disagree"
                    )

            lower_percentile, upper_percentile = map(
                float, config["percentiles"]
            )
            lower_quantile = lower_percentile / 100
            upper_quantile = upper_percentile / 100

            for method, method_rows in frame.groupby("method", sort=False):
                if method not in METHOD_LABELS:
                    continue
                summary = (
                    method_rows.groupby(
                        DIMENSION_COLUMNS[varied_variable], sort=False
                    )["runtime_seconds"]
                    .quantile([lower_quantile, 0.5, upper_quantile])
                    .unstack()
                )
                (line,) = axis.plot(
                    summary.index,
                    summary[0.5],
                    "-o",
                    linewidth=1.3,
                    markersize=3.5,
                    label=METHOD_LABELS[method],
                )
                legend_handles.setdefault(method, line)
                axis.fill_between(
                    summary.index,
                    summary[lower_quantile],
                    summary[upper_quantile],
                    color=line.get_color(),
                    alpha=0.2,
                    linewidth=0,
                )

            title = ", ".join(
                rf"$d_{{{variable}}}="
                rf"{int(config[DIMENSION_COLUMNS[variable]])}$"
                for variable in fixed_variables
            )
            axis.set_title(title, fontsize=8, pad=4)
            axis.locator_params(axis="x", nbins=6)
            axis.set_yscale("log")
            axis.set_xlabel(
                f"{'Target' if varied_variable == 'M' else 'Source'} "
                f"({varied_variable}) dimension",
                fontsize=8,
            )
            if column == 0:
                axis.set_ylabel("Runtime (seconds)", fontsize=8)
            axis.tick_params(axis="both", which="major", labelsize=7)
            axis.tick_params(axis="x", labelrotation=90)
            axis.grid(True, linewidth=0.5, alpha=0.4)

        for axis in axes[row, len(experiments) :]:
            axis.set_visible(False)
        axes[row, 0].text(
            -0.12,
            1.20,
            chr(ord("A") + row),
            transform=axes[row, 0].transAxes,
            fontsize=10,
            fontfamily="Arial",
            fontweight="bold",
            ha="right",
            va="top",
        )

    missing_methods = METHOD_LABELS.keys() - legend_handles.keys()
    if missing_methods:
        raise ValueError(
            "runtime grids are missing methods: "
            + ", ".join(sorted(missing_methods))
        )

    figure.legend(
        [legend_handles[method] for method in METHOD_LABELS],
        METHOD_LABELS.values(),
        loc="upper left",
        bbox_to_anchor=(0.005, 0.995),
        ncol=1,
        fontsize=7,
        frameon=True,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )

    output_path = RESULTS_DIR / "runtime_grid.pdf"
    figure.tight_layout(
        pad=0.4,
        w_pad=0.5,
        h_pad=0.6,
        rect=(0, 0, 0.98, 0.9),
    )
    figure.savefig(output_path, dpi=300, metadata={"CreationDate": None})
    plt.close(figure)
    return output_path


def main() -> None:
    """Discover the saved experiments and write one combined runtime grid."""
    grouped_experiments: dict[
        str, list[tuple[Path, dict[str, object]]]
    ] = {"X": [], "M": []}

    for config_path in RESULTS_DIR.glob("*/inputs.yaml"):
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError(f"{config_path}: inputs must be a YAML mapping")

        varied_variable = str(config.get("vary", "")).upper()
        if varied_variable in grouped_experiments:
            grouped_experiments[varied_variable].append(
                (config_path.parent, config)
            )

    for varied_variable, experiments in grouped_experiments.items():
        if not experiments:
            raise ValueError(f"no {varied_variable}-sweep experiments found")
        fixed_variables = [
            variable
            for variable in DIMENSION_COLUMNS
            if variable != varied_variable
        ]
        experiments.sort(
            key=lambda experiment: tuple(
                int(experiment[1][DIMENSION_COLUMNS[variable]])
                for variable in fixed_variables
            )
        )
    output_path = _plot_grid(grouped_experiments)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
