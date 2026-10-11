"""Reproduce Flow-PID's cooperative-gain Gaussian sweep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
if __package__:
    from ._common import COMPONENTS, FLOW_GENERATE, RESULTS_DIR
else:
    from _common import COMPONENTS, FLOW_GENERATE, RESULTS_DIR

from benchmarks.adapters import run_pid_methods


FIG_WIDTH = 5.6
FIG_HEIGHT = 3.4

SUPTITLE_SIZE = 0
PANEL_LABEL_SIZE = 10
PANEL_LABEL_X = -0.10
SUBPLOT_TITLE_SIZE = 8
AXIS_LABEL_SIZE = 8
TICK_LABEL_SIZE = 7
MAGNITUDE_SIZE = 8
LEGEND_SIZE = 7

LINE_WIDTH = 1.0
MARKER_SIZE = 2.5
ADD_REG=True

def build_gain_covariances() -> list[tuple[int, float, np.ndarray]]:
    """Build the exact gain sequence from gain_angle_sweeps.py."""
    gains = np.linspace(0, 3, 10)
    gains[3] = 0.99  # Original code: a gain of one is unstable.
    rows = []
    for index, gain_x in enumerate(gains):
        covariance, *_ = FLOW_GENERATE.generate_cov_from_config(
            d=(2, 2, 2), gain=(gain_x, np.sqrt(2)), theta=0
        )
        rows.append((index, float(gain_x), covariance))
    return rows


def plot_gain_sweep(
    frame: pd.DataFrame,
    output_directory: Path,
    *,
    x: str = "gain",
    xlabel: str = r"Gain in $X$",
    filename: str = "gain_sweep.pdf",
) -> None:
    """Save one sweep in the compact five-panel gain-comparison layout."""


    method_panels = (
        ("analytical", "Eigen-PID"),
        ("thin", "Thin-PID"),
        ("tilde", "Tilde-PID"),
    )
    difference_panels = (
        ("thin", "Eigen minus Thin"),
        ("tilde", "Eigen minus Tilde"),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    grid = figure.add_gridspec(2, 3)
    eigen_axis = figure.add_subplot(grid[0, 0])
    thin_axis = figure.add_subplot(grid[0, 1], sharey=eigen_axis)
    tilde_axis = figure.add_subplot(grid[0, 2], sharey=eigen_axis)
    value_axes = (eigen_axis, thin_axis, tilde_axis)
    thin_difference_axis = figure.add_subplot(grid[1, 1], sharex=thin_axis)
    tilde_difference_axis = figure.add_subplot(
        grid[1, 2],
        sharex=tilde_axis,
    )
    difference_axes = (thin_difference_axis, tilde_difference_axis)
    legend_axis = figure.add_subplot(grid[1, 0])
    legend_axis.axis("off")

    for axis, (method, panel_title) in zip(value_axes, method_panels):
        rows = frame[frame["method"] == method].sort_values(x)
        for component, color, _ in COMPONENTS:
            axis.plot(
                rows[x],
                rows[component],
                color=color,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
            )
        axis.set_title(panel_title, fontsize=SUBPLOT_TITLE_SIZE)
    value_axes[0].set_title(
        "A",
        loc="left",
        x=PANEL_LABEL_X,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        horizontalalignment="right",
    )
    value_axes[0].set_ylabel("PID value\n(bits)", fontsize=AXIS_LABEL_SIZE)

    eigen = frame[frame["method"] == "analytical"].sort_values(x)
    for axis, (method, panel_title) in zip(difference_axes, difference_panels):
        other = frame[frame["method"] == method].sort_values(x)
        for component, color, _ in COMPONENTS:
            paired = eigen[[x, component]].merge(
                other[[x, component]],
                on=x,
                suffixes=("_eigen", "_other"),
            )
            axis.plot(
                paired[x],
                paired[f"{component}_eigen"] - paired[f"{component}_other"],
                color=color,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
            )
        axis.axhline(0, color="0.5", linewidth=LINE_WIDTH, zorder=0.1)
        axis.set_title(panel_title, fontsize=SUBPLOT_TITLE_SIZE)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    difference_axes[0].set_title(
        "B",
        loc="left",
        x=PANEL_LABEL_X,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        horizontalalignment="right",
    )
    difference_axes[0].set_ylabel(
        "Difference\n(bits)",
        fontsize=AXIS_LABEL_SIZE,
    )

    for axis in (*value_axes, *difference_axes):
        axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
        axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        axis.yaxis.get_offset_text().set_fontsize(MAGNITUDE_SIZE)
        axis.margins(y=0.05)
        axis.grid(True)
    for axis in difference_axes:
        axis.yaxis.get_offset_text().set_visible(True)

    legend_axis.legend(
        [
            Line2D(
                [],
                [],
                color=color,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
            )
            for _, color, _ in COMPONENTS
        ],
        [label for _, _, label in COMPONENTS],
        loc="center",
        ncol=1,
        frameon=False,
        title="PID component",
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_SIZE,
    )
    figure.tight_layout(
        rect=(0, 0, 1, 1),
        pad=0,
        h_pad=1.5,
        w_pad=0.1,
    )
    figure.savefig(output_directory / filename, metadata={"CreationDate": None})
    plt.close(figure)


def main() -> None:
    rows: list[dict[str, float | int | str]] = []
    experiment_directory = RESULTS_DIR / "gain_sweep"
    experiment_directory.mkdir(parents=True, exist_ok=True)
    for index, gain, covariance in build_gain_covariances():
        for result in run_pid_methods(covariance, 2, 2, 2, add_reg=True):
            rows.append({"id": index, "gain": gain, **result})

    frame = pd.DataFrame(rows)
    frame.to_csv(experiment_directory / "gain_sweep_results.csv", index=False)
    plot_gain_sweep(frame, experiment_directory)


if __name__ == "__main__":
    main()
