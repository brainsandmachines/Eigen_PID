"""Shared fixtures and plotting for the Flow-PID Gaussian reproductions."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

# IDEs commonly execute an example by file path, without its package context.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.adapters import ZHAO_ROOT, _load_file_module, load_zhao


RESULTS_DIR = Path(__file__).resolve().parent / "results"
FONT_FAMILY = "Arial"
for font_path in (Path.home() / ".local/share/fonts/msttcorefonts").glob(
    "Arial*.TTF"
):
    font_manager.fontManager.addfont(font_path)
mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "mathtext.cal": f"{FONT_FAMILY}:italic",
        "mathtext.sf": FONT_FAMILY,
        "mathtext.tt": FONT_FAMILY,
    }
)

# Load only the official Gaussian code. Importing Flow-PID's package initializer
# would also import its unrelated neural dependencies (including PyTorch).
load_zhao()
sys.modules["utils"].solve = sys.modules["_zhao_gaussian_estimate_channel"].solve
FLOW_GENERATE = _load_file_module(
    "_zhao_gaussian_generate", ZHAO_ROOT / "utils" / "generate.py"
)
FLOW_WHITEN = sys.modules["utils"].whiten

COMPONENTS = (
    ("mutual_information_xy", "black", r"$I(M; (X, Y))$"),
    ("unique_x", "C0", r"$U_X$"),
    ("unique_y", "C1", r"$U_Y$"),
    ("redundancy", "C2", r"$R$"),
    ("synergy", "C3", r"$S$"),
)
ATOM_COMPONENTS = tuple(
    item for item in COMPONENTS if item[0] != "mutual_information_xy"
)
EIGEN_LINESTYLE = (0, (9, 2, 1, 2))
METHOD_STYLES = (
    ("thin", "-", "", "Thin-PID"),
    ("tilde", "--", "", "Tilde-PID"),
    ("analytical", EIGEN_LINESTYLE, "x", "Eigen-PID"),
)
METHOD_ALPHA = 0.65
LEGEND_ANCHOR_X = 1.02


def _add_component_and_method_legends(axis, method_handles, method_labels) -> None:
    """Add separate component-color and PID-definition legends, close to the axis."""
    from matplotlib.lines import Line2D

    component_legend = axis.legend(
        [Line2D([], [], color=color) for _, color, _ in COMPONENTS],
        [label for _, _, label in COMPONENTS],
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(LEGEND_ANCHOR_X, 1),
        title="PID component",
        alignment="left",
    )
    axis.add_artist(component_legend)
    axis.legend(
        method_handles,
        method_labels,
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(LEGEND_ANCHOR_X, 0.35),
        title="PID definition",
        alignment="left",
    )


def plot_sweep(
    frame,
    x: str,
    xlabel: str,
    title: str,
    output_stem: str,
    output_directory: Path | None = None,
) -> None:
    """Save one combined PID-value plot and one combined error plot for a sweep.

    All PID components share a single axis, color-coded by component; the
    three PID definitions (Thin/Tilde/Eigen) are distinguished by linestyle.
    Optionally place the plots inside the experiment directory.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_directory = RESULTS_DIR if output_directory is None else output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    for method, linestyle, marker, _ in METHOD_STYLES:
        rows = frame[frame["method"] == method].sort_values(x)
        for component, color, _ in COMPONENTS:
            axis.plot(
                rows[x],
                rows[component],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                alpha=METHOD_ALPHA,
            )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Partial information (bits)")
    axis.set_title(title)
    axis.grid(True)
    method_handles = [
        Line2D(
            [],
            [],
            color="black",
            linestyle=linestyle,
            marker=marker,
            alpha=METHOD_ALPHA,
        )
        for _, linestyle, marker, _ in METHOD_STYLES
    ]
    method_labels = [label for _, _, _, label in METHOD_STYLES]
    _add_component_and_method_legends(axis, method_handles, method_labels)
    figure.tight_layout(rect=(0, 0, 0.82, 1))
    figure.savefig(output_directory / f"{output_stem}.png", dpi=160)
    plt.close(figure)

    error_methods = tuple(style for style in METHOD_STYLES if style[0] != "analytical")
    error_figure, error_axis = plt.subplots(figsize=(8, 5.5))
    for method, linestyle, _, _ in error_methods:
        rows = frame[frame["method"] == method].sort_values(x)
        for atom, color, _ in ATOM_COMPONENTS:
            column = f"{atom}_absolute_error_vs_analytical"
            error_axis.plot(rows[x], rows[column], color=color, linestyle=linestyle)
    error_axis.set_xlabel(xlabel)
    error_axis.set_ylabel("Absolute error vs. Eigen-PID (bits)")
    error_axis.set_title(f"{title} errors")
    error_axis.grid(True)
    error_component_legend = error_axis.legend(
        [Line2D([], [], color=color) for _, color, _ in ATOM_COMPONENTS],
        [label for _, _, label in ATOM_COMPONENTS],
        loc="upper left",
        ncol=1,
        frameon=False,
        bbox_to_anchor=(LEGEND_ANCHOR_X, 1),
        title="PID component",
        alignment="left",
    )
    error_axis.add_artist(error_component_legend)
    error_axis.legend(
        [
            Line2D([], [], color="black", linestyle=linestyle)
            for _, linestyle, _, _ in error_methods
        ],
        [label for _, _, _, label in error_methods],
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(LEGEND_ANCHOR_X, 0.35),
        title="PID definition",
        alignment="left",
    )
    error_figure.tight_layout(rect=(0, 0, 0.82, 1))
    error_figure.savefig(output_directory / f"{output_stem}_error.png", dpi=160)
    plt.close(error_figure)
