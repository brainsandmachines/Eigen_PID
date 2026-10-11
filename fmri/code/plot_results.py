"""Draw the two paper PID figures exclusively from saved estimates."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from read_estimates import read_demo_estimates


ATOMS = ["uix", "uiy", "red", "syn"]
LABELS = ["Unique\nleft V1", "Unique\nright V1", "Redundancy", "Synergy"]
METHODS = ("raw", "combined")


def figure_style():
    font_manager.findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "mathtext.fontset": "custom", "mathtext.bf": "Arial:bold",
                         "axes.spines.top": False, "axes.spines.right": False})


def plot(results, output):
    data = read_demo_estimates(results)
    keys = ["method", "target", "condition", "subject", "component"]
    points = data[data.component.isin(ATOMS)].sort_values(keys)
    output.mkdir(parents=True, exist_ok=True)
    points.to_csv(results / "plotted_values.csv", index=False)
    points.groupby(keys[:3] + ["component"]).bits.agg(["mean", "std", "min", "max"]).to_csv(results / "plot_summary.csv")
    sns.set_theme(style="whitegrid", context="paper")
    figure_style()
    # Match the main paper's Arial typography at the final 6.75-inch width.
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.edgecolor": "black", "axes.linewidth": 0.8,
                         "axes.spines.top": True, "axes.spines.right": True,
                         "xtick.color": "black", "ytick.color": "black"})
    colors = sns.color_palette("tab10", 8)
    palette = dict(zip(range(1, 9), colors))
    for method in METHODS:
        selected = points[points.method == method]
        minimum, maximum = min(0, selected.bits.min()), max(0, selected.bits.max())
        span = maximum - minimum
        assert span > 0
        fig, axes = plt.subplots(2, 2, figsize=(6.75, 5), sharey=True, sharex=True)
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.92, hspace=0.39, wspace=0.17)
        for row, condition in enumerate(("real", "shuffled")):
            for column, target in enumerate(("lh", "rh")):
                ax = axes[row, column]
                cell = selected[(selected.target == target) & (selected.condition == condition)]
                table = cell.pivot(index="subject", columns="component", values="bits").reindex(index=range(1, 9), columns=ATOMS)
                assert table.notna().all().all()
                # Seaborn's strip jitter uses NumPy's global RNG; fix it for each panel.
                np.random.seed(2026)
                sns.stripplot(data=cell, x="component", y="bits", order=ATOMS,
                              hue="subject", hue_order=range(1, 9), palette=palette,
                              jitter=0.16, size=4.5, linewidth=0, legend=False, ax=ax, zorder=3)
                # There is exactly one value per participant and atom in both real and shuffled data.
                sns.pointplot(data=cell, x="component", y="bits", order=ATOMS,
                              estimator="mean", errorbar=("ci", 95), n_boot=10000, seed=2026,
                              color="black", markers="_", markersize=10, markeredgewidth=1.5,
                              linestyles="none", capsize=0.12, err_kws={"linewidth": 1.1},
                              ax=ax, zorder=4)
                ax.axhline(0, color="0.35", linewidth=0.6, zorder=1)
                ax.yaxis.grid(True, color="0.85", linewidth=0.5)
                ax.set_axisbelow(True)
                ax.set(xlabel="", ylabel="", xlim=(-0.5, 3.5), ylim=(minimum - 0.05 * span, maximum + 0.08 * span), xticks=range(4), xticklabels=LABELS)
                ax.tick_params(axis="x", length=0, pad=6)
                letter = "ABCD"[2 * row + column]
                side = "left" if target == "lh" else "right"
                condition_label = "real data" if condition == "real" else "shuffled observations"
                ax.set_title(f"Target: {side} hV4 ({condition_label})", loc="left", pad=8)
                ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                            xytext=(-17, 8), textcoords="offset points",
                            fontsize=10, fontweight="bold", ha="left", va="baseline")
                if column == 0:
                    ax.set_ylabel("Information (bits)")
        handles = [Line2D([], [], marker="o", color=color, linestyle="none", markersize=4, label=f"S{i + 1}") for i, color in enumerate(colors)]
        handles.append(Line2D([], [], marker="_", color="black", linestyle="none", markersize=9, label="Mean (95% CI)"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=9, frameon=False,
                   columnspacing=0.7, handletextpad=0.35)
        fig.savefig(output / f"pid_{method}.pdf")
        fig.savefig(output / f"pid_{method}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--results", type=Path, default=base / "results")
    parser.add_argument("--output", type=Path, default=base / "plots")
    args = parser.parse_args()
    plot(args.results, args.output)
