"""Export descriptive summaries and timing comparisons from completed records."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from result_io import save_json
from run_thin_benchmarks import export as export_reference_timings
from types import SimpleNamespace
from read_estimates import read_demo_estimates


def summarize(results):
    points = read_demo_estimates(results)
    identities = set(points[["subject", "target", "condition", "replicate"]].itertuples(index=False, name=None))
    cases = [json.loads((results / "cases" / f"subj{s:02d}_{t}_{c}_{r}.json").read_text())
             for s, t, c, r in sorted(identities)]
    assert len(cases) == 32
    summary = points.groupby(["target", "condition", "method", "component"]).bits.agg(["mean", "std", "min", "max"])
    summary.to_csv(results / "descriptive_summary.csv")
    dimensions, timing = [], []
    for subject in range(1, 9):
        real = [case for case in cases if case["subject"] == subject and case["condition"] == "real"]
        assert len(real) == 2
        counts = real[0]["roi_counts"]
        dimensions.append(dict(subject=subject, images=real[0]["n"], **{roi: counts[roi]["retained"] for roi in counts},
                               duplicates_removed=sum(v["duplicates"] for v in counts.values())))
    pd.DataFrame(dimensions).to_csv(results / "roi_dimensions.csv", index=False)
    for condition in ("real", "shuffled"):
        for target in ("lh", "rh"):
            selected = [case for case in cases if case["condition"] == condition and case["target"] == target]
            seconds = np.array([case["timings"]["total"] for case in selected])
            timing.append(dict(condition=condition, target=target, cases=len(selected),
                               total_seconds=float(seconds.sum()), mean_seconds=float(seconds.mean()),
                               median_seconds=float(np.median(seconds)), min_seconds=float(seconds.min()), max_seconds=float(seconds.max())))
    runtime = dict(workloads=timing, shuffled_replicate=1)
    reference = results / "reference_runtime"
    if any(reference.glob("subj*/*_hV4_thin.json")):
        export_reference_timings(SimpleNamespace(output=reference, results=results, threads=cases[0]["config"]["threads"]))
        runtime.update(reference_timings="reference_runtime/thin_participants.csv",
                       projection_assumption="Recorded evaluation count times each participant/target's measured Thin-PID runtime; excludes surrogate generation and covariance fitting.")
    save_json(results / "runtime_summary.json", runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    summarize(parser.parse_args().results)
