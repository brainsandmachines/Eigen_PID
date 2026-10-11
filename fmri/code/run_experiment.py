"""Run the eight-participant demonstration; persist each case before continuing.

Example: python fmri/code/run_experiment.py --data DATA_DIR
The JSON case records are authoritative. CSV exports can be rebuilt at any time.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch

from eigen_pid import debias_eigen_pid
from nsd_reader import load_region
from result_io import save_json


METHODS = ("raw", "combined", "permutation")


def saved_config(path, config):
    """Read current configuration fields from historical records."""
    saved = {"disable_numerical_checks": False, **json.loads(path.read_text())["config"]}
    return {key: saved.get(key) for key in config}


def export_csv(output):
    records = [json.loads(path.read_text()) for path in sorted((output / "cases").glob("*.json"))]
    assert all((r["condition"], r["replicate"]) in (("real", 0), ("shuffled", 1)) for r in records)
    columns = ["subject", "target", "condition", "replicate", "method", "component", "bits"]
    with (output / "estimates.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            identity = {key: record[key] for key in columns[:4]}
            for method in METHODS:
                for component, value in record["estimates"][method].items():
                    writer.writerow(dict(identity, method=method, component=component, bits=value))
    with (output / "timings.csv").open("w", newline="") as file:
        columns = ["subject", "target", "condition", "replicate", "stage", "seconds"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for record in records:
            for stage, value in record["timings"].items():
                writer.writerow({**{key: record[key] for key in columns[:4]}, "stage": stage, "seconds": value})
    return records


def run(args):
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases").mkdir(exist_ok=True)
    if args.export_only:
        print(f"Exported {len(export_csv(args.output))} saved cases.")
        return
    assert args.data is not None, "Supply --data (unless using --export-only)"
    device = torch.device(args.device)
    config = dict(covariance=args.covariance, n_shuffles=args.shuffles, n_resamples=args.resamples,
                  seed=args.seed, threads=args.threads,
                  dtype="float64", device=str(device),
                  disable_numerical_checks=args.disable_numerical_checks,
                  sources=["lh:V1", "rh:V1"], schema=2)
    manifest = args.output / "run.json"
    if manifest.exists():
        assert saved_config(manifest, config) == config, "Saved run uses different settings; choose another output folder"
    else:
        save_json(manifest, dict(config=config, started_utc=datetime.now(timezone.utc).isoformat(),
                                torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                                platform=platform.platform(), torch_config=torch.__config__.show()))
    # One real analysis and one shuffled control per participant and target.
    # The condition index also fixes the random seed and the saved case identity.
    for replicate, condition in enumerate(("real", "shuffled")):
        if condition not in args.conditions:
            continue
        for subject in args.subjects:
            paths = {target: args.output / "cases" / f"subj{subject:02d}_{target}_{condition}_{replicate}.json" for target in ("lh", "rh")}
            if all(path.exists() for path in paths.values()):
                for path in paths.values():
                    assert saved_config(path, config) == config
                continue
            load_start = perf_counter()
            responses, counts = {}, {}
            for region in ("lh:V1", "rh:V1", "lh:hV4", "rh:hV4"):
                responses[region], counts[region] = load_region(args.data / f"subj{subject:02d}", region)
            load_seconds = perf_counter() - load_start
            x, y = responses["lh:V1"], responses["rh:V1"]
            for target, path in paths.items():
                if path.exists():
                    assert saved_config(path, config) == config
                    continue
                case_start = perf_counter()
                m = responses[f"{target}:hV4"]
                assert len(m) == len(x) == len(y)
                seed = args.seed + subject * 10000 + (target == "rh") * 1000 + replicate
                if replicate:
                    m = m[np.random.default_rng(seed).permutation(len(m))]
                data = torch.from_numpy(np.concatenate((m, x, y), axis=1)).to(device)
                dimensions = [m.shape[1], x.shape[1], y.shape[1]]
                print(f"START subj{subject:02d} {target}:hV4 {condition} {replicate}: n={len(m)}, dimensions={dimensions}", flush=True)
                estimate = debias_eigen_pid(data, *dimensions, covariance=args.covariance,
                                            n_shuffles=args.shuffles, n_resamples=args.resamples,
                                            seed=seed + 1000000, progress=args.progress,
                                            disable_numerical_checks=args.disable_numerical_checks)
                timing = {**estimate.pop("timings"), "case_including_assembly": perf_counter() - case_start,
                          "shared_four_roi_loading": load_seconds}
                record = dict(subject=subject, target=target, condition=condition, replicate=replicate,
                              n=len(m), dimensions=dimensions, roi_counts=counts, seed=seed, resampling_seed=seed + 1000000,
                              config=config, estimates=estimate, timings=timing, completed_utc=datetime.now(timezone.utc).isoformat())
                save_json(path, record)
                export_csv(args.output)
                print(f"DONE {path.name}: {timing['total']:.2f}s; combined={estimate['combined']}", flush=True)
    records = export_csv(args.output)
    expected = len(args.subjects) * 2 * len(args.conditions)
    selected = [r for r in records if r["subject"] in args.subjects and r["condition"] in args.conditions]
    assert len(selected) == expected, (len(selected), expected)
    save_json(args.output / "completion.json", dict(
        cases=len(records),
        pid_evaluations=sum(1 + r["config"]["n_shuffles"] + r["config"]["n_resamples"] for r in records),
        estimation_seconds=sum(r["timings"]["total"] for r in records),
        completed_utc=max(r["completed_utc"] for r in records)))
    print(f"COMPLETE: {len(records)} cases", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Algonauts training-data directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 9)), choices=range(1, 9))
    parser.add_argument("--conditions", nargs="+", choices=("real", "shuffled"),
                        default=["real", "shuffled"], help="Analyze real data, shuffled controls, or both")
    parser.add_argument("--covariance", choices=("empirical", "ledoit-wolf"), default="empirical")
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0")
    parser.add_argument("--disable-numerical-checks", action="store_true",
                        help="Skip numerical diagnostics; assumes numerically suitable covariances")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    run(parser.parse_args())
