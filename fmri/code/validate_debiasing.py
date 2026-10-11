"""Evaluate corrections on Gaussian populations with known PID.

The default is a small reproducible check. --nsd uses subject 1's right-hV4
dimensions and fitted covariance for the computational regime of the demo.
Population truth is evaluated directly from the generating covariance.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from eigen_pid import debias_eigen_pid
from eigen_pid.debias_eigen_pid import estimate_covariance, pid_from_covariance
from nsd_reader import load_region
from result_io import save_json


def evaluate(args):
    torch.set_num_threads(args.threads)
    rng = torch.Generator().manual_seed(args.seed)
    if args.nsd:
        assert args.data is not None, "--nsd requires --data"
        regions = [load_region(args.data / "subj01", roi)[0] for roi in ("rh:hV4", "lh:V1", "rh:V1")]
        dimensions = [region.shape[1] for region in regions]
        n = len(regions[0])
        fitted = estimate_covariance(torch.from_numpy(np.concatenate(regions, axis=1)))
    else:
        dimensions, n = [2, 3, 1], 100
        a = torch.randn((6, 6), dtype=torch.float64, generator=rng)
        fitted = a @ a.T + torch.eye(6)
    null = fitted.clone()
    null[:dimensions[0], dimensions[0]:] = 0
    null[dimensions[0]:, :dimensions[0]] = 0
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(dimensions=dimensions, n=n, replicates=args.replicates, shuffles=args.shuffles,
                    resamples=args.resamples, seed=args.seed, nsd=args.nsd, torch=torch.__version__)
    save_json(args.output / "config.json", metadata)
    rows = []
    for model, covariance in (("informative", fitted), ("null", null)):
        truth = pid_from_covariance(covariance, *dimensions)
        if model == "null":
            assert max(abs(value) for value in truth.values()) < 1e-8
        factor = torch.linalg.cholesky(covariance)
        for replicate in range(args.replicates):
            sample = torch.randn((n, sum(dimensions)), dtype=torch.float64, generator=rng) @ factor.T
            estimate = debias_eigen_pid(sample, *dimensions, n_shuffles=args.shuffles,
                                        n_resamples=args.resamples, seed=args.seed + replicate)
            save_json(args.output / f"{model}_{replicate}.json", dict(truth=truth, estimate=estimate))
            for method in ("raw", "combined", "permutation"):
                for component, value in estimate[method].items():
                    rows.append(dict(model=model, replicate=replicate, method=method, component=component,
                                     truth=truth[component], estimate=value, error=value - truth[component]))
            print(f"{model} {replicate + 1}/{args.replicates}: {estimate['timings']['total']:.2f}s", flush=True)
    with (args.output / "estimates.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsd", action="store_true")
    parser.add_argument("--data", type=Path, help="Algonauts training-data directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--resamples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    evaluate(parser.parse_args())
