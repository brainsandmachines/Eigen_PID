"""Time unchanged CPU Tilde/Thin-PID solvers on saved NSD covariances.

Loads pinned upstream repositories from explicit checkout paths.
Solver imports and input loading are not timed.
Each solver/target runs in its own process; timeouts give only a lower bound.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter, time

import numpy as np

from result_io import save_json as save_result


def worker(args):
    from reference_solvers import COMMITS, load_solver
    from threadpoolctl import threadpool_info, threadpool_limits
    import scipy

    checkout = args.tilde_repo if args.worker == "tilde" else args.thin_repo
    assert checkout is not None, "Supply the selected solver checkout"
    solver, adapter = load_solver(args.worker, checkout)
    input_path = args.input or args.output / f"{args.target}_covariance.npz"
    case = np.load(input_path)
    covariance = case["covariance"]
    dm, dx, dy = map(int, case["dimensions"])
    assert covariance.shape == (dm + dx + dy,) * 2
    assert covariance.dtype == np.float64 and np.isfinite(covariance).all()
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    with threadpool_limits(limits=args.threads):
        result = dict(solver=args.worker, subject=args.subject, target=args.target, dimensions=[dm, dx, dy],
                      input_sha256=input_sha256,
                      dtype=str(covariance.dtype), threads=args.threads, adapter=adapter,
                      commit=COMMITS[args.worker],
                      numpy=np.__version__, scipy=scipy.__version__, libraries=[{k: v for k, v in lib.items() if k != "filepath"}
                                 for lib in threadpool_info()],
                      status="running", started_at=time(), heartbeat_at=time())
        path = args.output / f"{args.target}_{args.worker}.json"
        start = perf_counter()
        result.update(elapsed_seconds=0.)
        save_result(path, result)
        stop = threading.Event()
        def checkpoint():
            while not stop.wait(60):
                result["elapsed_seconds"] = perf_counter() - start
                result["heartbeat_at"] = time()
                save_result(path, result)
        heartbeat = threading.Thread(target=checkpoint, daemon=True)
        heartbeat.start()
        try:
            values = solver(covariance, dm, dx, dy, unbiased=False)
            elapsed = perf_counter() - start
        finally:
            stop.set()
            heartbeat.join()
        keys = ("mix", "miy", "mixy", "union_info", "objective", "uix", "uiy", "red", "syn")
        assert len(values) == len(keys) and np.isfinite(values).all()
        result.pop("heartbeat_at", None)
        result.update(status="completed", seconds=elapsed, elapsed_seconds=elapsed, finished_at=time(),
                      values=dict(zip(keys, map(float, values))))
        save_result(path, result)
        print(f"{args.target} {args.worker}: {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thin-repo", type=Path, help="External flow-pid checkout")
    parser.add_argument("--tilde-repo", type=Path, help="External gpid checkout")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "results/reference_runtime")
    parser.add_argument("--input", type=Path, help="Explicit local covariance NPZ")
    parser.add_argument("--subject", type=int)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=21600)
    parser.add_argument("--worker", choices=("tilde", "thin"))
    parser.add_argument("--target", choices=("lh_hV4", "rh_hV4"))
    args = parser.parse_args()
    assert args.threads > 0 and args.timeout > 0
    if args.worker:
        assert args.target is not None, "Supply --target"
        args.output.mkdir(parents=True, exist_ok=True)
        worker(args)
    else:
        assert args.thin_repo is not None and args.tilde_repo is not None
        args.output.mkdir(parents=True, exist_ok=True)
        for solver in ("thin", "tilde"):
            for target in ("lh_hV4", "rh_hV4"):
                save_result(args.output / f"{target}_{solver}.json",
                            dict(solver=solver, target=target, status="queued", timeout_seconds=args.timeout))
        for solver in ("thin", "tilde"):
            for target in ("lh_hV4", "rh_hV4"):
                print(f"Starting {target} {solver}; limit {args.timeout:g}s", flush=True)
                command = [sys.executable, __file__, "--worker", solver, "--target", target,
                           "--thin-repo", str(args.thin_repo), "--tilde-repo", str(args.tilde_repo),
                           "--output", str(args.output),
                           "--threads", str(args.threads)]
                case_start = perf_counter()
                with (args.output / f"{target}_{solver}.log").open("w") as log:
                    try:
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                   timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        path = args.output / f"{target}_{solver}.json"
                        result = json.loads(path.read_text())
                        result.update(status="timeout", timeout_seconds=args.timeout, finished_at=time())
                        result["elapsed_seconds"] = perf_counter() - case_start
                        save_result(path, result)
                        print(f"{target} {solver}: timeout", flush=True)
                    else:
                        path = args.output / f"{target}_{solver}.json"
                        result = json.loads(path.read_text())
                        if completed.returncode != 0:
                            result.update(status="failed", returncode=completed.returncode, finished_at=time())
                            result["elapsed_seconds"] = perf_counter() - case_start
                            save_result(path, result)
                        print(path.read_text(), flush=True)
