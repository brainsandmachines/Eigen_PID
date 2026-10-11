"""Resume a sequential Thin-PID queue; save heartbeats, results, and projections.

Workers may use a separate Python environment with the reference dependencies.
Covariance inputs are written to a user-supplied directory outside the repository.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import time
import traceback
import sys

import numpy as np
import torch

from result_io import save_json as save_result
from reference_solvers import COMMITS
from eigen_pid.debias_eigen_pid import estimate_covariance, pid_from_covariance
from nsd_reader import load_region


ATOMS = ("uix", "uiy", "red", "syn")
THIN_COMMIT = COMMITS["thin"]


def read(path):
    return json.loads(path.read_text())


def prepare(args, subject, target, real):
    """Reproduce the analysis covariance and verify Eigen-PID before timing Thin."""
    folder = args.inputs / f"subj{subject:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{target}_hV4_covariance.npz"
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists():
        info = read(metadata)
        assert info['subject'] == subject and info['target'] == target
        assert info['dimensions'] == real['dimensions'] and info['n'] == real['n']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
        for key in ATOMS:
            np.testing.assert_allclose(info['eigen_values'][key], real['estimates']['raw'][key], atol=1e-8, rtol=0)
        return path, info
    arrays = [load_region(args.data / f"subj{subject:02d}", roi)[0]
              for roi in (f"{target}:hV4", "lh:V1", "rh:V1")]
    dimensions = [a.shape[1] for a in arrays]
    assert dimensions == real['dimensions'] and all(len(a) == real['n'] for a in arrays)
    assert real['config']['covariance'] == 'empirical'
    covariance = estimate_covariance(torch.from_numpy(np.concatenate(arrays, axis=1)))
    values = pid_from_covariance(covariance, *dimensions)
    for key, value in values.items():
        np.testing.assert_allclose(value, real['estimates']['raw'][key], atol=1e-8, rtol=0)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as file:
        np.savez(file, covariance=covariance.numpy(), dimensions=dimensions)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    info = dict(subject=subject, target=target, n=real['n'], dimensions=dimensions,
                estimator='empirical', order=['M', 'X', 'Y'], dtype='float64',
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), eigen_values=values)
    save_result(metadata, info)
    return path, info


def export(args):
    """Keep unfinished cases explicit; never present partial sums as full totals."""
    rows = []
    for subject in range(1, 9):
        for target in ('lh', 'rh'):
            folder = args.output / f"subj{subject:02d}"
            path = folder / f'{target}_hV4_thin.json'
            result = read(path) if path.exists() else dict(status='queued')
            real = read(args.results / 'cases' / f'subj{subject:02d}_{target}_real_0.json')
            evaluations = 1 + real['config']['n_shuffles'] + real['config']['n_resamples']
            row = dict(subject=subject, target=target, status=result['status'],
                       single_seconds=None, evaluations=evaluations, projected_minutes=None, max_atom_difference=None)
            if result['status'] == 'completed':
                assert result['commit'] == THIN_COMMIT and result['threads'] == args.threads
                assert result['dimensions'] == real['dimensions']
                row.update(single_seconds=result['seconds'], projected_minutes=evaluations * result['seconds'] / 60,
                           max_atom_difference=max(abs(result['values'][k] - real['estimates']['raw'][k]) for k in ATOMS))
            rows.append(row)
    path = args.output / 'thin_participants.csv'
    with path.with_suffix('.tmp').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    path.with_suffix('.tmp').replace(path)
    totals = []
    for target in ('lh', 'rh'):
        completed = [r for r in rows if r['target'] == target and r['status'] == 'completed']
        totals.append(dict(target=target, completed=len(completed), required=8,
                           total_projected_minutes=sum(r['projected_minutes'] for r in completed) if len(completed) == 8 else None))
    save_result(args.output / 'thin_participants_summary.json', dict(updated_at=time(), targets=totals,
                projection_assumption='Recorded evaluation count times each participant/target single-call time; excludes surrogate generation and covariance fitting.'))
    return rows


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    # The OS releases this lock even if the detached queue is killed.
    with (args.output / 'thin_queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        state = dict(status='running', started_at=time(), subjects=args.subjects,
                     threads=args.threads, timeout_seconds=args.timeout)
        save_result(args.output / 'thin_queue.json', state)
        export(args)
        for subject in args.subjects:
            for target in ('lh', 'rh'):
                folder = args.output / f'subj{subject:02d}'
                folder.mkdir(exist_ok=True)
                path = folder / f'{target}_hV4_thin.json'
                real = read(args.results / 'cases' / f'subj{subject:02d}_{target}_real_0.json')
                if path.exists() and read(path)['status'] == 'completed':
                    previous = read(path)
                    assert previous['subject'] == subject and previous['dimensions'] == real['dimensions']
                    assert previous['threads'] == args.threads and previous['commit'] == THIN_COMMIT
                    cached = args.inputs / f'subj{subject:02d}' / f'{target}_hV4_covariance.npz'
                    if cached.exists() and "input_sha256" in previous:
                        assert hashlib.sha256(cached.read_bytes()).hexdigest() == previous['input_sha256']
                    continue
                state.update(subject=subject, target=target, updated_at=time())
                save_result(args.output / 'thin_queue.json', state)
                save_result(path, dict(subject=subject, target=f'{target}_hV4', solver='thin', status='preparing'))
                print(f'START subj{subject:02d} {target}: preparing covariance', flush=True)
                try:
                    covariance, info = prepare(args, subject, target, real)
                    command = [str(args.python), str(Path(__file__).with_name('benchmark_reference_pid.py')),
                               '--worker', 'thin', '--subject', str(subject), '--target', f'{target}_hV4',
                               '--input', str(covariance), '--output', str(folder),
                               '--thin-repo', str(args.thin_repo), '--threads', str(args.threads)]
                    with (folder / f'{target}_hV4_thin.log').open('a') as log:
                        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    record = read(path)
                    assert process.returncode == 0, f'Worker exit {process.returncode}; see solver log'
                    assert record['status'] == 'completed' and record['input_sha256'] == info['sha256']
                    record['atom_differences'] = {k: record['values'][k] - real['estimates']['raw'][k] for k in ATOMS}
                    record['max_atom_difference'] = max(abs(v) for v in record['atom_differences'].values())
                    save_result(path, record)
                except subprocess.TimeoutExpired:
                    record = read(path)
                    record.update(status='timeout', timeout_seconds=args.timeout, finished_at=time())
                    save_result(path, record)
                    print(f'TIMEOUT subj{subject:02d} {target}', flush=True)
                except Exception as error:
                    # A failed case stays visible, while independent queued cases can proceed.
                    traceback.print_exc()
                    record = read(path)
                    record.update(status='failed', error=repr(error), finished_at=time())
                    save_result(path, record)
                rows = export(args)
                print(f"DONE subj{subject:02d} {target}: {read(path)['status']}", flush=True)
        rows = export(args)
        state.update(status='completed' if all(r['status'] == 'completed' for r in rows) else 'finished_with_incomplete_cases',
                     finished_at=time(), completed=sum(r['status'] == 'completed' for r in rows), required=16)
        save_result(args.output / 'thin_queue.json', state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', type=int, choices=range(1, 9), default=list(range(1, 9)))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=base / 'results')
    parser.add_argument('--output', type=Path, default=base / 'results/reference_runtime')
    parser.add_argument('--thin-repo', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--threads', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=21600)
    args = parser.parse_args()
    assert args.threads > 0 and args.timeout > 0 and len(set(args.subjects)) == len(args.subjects)
    run(args)
