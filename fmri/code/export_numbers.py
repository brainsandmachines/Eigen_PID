"""Regenerate manuscript tables and numerical macros from saved experiment results."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def export(results, output):
    output.mkdir(parents=True, exist_ok=True)
    with (results / 'descriptive_summary.csv').open() as file:
        summary = list(csv.DictReader(file))
    def mean(target, condition, method, component):
        rows = [row for row in summary if (row['target'], row['condition'], row['method'], row['component'])
                == (target, condition, method, component)]
        assert len(rows) == 1
        return float(rows[0]['mean'])
    runtime = json.loads((results / 'runtime_summary.json').read_text())
    assert sum(w['cases'] for w in runtime['workloads']) == 32
    assert runtime['shuffled_replicate'] == 1
    null_max = max(abs(mean(target, 'shuffled', 'permutation', atom))
                   for target in ('lh', 'rh') for atom in ('uix', 'uiy', 'red', 'syn'))
    numbers = {'RawNullLeft': f"{mean('lh', 'shuffled', 'raw', 'mixy'):.1f}",
               'RawNullRight': f"{mean('rh', 'shuffled', 'raw', 'mixy'):.1f}",
               'CorrectedNullLeft': f"{mean('lh', 'shuffled', 'combined', 'mixy'):.3f}",
               'CorrectedNullRight': f"{mean('rh', 'shuffled', 'combined', 'mixy'):.3f}",
               'PermutationNullMaximum': f'{null_max:.3f}',
               'RealMinutes': f"{sum(w['total_seconds'] for w in runtime['workloads'] if w['condition'] == 'real') / 60:.1f}",
               'WholeMinutes': f"{sum(w['total_seconds'] for w in runtime['workloads']) / 60:.1f}"}
    gpu_paths = sorted((results / 'gpu' / 'cases').glob('subj*_real_0.json'))
    if gpu_paths:
        gpu_cases = [json.loads(path.read_text()) for path in gpu_paths]
        assert {(case['subject'], case['target']) for case in gpu_cases} == {
            (subject, target) for subject in range(1, 9) for target in ('lh', 'rh')}
        assert len(gpu_cases) == 16
        assert all(case['config']['disable_numerical_checks'] and
                   case['config']['device'].startswith('cuda') and
                   case['config']['n_shuffles'] == case['config']['n_resamples'] == 20
                   for case in gpu_cases)
        seconds = sum(case['timings']['total'] for case in gpu_cases)
        numbers['GPUMinutes'] = f"{seconds / 60:.1f}"
        (results / 'gpu' / 'runtime_summary.json').write_text(json.dumps({
            'source': 'cases/subj*_{lh,rh}_real_0.json: timings.total',
            'cases': 16, 'pid_evaluations': 656, 'total_seconds': seconds,
        }, indent=2) + '\n')
    with (results / 'roi_dimensions.csv').open() as file:
        dimensions = list(csv.DictReader(file))
    assert len(dimensions) == 8
    lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
             'Participant & ' + ' & '.join('S' + row['subject'] for row in dimensions) + r' \\', r'\midrule']
    for key, label in (('images', 'Images ($N$)'), ('lh:V1', 'Left V1 vertices ($d_{X}$)'), ('rh:V1', 'Right V1 vertices ($d_{Y}$)'),
                       ('lh:hV4', 'Left hV4 vertices ($d_{M}$)'), ('rh:hV4', 'Right hV4 vertices ($d_{M}$)')):
        cells = [label] + [f"{int(row[key]):,}" for row in dimensions]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / 'dimensions_table.tex').write_text('\n'.join(lines) + '\n')
    # One original PID call per participant; SD describes between-participant variation.
    eigen = {}
    for target in ('lh', 'rh'):
        cases = [json.loads(path.read_text())
                 for path in sorted((results / 'cases').glob(f'subj*_{target}_real_0.json'))]
        assert [case['subject'] for case in cases] == list(range(1, 9))
        assert all(case['config']['n_shuffles'] == case['config']['n_resamples'] == 20
                   for case in cases)
        single = [case['timings']['raw_pid'] for case in cases]
        complete = [case['timings']['total'] for case in cases]
        eigen[target] = {
            'participants': [case['subject'] for case in cases],
            'single_seconds': single,
            'complete_seconds': complete,
            'single_mean_seconds': statistics.mean(single),
            'single_sd_seconds': statistics.stdev(single),
            'complete_mean_seconds': statistics.mean(complete),
            'complete_total_seconds': sum(complete),
            'complete_evaluations': sum(1 + case['config']['n_shuffles'] + case['config']['n_resamples'] for case in cases),
        }
    (results / 'eigen_participants_runtime.json').write_text(json.dumps({
        'source': 'cases/subj*_{lh,rh}_real_0.json: timings.raw_pid and timings.total',
        'sd': 'Sample SD across participants (ddof=1); one original call per participant.',
        'targets': eigen,
    }, indent=2) + '\n')
    # Compare both methods across all eight participants.
    with (results / 'reference_runtime' / 'thin_participants.csv').open() as file:
        thin_rows = list(csv.DictReader(file))
    assert len(thin_rows) == 16
    thin = {}
    for target in ('lh', 'rh'):
        rows = [row for row in thin_rows if row['target'] == target]
        assert sorted(int(row['subject']) for row in rows) == list(range(1, 9))
        assert all(row['status'] == 'completed' for row in rows)
        thin[target] = statistics.mean(float(row['single_seconds']) for row in rows)
    lines = [r'\begin{threeparttable}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'\multirow{2}{*}{\textbf{Target}} & \multicolumn{2}{c}{\textbf{Single call} (seconds)} & \multicolumn{2}{c}{\textbf{Complete analysis} (minutes)} \\',
             r' & Eigen-PID & Thin-PID & Eigen-PID & Thin-PID \\', r'\midrule']
    for target in ('lh', 'rh'):
        timing = eigen[target]
        side = 'Left' if target == 'lh' else 'Right'
        assert timing['complete_evaluations'] == (1 + 20 + 20) * 8
        eigen_minutes = timing['complete_total_seconds'] / 60
        thin_minutes = thin[target] * timing['complete_evaluations'] / 60
        numbers[f'EigenComplete{side}Minutes'] = f'{eigen_minutes:.1f}'
        numbers[f'ThinComplete{side}Minutes'] = f'{thin_minutes:,.0f}'
        cells = [side + ' hV4',
                 f"{timing['single_mean_seconds']:,.2f}", f"{thin[target]:,.2f}",
                 f"{eigen_minutes:,.2f}", f"{thin_minutes:,.2f}" + r"{\tnote{\dag}}"]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\begin{tablenotes}[flushleft]', r'\footnotesize',
              r"\item[\dag] Conservative projections obtained by multiplying the mean single-call time across all eight participants by 328, excluding the time to generate resampled datasets and estimate their covariances.",
              r'\end{tablenotes}', r'\end{threeparttable}']
    (output / 'runtime_table.tex').write_text('\n'.join(lines) + '\n')
    numbers['EigenTotalMinutes'] = f"{sum(t['complete_total_seconds'] for t in eigen.values()) / 60:.1f}"
    numbers['ThinTotalDays'] = f"{sum(thin[target] * eigen[target]['complete_evaluations'] for target in thin) / 86400:.0f}"
    (output / 'numbers.tex').write_text('\n'.join('\\newcommand{\\' + key + '}{' + value + '}' for key, value in numbers.items()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.results, args.output)
