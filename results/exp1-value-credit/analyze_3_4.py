"""Stage 3.4 readout (PLAN §3.4): best_val and test_at_best_val per variant, mean ± std over seeds, and
paired per-seed differences B−A, C−A, C−B (seeds share model init and data order across variants).

    python results/exp1-value-credit/analyze_3_4.py [RUNS_DIR]     # default: results/exp1-value-credit/3.4
Writes 3.4-summary.json next to this script and prints a table.
"""
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
runs = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / '3.4'
VARIANTS, SEEDS, METRICS = 'ABC', (42, 43, 44), ('best_val', 'test_at_best_val')


def load(variant, seed):
    path = runs / f'{variant}-s{seed}' / 'result.json'
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    config = json.loads((path.parent / 'config.json').read_text())
    curve = [json.loads(line) for line in (path.parent / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    return dict(best_val=result['best_val'], test_at_best_val=result['test_at_best_val'],
                value_credit=config['recipe'].get('value_credit'), value_gamma=config['recipe'].get('value_gamma'),
                val_curve=[(r['step'], r['val_loss']) for r in curve if 'val_loss' in r],
                seconds=curve[-1]['elapsed_seconds'] if curve else None)


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(mean=st.mean(values), std=st.stdev(values) if len(values) > 1 else 0.0, n=len(values))


data = {v: {s: load(v, s) for s in SEEDS} for v in VARIANTS}
expected = {'A': ('sampled', None), 'B': ('analytic', None), 'C': ('analytic', 1.0)}
for v in VARIANTS:  # guard against a mislabelled run
    for s, r in data[v].items():
        if r is not None and (r['value_credit'], r['value_gamma']) != expected[v]:
            raise SystemExit(f'{v}-s{s}: config says {r["value_credit"]}, {r["value_gamma"]}; expected {expected[v]}')

summary = dict(runs=str(runs), per_variant={}, paired={}, per_run={})
for v in VARIANTS:
    summary['per_variant'][v] = {m: stats([r and r[m] for r in data[v].values()]) for m in METRICS}
    summary['per_variant'][v]['seconds'] = stats([r and r['seconds'] for r in data[v].values()])
    for s, r in data[v].items():
        if r is not None:
            summary['per_run'][f'{v}-s{s}'] = {k: r[k] for k in (*METRICS, 'seconds')}
for a, b in (('B', 'A'), ('C', 'A'), ('C', 'B')):
    for m in METRICS:
        diffs = {s: data[a][s][m] - data[b][s][m] for s in SEEDS if data[a][s] and data[b][s]}
        summary['paired'][f'{a}-{b} {m}'] = dict(per_seed=diffs, **(stats(list(diffs.values())) or {}))

(HERE / '3.4-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(f"{'variant':<8}" + ''.join(f'{m:>26}' for m in METRICS))
for v in VARIANTS:
    cells = []
    for m in METRICS:
        x = summary['per_variant'][v][m]
        cells.append(f"{x['mean']:.4f} ± {x['std']:.4f} (n={x['n']})" if x else 'n/a')
    print(f'{v:<8}' + ''.join(f'{c:>26}' for c in cells))
print('paired differences (negative = first variant better):')
for key, x in summary['paired'].items():
    if x.get('n'):
        seeds = ' '.join(f'{d:+.4f}' for d in x['per_seed'].values())
        print(f"  {key:<24} mean {x['mean']:+.4f} ± {x['std']:.4f}   per seed: {seeds}")
