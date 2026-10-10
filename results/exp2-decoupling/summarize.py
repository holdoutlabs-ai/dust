"""Experiment 2 readout: one row per configuration (all flags except the seed), mean +- std over seeds of best_val and
test_at_best_val, and paired per-seed differences against a reference configuration.

    python results/exp2-decoupling/summarize.py RUNS_DIR [--reference NAME_PREFIX] [--out FILE.json]

RUNS_DIR holds one directory per run (config.json, metrics.jsonl, result.json), as written by baselines/backprop.py.
"""
import argparse
import json
import statistics as st
from pathlib import Path

KEYS = ('optimizer', 'lr', 'momentum', 'adam_lr', 'grad_delay', 'delay_mode', 'weight_prediction')


def load(run):
    config = json.loads((run / 'config.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    return dict(name=run.name, seed=config['seed'], key=tuple(config.get(k) for k in KEYS),
                best_val=result['best_val'], test=result['test_at_best_val'],
                final_val=[m['val_loss'] for m in metrics if 'val_loss' in m][-1],
                seconds=metrics[-1]['elapsed_seconds'] if metrics else None)


def label(key):
    parts = [f'{k}={v}' for k, v in zip(KEYS, key) if v not in (None, 0, False, 'uniform')]
    return ' '.join(parts)


def stats(values):
    return (st.mean(values), st.stdev(values) if len(values) > 1 else 0.0, len(values))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path)
    parser.add_argument('--reference', help='name prefix of the reference configuration, e.g. muon-lr0.02')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)

    runs = [load(d) for d in sorted(args.runs.iterdir()) if (d / 'result.json').is_file()]
    groups = {}
    for r in runs:
        groups.setdefault(r['key'], []).append(r)
    reference = None
    if args.reference:
        matches = [k for k, rs in groups.items() if any(r['name'].startswith(args.reference) for r in rs)]
        if len(matches) != 1:
            raise SystemExit(f'--reference {args.reference!r} matches {len(matches)} configurations')
        reference = {r['seed']: r for r in groups[matches[0]]}

    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: st.mean(r['best_val'] for r in kv[1])):
        bv, test = stats([r['best_val'] for r in rs]), stats([r['test'] for r in rs])
        row = dict(config=dict(zip(KEYS, key)), label=label(key), runs=[r['name'] for r in rs],
                   best_val=dict(zip(('mean', 'std', 'n'), bv)), test=dict(zip(('mean', 'std', 'n'), test)),
                   seconds_mean=st.mean(r['seconds'] for r in rs if r['seconds'] is not None))
        if reference:
            diffs = {r['seed']: r['best_val'] - reference[r['seed']]['best_val'] for r in rs if r['seed'] in reference}
            if diffs:
                m, s, n = stats(list(diffs.values()))
                row['paired_vs_reference'] = dict(per_seed=diffs, mean=m, std=s, n=n)
        rows.append(row)

    for row in rows:
        b, t = row['best_val'], row['test']
        line = f"{row['label']:<58} val {b['mean']:.4f} +- {b['std']:.4f} (n={b['n']})  test {t['mean']:.4f}  {row['seconds_mean']:.0f}s"
        if 'paired_vs_reference' in row:
            p = row['paired_vs_reference']
            line += f"  dref {p['mean']:+.4f} +- {p['std']:.4f}"
        print(line)
    if args.out:
        args.out.write_text(json.dumps(dict(runs_dir=str(args.runs), reference=args.reference, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
