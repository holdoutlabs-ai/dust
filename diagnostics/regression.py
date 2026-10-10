"""Stage 3.1: the analytic value credit (variant B) is the mean of the sampled estimator (variant A).

For fixed model, batch, targets and gamma, DUST.local_errors(kind, K, targets) for kind in {'v', 've'}
is uncentred and divided by K and sigma, so its expectation is DUST.analytic_value_errors(targets, gamma).
This script checks that the sampled estimate converges to B in direction and norm, with squared error
shrinking like 1/K, for several gammas: c_v on short- and full-window blocks, value_embeds on short- and
full-window VE blocks (a full-window block without VE does not occur: the last block is always long and has VE).

Pass rule (PLAN 3.1 amendment, 2026-10-10), per seed at the largest K, with p = cos * ratio = <a, B> / |B|^2:
  |ratio^2 - rel - 1| < 0.05      identically |p - 1| < 0.025: the projection on B is unbiased (norm/scale)
  |cos - 1/sqrt(1+rel)| < 0.003   a function of the same (p, ratio); weaker, kept as approved
  slope of log rel vs log K in (-1.25, -0.75), rel decreasing in K: guards bias orthogonal to B (a floor)
  wrong-gamma control: rel vs B(gamma=1) > 2 * rel vs B, on gamma=0.5 data
Note 3.1 cannot see the mask edge (one key at lag = window, weight ~gamma^window / window); stage 3.2 pins it.

    python -m diagnostics.regression --out results/exp1-value-credit/3.1-regression.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.tiny import tiny_config, tiny_recipe  # noqa: E402
from dust import DUST, make_model  # noqa: E402
from model import has_ve  # noqa: E402

WRONG_GAMMA = 1.0
CONTROL_GAMMA = 0.5


def setup(n_layer=2, nrep=8, seed=0):
    """Tiny model with random ve_gate, a captured random batch, and fixed random targets."""
    device = torch.device('cpu')
    model = make_model(tiny_config(n_layer=n_layer), device, 42)
    torch.manual_seed(seed)
    with torch.no_grad():
        for block in model.transformer.h:
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight.normal_(0, 0.5)
    tokens = torch.randint(0, 4096, (2, 129), generator=torch.Generator().manual_seed(seed + 1))
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    # The recipe's draw counts are unused here (local_errors takes K), but DUST.__init__ requires every
    # count to be a multiple of nrep; tiny_recipe's smallest count is q = 4 * draws_scale.
    scale = nrep // math.gcd(4, nrep)
    est = DUST(model, tiny_recipe(draws_scale=scale, nrep=nrep), seed=seed)
    with torch.no_grad():
        est.capture(x, y)
    cfg = model.config
    g = torch.Generator().manual_seed(seed + 2)
    targets = {
        i: torch.randn(est.batch, est.length, cfg.n_embd, generator=g) for i in range(cfg.n_layer)
    }
    info = {
        i: dict(window=int(model.window_sizes[i][0]), has_ve=bool(has_ve(i, cfg.n_layer)))
        for i in range(cfg.n_layer)
    }
    return est, targets, info


def compare(a, b):
    """cos, norm ratio and relative squared error of a against reference b (flattened)."""
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = a.norm(), b.norm()
    return dict(
        cos=float(a @ b / (na * nb)),
        ratio=float(na / nb),
        rel=float((a - b).pow(2).sum() / nb.pow(2)),
    )


def slope(ks, values):
    """Least-squares slope of log(values) against log(ks)."""
    xs = [math.log(k) for k in ks]
    ys = [math.log(v) for v in values]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else float('nan')


def site_block(site):
    parts = site.split('.')
    return int(parts[2]) if site.startswith('transformer.h.') else int(parts[1])


def evaluate(last, slope_value, rel_means):
    """Gating checks for one (config, gamma, site) entry; `last` is its per_K record at the largest K."""
    return {
        '|ratio^2-rel-1|<0.05': all(abs(r ** 2 - e - 1) < 0.05 for r, e in zip(last['ratio'], last['rel'])),
        '|cos-1/sqrt(1+rel)|<0.003': all(
            abs(c - (1 + e) ** -0.5) < 0.003 for c, e in zip(last['cos'], last['rel'])
        ),
        'slope_in(-1.25,-0.75)': -1.25 < slope_value < -0.75,
        'rel_decreasing': all(r1 < r0 for r0, r1 in zip(rel_means, rel_means[1:])),
    }


def run(gammas=(0.98, 0.9, 0.5), draws=(64, 256, 1024, 4096), seeds=(0, 1, 2), layers=(2, 4), nrep=8):
    start = time.perf_counter()
    draws = sorted(draws)
    # local_errors runs draws // nrep chunks but divides by draws: a remainder would silently rescale.
    if any(K % nrep for K in draws):
        raise ValueError(f'every K in draws must be a multiple of nrep={nrep}')
    results, controls, checks, informational = [], [], {}, {}
    with torch.no_grad():
        for n_layer in layers:
            est, targets, info = setup(n_layer=n_layer, nrep=nrep)
            try:
                for g_index, gamma in enumerate(gammas):
                    est.settings.gamma = gamma
                    reference = est.analytic_value_errors(targets, gamma)
                    wrong = est.analytic_value_errors(targets, WRONG_GAMMA) if gamma == CONTROL_GAMMA else None
                    for k_index, kind in enumerate(('v', 've')):
                        # samples[K][seed] = {site: tensor}
                        samples = {}
                        for K in draws:
                            samples[K] = []
                            for seed in seeds:
                                # independent noise per (seed, K, gamma, kind)
                                est.generator.manual_seed(100_003 * seed + K + 7 * k_index + 1_000_003 * g_index)
                                samples[K].append(est.local_errors(kind, K, targets))
                        sites = sorted(samples[draws[0]][0], key=site_block)
                        for site in sites:
                            block = site_block(site)
                            b = reference[site]
                            per_k = []
                            for K in draws:
                                stats = [compare(s[site], b) for s in samples[K]]
                                mean = compare(torch.stack([s[site] for s in samples[K]]).mean(0), b)
                                per_k.append(dict(
                                    K=K,
                                    cos=[r['cos'] for r in stats],
                                    ratio=[r['ratio'] for r in stats],
                                    rel=[r['rel'] for r in stats],
                                    cos_mean=mean['cos'], ratio_mean=mean['ratio'],
                                ))
                            rel_means = [sum(p['rel']) / len(p['rel']) for p in per_k]
                            entry = dict(
                                n_layer=n_layer, gamma=gamma, site=site, block=block,
                                window=info[block]['window'], has_ve=info[block]['has_ve'], kind=kind,
                                per_K=per_k, slope=slope(draws, rel_means),
                            )
                            results.append(entry)

                            name = f'L{n_layer}/g{gamma}/{site}'
                            last = per_k[-1]
                            for check, ok in evaluate(last, entry['slope'], rel_means).items():
                                checks[f'{name}/{check}'] = ok
                            # The original absolute thresholds, informational only (not part of `passed`).
                            mean_ratio = sum(last['ratio']) / len(last['ratio'])
                            informational[f'{name}/min_cos>0.99'] = min(last['cos']) > 0.99
                            informational[f'{name}/|ratio-1|<0.03'] = abs(mean_ratio - 1) < 0.03

                            if wrong is not None:
                                a = samples[draws[-1]][0][site]
                                vs_b, vs_w = compare(a, b), compare(a, wrong[site])
                                controls.append(dict(
                                    n_layer=n_layer, site=site, gamma=gamma, wrong_gamma=WRONG_GAMMA,
                                    K=draws[-1], cos_vs_B=vs_b['cos'], cos_vs_wrong_gamma=vs_w['cos'],
                                    rel_vs_B=vs_b['rel'], rel_vs_wrong_gamma=vs_w['rel'],
                                ))
                                checks[f'control/L{n_layer}/{site}/rel_wrong>2*rel_B'] = (
                                    vs_w['rel'] > 2 * vs_b['rel']
                                )
            finally:
                est.close()
    elapsed = time.perf_counter() - start
    config = dict(
        gammas=list(gammas), draws=list(draws), seeds=list(seeds), layers=list(layers), nrep=nrep,
        sigma=0.05, control_gamma=CONTROL_GAMMA, wrong_gamma=WRONG_GAMMA,
        torch=torch.__version__, elapsed_s=round(elapsed, 2),
    )
    return dict(
        config=config, results=results, controls=controls, checks=checks, info_checks=informational,
        passed=bool(checks) and all(checks.values()),
    )


def print_table(out):
    print(f"{'L':>2} {'gamma':>5} {'site':<28} {'win':>4} {'ve':>3} {'K':>5} "
          f"{'min_cos':>8} {'ratio':>7} {'rel':>9} {'cos_mean':>8} {'slope':>7}")
    for r in out['results']:
        last = r['per_K'][-1]
        rel = sum(last['rel']) / len(last['rel'])
        ratio = sum(last['ratio']) / len(last['ratio'])
        print(f"{r['n_layer']:>2} {r['gamma']:>5} {r['site']:<28} {r['window']:>4} {'y' if r['has_ve'] else 'n':>3} "
              f"{last['K']:>5} {min(last['cos']):>8.5f} {ratio:>7.4f} {rel:>9.2e} {last['cos_mean']:>8.5f} "
              f"{r['slope']:>7.3f}")
    for c in out['controls']:
        print(f"control L{c['n_layer']} {c['site']}: gamma={c['gamma']} vs B cos={c['cos_vs_B']:.4f} "
              f"rel={c['rel_vs_B']:.2e} | vs gamma={c['wrong_gamma']} cos={c['cos_vs_wrong_gamma']:.4f} "
              f"rel={c['rel_vs_wrong_gamma']:.2e}")
    failed = [k for k, v in out['checks'].items() if not v]
    print(f"checks: {len(out['checks']) - len(failed)}/{len(out['checks'])} passed")
    for k in failed:
        print(f"  FAILED {k}")
    info = out.get('info_checks', {})
    print(f"informational (original PLAN thresholds, not gating): "
          f"{sum(info.values())}/{len(info)} met")
    print(f"passed={out['passed']}  elapsed={out['config']['elapsed_s']}s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True, help='JSON output path.')
    parser.add_argument('--quick', action='store_true',
                        help='draws 64,256,1024; gamma 0.5; seeds 0,1; n_layer 2.')
    parser.add_argument('--nrep', type=int, default=8)
    args = parser.parse_args(argv)
    if args.quick:
        out = run(gammas=(0.5,), draws=(64, 256, 1024), seeds=(0, 1), layers=(2,), nrep=args.nrep)
    else:
        out = run(nrep=args.nrep)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print_table(out)
    print(f"wrote {path}")
    if not out['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
