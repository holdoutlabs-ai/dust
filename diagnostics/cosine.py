"""Per-parameter cosine of the DUST gradient estimate against autograd (PLAN stage 3.3, §2.2).

For each batch the true mean-loss gradient comes from one autograd pass; each variant then estimates
it with DUST under each noise seed (paired: same batches, same seeds across variants). Reported per
parameter, per layer type (mean over that type's parameters, paper Appendix G), for the concatenation
of all parameters ("full") and of the value / value-embedding parameters ("value_subset").

Variants: A = sampled value credit, B = analytic with the recipe gamma, C = analytic with gamma = 1.

    python -m diagnostics.cosine --config tiny --seeds 0,1 --batches 2 --device cpu --out c.json
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

import torch

from dust import DUST, Recipe, recipe, make_model, load_sequences, GPTConfig
from diagnostics.tiny import tiny_config, tiny_recipe

VARIANTS = {
    'A': dict(value_credit='sampled', value_gamma=None),
    'B': dict(value_credit='analytic', value_gamma=None),
    'C': dict(value_credit='analytic', value_gamma=1.0),
}

LAYER_TYPES = ('wte', 'lm_head', 'lambdas', 've', 'q', 'k', 'v', 'gate', 'attn_out', 'mlp_in', 'mlp_out')
VALUE_TYPES = ('v', 've')

_INFIXES = (
    ('.attn.c_q.', 'q'), ('.attn.c_k.', 'k'), ('.attn.c_v.', 'v'), ('.attn.ve_gate.', 'gate'),
    ('.attn.c_proj.', 'attn_out'), ('.mlp.c_fc.', 'mlp_in'), ('.mlp.c_proj.', 'mlp_out'),
)


def layer_type(name):
    """Map a parameter name to one of LAYER_TYPES; unknown names raise so nothing is silently dropped."""
    if name == 'transformer.wte.weight':
        return 'wte'
    if name == 'lm_head.weight':
        return 'lm_head'
    if name.endswith('lambdas'):
        return 'lambdas'
    if name.startswith('value_embeds.'):
        return 've'
    if name.startswith('transformer.h.'):
        for infix, kind in _INFIXES:
            if infix in name:
                return kind
    raise ValueError(f'Unknown parameter for layer_type: {name!r}')


def variant_settings(base, variant):
    """The base recipe with the variant's value credit; analytic variants drop the local v / ve draws."""
    spec = VARIANTS[variant]
    if spec['value_credit'] == 'sampled':
        return replace(base, **spec)
    return replace(base, **spec, local={**base.local, 'v': 0, 've': 0})


def _autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def autograd_gradients(model, x, y):
    """Autograd gradient of model(x, y) (mean token loss) as float32 copies; leaves the model grad-free."""
    device = next(model.parameters()).device
    try:
        model.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            with _autocast(device):
                loss = model(x, y)
            loss.backward()
        return {
            name: (p.grad.float().clone() if p.grad is not None else torch.zeros_like(p, dtype=torch.float32))
            for name, p in model.named_parameters()
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)


def dust_gradients(model, settings, x, y, seed):
    """DUST estimate of the mean-loss gradient, the seconds estimate() took, and peak CUDA bytes (None on CPU)."""
    device = next(model.parameters()).device
    cuda = device.type == 'cuda'
    est = DUST(model, settings, seed=seed)
    try:
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        gradients = est.estimate(x, y)
        if cuda:
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device) if cuda else None
        valid_tokens = (y != -1).sum()
        return {name: (g / valid_tokens).float() for name, g in gradients.items()}, seconds, peak
    finally:
        est.close()


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = a.norm().item(), b.norm().item()
    if na == 0 or nb == 0 or not (math.isfinite(na) and math.isfinite(nb)):
        return float('nan')
    return (a @ b).item() / (na * nb)


def _ratio(est, true):
    nt = true.double().norm().item()
    return est.double().norm().item() / nt if nt != 0 else float('nan')


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float('nan')


def compare(est, true):
    """Cosines and norm ratios of est against true, per parameter, per layer type and concatenated."""
    names = list(true)
    per_param = {
        n: dict(cos=_cos(est[n], true[n]), ratio=_ratio(est[n], true[n]), type=layer_type(n)) for n in names
    }
    per_type = {}
    for kind in LAYER_TYPES:
        members = [per_param[n] for n in names if per_param[n]['type'] == kind]
        if members:
            per_type[kind] = dict(
                cos_mean=_mean(m['cos'] for m in members), ratio_mean=_mean(m['ratio'] for m in members),
            )

    def concat(subset):
        if not subset:
            return float('nan')
        return _cos(torch.cat([est[n].flatten() for n in subset]), torch.cat([true[n].flatten() for n in subset]))

    value_names = [n for n in names if per_param[n]['type'] in VALUE_TYPES]
    return dict(per_param=per_param, per_type=per_type, full=concat(names), value_subset=concat(value_names))


def _stats(values):
    values = list(values)
    if not values:
        return float('nan'), float('nan')
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))  # sample std (ddof=1)
    return mean, std


def run(model, batches, variants, seeds, settings_base):
    """Paired comparison over batches x seeds for each variant; returns aggregates and per-record results."""
    raw = {v: [] for v in variants}
    times = {v: [] for v in variants}
    peaks = {v: [] for v in variants}
    records = []
    settings = {v: variant_settings(settings_base, v) for v in variants}
    if batches:  # one discarded estimate so CUDA warm-up isn't charged to the first variant's timing
        dust_gradients(model, settings[variants[0]], *batches[0], seeds[0])
    for j, (x, y) in enumerate(batches):
        true = autograd_gradients(model, x, y)
        for v in variants:
            for seed in seeds:
                est, seconds, peak = dust_gradients(model, settings[v], x, y, seed)
                result = compare(est, true)
                del est
                raw[v].append(result)
                times[v].append(seconds)
                if peak is not None:
                    peaks[v].append(peak)
                records.append(dict(
                    variant=v, seed=seed, batch=j, per_type=result['per_type'],
                    full=result['full'], value_subset=result['value_subset'], seconds=seconds,
                ))
        del true

    out = {}
    for v in variants:
        results = raw[v]
        per_param = {}
        for name in results[0]['per_param']:
            cm, cs = _stats(r['per_param'][name]['cos'] for r in results)
            rm, rs = _stats(r['per_param'][name]['ratio'] for r in results)
            per_param[name] = dict(cos_mean=cm, cos_std=cs, ratio_mean=rm, ratio_std=rs,
                                   type=results[0]['per_param'][name]['type'])
        per_type = {}
        for kind in results[0]['per_type']:
            cm, cs = _stats(r['per_type'][kind]['cos_mean'] for r in results)
            rm, rs = _stats(r['per_type'][kind]['ratio_mean'] for r in results)
            per_type[kind] = dict(cos_mean=cm, cos_std=cs, ratio_mean=rm, ratio_std=rs)
        full = dict(zip(('mean', 'std'), _stats(r['full'] for r in results)))
        value_subset = dict(zip(('mean', 'std'), _stats(r['value_subset'] for r in results)))
        out[v] = dict(
            settings=asdict(settings[v]), per_param=per_param, per_type=per_type, full=full,
            value_subset=value_subset, seconds_mean=_mean(times[v]),
            peak_gib_max=max(peaks[v]) / 2 ** 30 if peaks[v] else None,
        )
    return dict(variants=out, records=records)


def format_table(variants):
    names = list(variants)
    kinds = [k for k in LAYER_TYPES if any(k in variants[v]['per_type'] for v in names)]
    rows = [('type', *names)]
    for k in kinds:
        cells = []
        for v in names:
            t = variants[v]['per_type'].get(k)
            cells.append(f"{t['cos_mean']:+.3f}+-{t['cos_std']:.3f}" if t else '-')
        rows.append((k, *cells))
    for label in ('full', 'value_subset'):
        rows.append((label, *(f"{variants[v][label]['mean']:+.3f}+-{variants[v][label]['std']:.3f}" for v in names)))
    rows.append(('seconds', *(f"{variants[v]['seconds_mean']:.2f}" for v in names)))
    rows.append(('peak_gib', *(
        f"{variants[v]['peak_gib_max']:.2f}" if variants[v]['peak_gib_max'] is not None else '-' for v in names
    )))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return '\n'.join('  '.join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', choices=('tiny', 'paper'), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--variants', default='A,B,C')
    parser.add_argument('--population', type=int, choices=(256, 1024, 4096, 16384), default=256)
    parser.add_argument('--tokens', choices=('1M', '10M'), default='1M')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--batches', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--nrep', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batches < 1 or not args.seeds.strip():
        parser.error('need --batches >= 1 and at least one seed')

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown or not variants:
        parser.error(f'--variants must be drawn from {",".join(VARIANTS)}, got {args.variants!r}')
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')

    notes = ['std is the sample std (ddof=1) over (seed, batch) records; 0 for one record',
             'peak_gib is process-wide peak during estimate() (includes model, batches, autograd grads)',
             'one discarded warm-up estimate precedes the timed ones']
    if args.config == 'paper':
        config, base = GPTConfig(), recipe(args.tokens, args.population)
    else:
        config, base = tiny_config(), tiny_recipe()
        notes.append('tiny: --population and --tokens ignored (tiny_recipe())')
    if args.nrep is not None:
        base = replace(base, nrep=args.nrep)

    model = make_model(config, device, 42)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    elif args.config == 'tiny':
        torch.manual_seed(0)
        with torch.no_grad():
            for block in model.transformer.h:
                if block.attn.ve_gate is not None:
                    block.attn.ve_gate.weight.normal_(0, 0.5)
        notes.append('tiny without checkpoint: ve_gate.weight ~ N(0, 0.5) under torch.manual_seed(0)')

    bs, seq = args.batch_size, config.sequence_len
    batches = []
    if args.data is not None:
        rows = load_sequences(args.data, args.batches * bs, sequence_len=2048)[:, :seq + 1]
        for j in range(args.batches):
            chunk = rows[j * bs:(j + 1) * bs].to(device)
            batches.append((chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()))
    else:
        for j in range(args.batches):
            tokens = torch.randint(0, 4096, (bs, seq + 1), generator=torch.Generator().manual_seed(1000 + j))
            tokens = tokens.to(device)
            batches.append((tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()))
        notes.append('random token batches: torch.Generator().manual_seed(1000 + j)')

    start = time.perf_counter()
    result = run(model, batches, variants, seeds, base)
    elapsed = time.perf_counter() - start

    args_record = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    report = {
        'args': {**args_record, 'variants': variants, 'seeds': seeds, 'device': str(device), 'notes': notes},
        'command': ' '.join(sys.argv if argv is None else ['python -m diagnostics.cosine', *argv]),
        'torch': torch.__version__,
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'recipe': asdict(base),
        'elapsed_seconds': elapsed,
        'variants': result['variants'],
        'records': result['records'],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as stream:
        json.dump(report, stream, indent=1)

    print(format_table(result['variants']))
    print(f'elapsed {elapsed:.1f}s -> {args.out}')
    return report


if __name__ == '__main__':
    main()
