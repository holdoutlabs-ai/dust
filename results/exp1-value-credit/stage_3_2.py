"""Stage 3.2 (PLAN): variant C fed the true dL/do reproduces autograd's dL/dv, dL/dve and weight gradients.

Records the numbers behind tests/test_analytic_autograd.py (same autograd oracle, imported from the test) as JSON.
    python results/exp1-value-credit/stage_3_2.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]

from diagnostics.tiny import tiny_config, tiny_recipe  # noqa: E402
from dust import DUST, make_model  # noqa: E402
from model import has_ve  # noqa: E402
from test_analytic_autograd import autograd_truth, cos_and_ratio, randomize_gates  # noqa: E402


def pin(n_layer):
    model = make_model(tiny_config(n_layer=n_layer), torch.device('cpu'), 42)
    randomize_gates(model)
    tokens = torch.randint(0, 4096, (2, 129), generator=torch.Generator().manual_seed(1))
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    est = DUST(model, tiny_recipe(), seed=0)
    rows = []
    try:
        with torch.no_grad():
            est.capture(x, y)
        truth, weight_grads = autograd_truth(model, x, y)
        with torch.no_grad():
            out = est.analytic_value_errors({i: truth['o'][i].clone() for i in range(n_layer)}, gamma=1.0)
            for i in range(n_layer):
                for kind, site in [('v', f'transformer.h.{i}.attn.c_v')] + (
                        [('ve', f'value_embeds.{i}')] if has_ve(i, n_layer) else []):
                    c, r = cos_and_ratio(out[site], truth[kind][i])
                    wc, wr = cos_and_ratio(est.reassemble(site, out[site]), weight_grads[site + '.weight'])
                    rows.append(dict(n_layer=n_layer, block=i, site=site, window=model.window_sizes[i][0],
                                     error_cos=float(c), error_ratio=float(r),
                                     weight_cos=float(wc), weight_ratio=float(wr)))
    finally:
        est.close()
    return rows


rows = pin(2) + pin(4)
passed = all(r[k] > 0.9999 for r in rows for k in ('error_cos', 'weight_cos')) and all(
    abs(r[k] - 1) < 1e-4 for r in rows for k in ('error_ratio', 'weight_ratio'))
out = dict(criterion='cos > 0.9999 and |ratio - 1| < 1e-4, gamma = 1, targets = autograd dL/do (summed loss)',
           torch=torch.__version__, dtype='float32 CPU', rows=rows, passed=passed)
path = Path(__file__).with_name('3.2-autograd-pin.json')
path.write_text(json.dumps(out, indent=2) + '\n')
for r in rows:
    print(f"L{r['n_layer']} b{r['block']} {r['site']:<26} w={r['window']:>3} "
          f"err cos={r['error_cos']:.8f} ratio={r['error_ratio']:.8f} | "
          f"weight cos={r['weight_cos']:.8f} ratio={r['weight_ratio']:.8f}")
print('passed', passed)
