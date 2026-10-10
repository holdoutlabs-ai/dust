# WP2 `diagnostics/regression.py`: stage 3.1, B equals A's mean
owner: dust-dev (module), dust-test-writer (`tests/test_regression.py`, in parallel)
interpreter: D:/Micromamba/micromambaenv/envs/dust/python.exe
test cmd: `D:/Micromamba/micromambaenv/envs/dust/python.exe -m pytest tests/test_regression.py -q`
stage cmd: `D:/Micromamba/micromambaenv/envs/dust/python.exe -m diagnostics.regression --out results/exp1-value-credit/3.1-regression.json`

## Goal

Show numerically that the analytic value credit (variant B) is the mean of the sampled estimator
(variant A) for the same γ: as the draw count K grows, the sampled estimate converges to B in
direction **and** norm, and the squared error shrinks like 1/K. Run for several γ (a transposed
decay passes at γ≈1 and fails at γ=0.5), on short-window and full-window blocks, with and without VE.

## Files

dust-dev writes: `diagnostics/regression.py`.
test-writer writes: `tests/test_regression.py`.
do not touch: `dust.py`, `model.py` (WP1 is done and reviewed), `diagnostics/tiny.py`, `diagnostics/cosine.py`
and `tests/test_cosine.py` (WP3 is writing these now), `tests/conftest.py`, other tests, docs.

## Math

For fixed model, batch, `targets` and γ, the sampled estimator
`ê_K = DUST.local_errors(kind, K, targets)` for kind ∈ {`'v'`, `'ve'`} is uncentred and divides by K
and σ, so `E[ê_K] = B := DUST.analytic_value_errors(targets, γ)` exactly (per site), with
`Var ∝ 1/K`. Metrics, per site (flattened tensors), with `a = ê_K` and `b = B[site]`:

```
cos   = <a, b> / (‖a‖ ‖b‖)
ratio = ‖a‖ / ‖b‖                       # biased up by noise: E‖a‖² = ‖b‖² + E‖a−b‖²
rel   = ‖a − b‖² / ‖b‖²                 # expect ∝ 1/K
slope = least-squares slope of log(mean_seeds rel) vs log K        # expect ≈ −1
```

Also report `cos_mean` and `ratio_mean` for the seed-averaged estimate `mean_seeds(ê_K)` against `b`
(the averaged estimate has `len(seeds)·K` effective draws).

**Control** (proves the test can fail): for γ=0.5 also compute `cos` and `rel` of `ê_4096` against
`analytic_value_errors(targets, 1.0)` (the wrong γ). It must be clearly worse than against B.

## Contract (APIs from WP1, already in `dust.py`)

```python
from dust import DUST, make_model
from diagnostics.tiny import tiny_config, tiny_recipe

DUST(model, settings, seed=...)          # registers hooks; call .close() when done
est.capture(x, y)                        # clean pass; fills est.outputs/inputs, est.batch, est.length
est.local_errors(kind, draws, targets, sigma=0.05) -> {site: Tensor[b, t, width]}
                                         # kinds 'v' (all blocks), 've' (VE blocks only); reads
                                         # est.settings.gamma and est.settings.nrep; noise from est.generator
est.analytic_value_errors(targets, gamma) -> {'transformer.h.{i}.attn.c_v': [b,t,n_embd],
                                              'value_embeds.{i}': [b,t,n_embd] (VE blocks only)}
targets: {i: Tensor[b, t, n_embd] float32}   # one per block i
```

`Recipe` is a mutable dataclass: set `est.settings.gamma = γ` to change the sampled decay, and
`est.generator.manual_seed(seed)` before each sampled run so seeds are reproducible.

## Module design

```python
def setup(n_layer=2, nrep=8, seed=0) -> (est, targets, info)
    # model = make_model(tiny_config(n_layer=n_layer), cpu, 42); randomize every ve_gate.weight
    #   with normal_(0, 0.5) under torch.manual_seed(seed) (zero at init hides the gate multiplier)
    # batch: torch.randint(0, 4096, (2, 129), generator=Generator().manual_seed(seed+1)); x,y = [:, :-1], [:, 1:]
    # est = DUST(model, tiny_recipe(nrep=nrep), seed=seed); est.capture(x, y) under torch.no_grad()
    # targets = {i: torch.randn(b, t, n_embd, generator=...seed+2)} (fixed; the identity holds for any targets)
    # info: per block window, has_ve
def compare(a, b) -> dict(cos=, ratio=, rel=)
def run(gammas=(0.98, 0.9, 0.5), draws=(64, 256, 1024, 4096), seeds=(0, 1, 2),
        layers=(2, 4), nrep=8) -> dict          # JSON-serialisable; includes 'passed' and 'checks'
def main(argv=None)   # argparse: --out PATH (required), --quick (draws 64,256,1024; gammas 0.5; seeds 0,1; layers 2),
                      # --nrep INT. Writes JSON (indent=2), prints a compact table, exits 1 if not passed.
```

`nrep` only changes how draws are chunked for `'v'`/`'ve'` (no centring in `local_errors`), so
`nrep=8` is the same estimator, just faster. Record nrep in the JSON. Everything runs under `torch.no_grad()`
in float32 on CPU. Target total runtime for the full run: under ~2 minutes; print timing.

Configs: `n_layer=2` gives block 0 = short window 64 (= sequence_len // 2), no VE; block 1 = full
window 128, VE. `n_layer=4` gives VE on blocks 1 and 3, and block 1 has the short window. So the
two configs cover every case: short/full window × with/without VE.

JSON layout:

```
{"config": {...gammas, draws, seeds, layers, nrep, torch version, elapsed_s},
 "results": [ {"n_layer":2, "gamma":0.5, "site":"transformer.h.0.attn.c_v", "block":0, "window":64,
               "has_ve":false, "kind":"v",
               "per_K": [{"K":64, "cos":[...per seed], "ratio":[...], "rel":[...],
                          "cos_mean":.., "ratio_mean":..}, ...],
               "slope": -1.0x}, ...],
 "controls": [ {"n_layer":2, "site":..., "cos_vs_B":.., "cos_vs_wrong_gamma":.., "rel_vs_B":.., "rel_vs_wrong_gamma":..}],
 "checks": {...name: bool}, "passed": bool}
```

## Pass criteria (`checks`), from PLAN §3.1

For every (n_layer, γ, site) at the largest K (4096):
- `min_seeds cos > 0.99` and `|mean_seeds ratio − 1| < 0.03`;
- `−1.25 < slope < −0.75`;
- rel decreasing in K for the seed-mean.
For every control: `rel_vs_wrong_gamma > 2 · rel_vs_B`.
`passed = all(checks)`. If a threshold fails, do **not** change it: report the numbers. The lead decides.

## Read (only these)

- `dust.py`: `DUST.__init__`, `capture`, `local_errors`, `analytic_value_errors` (signatures and docstrings).
- `diagnostics/tiny.py`, `tests/conftest.py`.

## Rules that apply (CLAUDE.md)

- Index conventions are where this goes wrong: `local_errors`' `credit[t, s]` is `Γᵀ`. Tests must use
  γ well below 1 (e.g. 0.5) so a transposed decay fails.
- Scale: the sampled `'v'` estimator is uncentred and divides by `draws` and `σ`, so B must match A's
  mean in **direction and norm**. A constant-factor mismatch is a bug.
- Tests run on CPU with the tiny config, no network, no data files (use random token batches), fixed
  seeds, and finish in minutes. Expected values come from math or autograd, never from the code under test.
- No changes to `dust.py` or `model.py` in this WP.

## Acceptance (test-writer: turn into tests in `tests/test_regression.py`)

1. `compare` on hand-built tensors: `compare(2b, b)` → cos 1, ratio 2, rel 1; orthogonal → cos 0;
   `compare(b, b)` → rel 0.
2. Quick run `run(gammas=(0.5,), draws=(64, 256, 1024), seeds=(0, 1), layers=(2,), nrep=8)` finishes in
   < 60 s, returns a JSON-serialisable dict (`json.dumps` works) with entries for both blocks' `c_v` and
   block 1's `value_embeds`, a `slope` in (−1.4, −0.6) for each, and `rel` at K=1024 < `rel` at K=64.
3. `nrep` independence of the mean: with γ=0.5, a single-seed sampled estimate at K=1024 with `nrep=1` and
   with `nrep=8` both have `cos > 0.9` vs `analytic_value_errors` (any chunking is the same estimator).
   You can build this with `setup` and the WP1 APIs directly.
4. Control discriminates: for γ=0.5 the control entry has `rel_vs_wrong_gamma > 2 · rel_vs_B`.
5. `main(['--quick', '--out', str(tmp_path / 'r.json')])` writes the file and the JSON has `passed` (bool)
   and `checks`.
