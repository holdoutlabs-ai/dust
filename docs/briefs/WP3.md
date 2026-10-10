# WP3 `diagnostics/cosine.py`: per-parameter cosine of DUST vs autograd (stage 3.3 tool)
owner: dust-dev (module), dust-test-writer (`tests/test_cosine.py`, in parallel)
interpreter: D:/Micromamba/micromambaenv/envs/dust/python.exe
test cmd: `D:/Micromamba/micromambaenv/envs/dust/python.exe -m pytest tests/test_cosine.py -q`

## Goal

A tool that measures how well each DUST variant estimates the true gradient: per parameter, cosine
and norm ratio of the DUST estimate against autograd on the same batch, aggregated by layer type
(paper Appendix G), over several noise seeds and batches, plus step wall-clock and peak memory.
Variants: **A** = `value_credit='sampled'`; **B** = `'analytic'`, γ from the recipe;
**C** = `'analytic'`, `value_gamma=1.0`.

## Files

dust-dev writes: `diagnostics/cosine.py`.
test-writer writes: `tests/test_cosine.py`.
do not touch: `dust.py`, `model.py` (WP1, done), `diagnostics/tiny.py`, `diagnostics/regression.py`
and `tests/test_regression.py` (WP2 is writing these now), `tests/conftest.py`, other tests, docs.

## Contract (APIs from WP1, already in `dust.py`)

```python
from dust import DUST, Recipe, recipe, make_model, load_sequences, GPTConfig
from diagnostics.tiny import tiny_config, tiny_recipe
recipe(tokens, population, value_credit='sampled', value_gamma=None) -> Recipe   # analytic: local v = ve = 0
tiny_recipe(draws_scale=1, nrep=2, gamma=0.98, ..., **overrides) -> Recipe       # overrides: value_credit=, value_gamma=
Recipe fields: writers, hidden, hubs, embedding, head, local, nrep, lr, momentum, gamma, value_credit, value_gamma
est = DUST(model, settings, seed=S); est.estimate(x, y) -> {param_name: Tensor}  # no_grad; SUMMED-loss scale
est.close()                                                                      # remove hooks (always, in finally)
```

`estimate` returns gradients of the **summed** token loss; divide by `valid_tokens = (y != -1).sum()`
to get the mean-loss gradient, which is what autograd gives for `model(x, y)` (default `reduction='mean'`).
For an analytic recipe with `local['v'] == local['ve'] == 0`, build it with
`dataclasses.replace(settings, value_credit='analytic', value_gamma=..., local={**settings.local, 'v': 0, 've': 0})`
(same as what `recipe()` does) so the tiny path matches the paper path.

## Module design

```python
VARIANTS = {'A': dict(value_credit='sampled', value_gamma=None),
            'B': dict(value_credit='analytic', value_gamma=None),
            'C': dict(value_credit='analytic', value_gamma=1.0)}

def layer_type(name) -> str
    # 'transformer.wte.weight'->'wte', 'lm_head.weight'->'lm_head', '*lambdas'->'lambdas',
    # 'value_embeds.*'->'ve', '*.attn.c_q.*'->'q', '.attn.c_k.'->'k', '.attn.c_v.'->'v',
    # '.attn.ve_gate.'->'gate', '.attn.c_proj.'->'attn_out', '.mlp.c_fc.'->'mlp_in', '.mlp.c_proj.'->'mlp_out'
    # unknown name -> raise ValueError (so a new parameter can't be silently dropped)

def variant_settings(base: Recipe, variant: str) -> Recipe

def autograd_gradients(model, x, y) -> {name: float32 Tensor}
    # model.requires_grad_(True); zero grads; with torch.autocast('cuda', bfloat16) on CUDA (nullcontext on CPU,
    # same as DUST.autocast): loss = model(x, y); loss.backward(); collect p.grad.float().clone();
    # then set grads to None and model.requires_grad_(False) (in finally).

def dust_gradients(model, settings, x, y, seed) -> ({name: float32 Tensor}, seconds, peak_bytes|None)
    # est = DUST(model, settings, seed); on CUDA: synchronize + reset_peak_memory_stats before,
    # synchronize after; time est.estimate(x, y) only; divide every tensor by valid_tokens; close in finally.

def compare(est, true) -> dict
    # per_param: {name: {cos, ratio=‖est‖/‖true‖, type}}
    # per_type: {type: {cos_mean, ratio_mean}} (mean over that type's parameters, as in Appendix G)
    # full: cos of the concatenation of all parameters (flattened, same name order)
    # value_subset: cos of the concatenation of the 'v' and 've' parameters
    # Treat a zero-norm true or est as cos = nan (don't crash). Pure function, no model needed.

def run(model, batches, variants, seeds, settings_base) -> dict
    # for each batch: true = autograd_gradients once; for each variant, seed: est = dust_gradients; record compare().
    # Same batches and the same noise seeds across variants (paired). Aggregate per variant:
    # per_param and per_type mean ± std over all (seed, batch) records; time and peak memory mean.

def main(argv=None)
```

CLI (PLAN §2.2):

```
python -m diagnostics.cosine --config {tiny,paper} [--checkpoint PATH] [--variants A,B,C]
    [--population 256] [--tokens 1M] [--seeds 0,1,2] [--batches 4] [--batch-size 8]
    [--data data/train_1M.pt] [--nrep INT] [--device cpu|cuda] --out results/exp1-value-credit/*.json
```

- `--config paper`: `GPTConfig()`, `recipe(tokens, population)` as the base recipe. `--config tiny`:
  `tiny_config()`, `tiny_recipe()` (population/tokens ignored, record that).
- Model: `make_model(config, device, 42)`; if `--checkpoint`, `model.load_state_dict(torch.load(path,
  map_location=device, weights_only=True))` (the format `dust.py` and `baselines/backprop.py` save:
  a plain `state_dict` of CPU tensors). For tiny without a checkpoint, randomize `ve_gate.weight`
  with `normal_(0, 0.5)` under `torch.manual_seed(0)` (zero at init makes gate = 1); record it.
- Batches: with `--data`, `rows = load_sequences(path, batches * batch_size, sequence_len=2048)`, crop to
  `rows[:, :config.sequence_len + 1]`, batch j = `rows[j*bs:(j+1)*bs]`, x = `[:, :-1]`, y = `[:, 1:]`.
  Without `--data`, random tokens `torch.randint(0, 4096, (bs, seq+1), generator=Generator().manual_seed(1000 + j))`.
- `--nrep` overrides `Recipe.nrep` (PLAN §3.3: if it OOMs on 8 GB, drop nrep to 1; it only changes chunking).
- `--device` defaults to cuda if available. On CUDA call `torch.set_float32_matmul_precision('high')` as `dust.py` does.
- Output JSON: `{"args": {...}, "command": " ".join(sys.argv), "torch": version, "device": name,
  "recipe": asdict(base), "variants": {V: {"settings": asdict, "per_param": {name: {cos_mean, cos_std,
  ratio_mean, ratio_std}}, "per_type": {...same...}, "full": {mean, std}, "value_subset": {mean, std},
  "seconds_mean":.., "peak_gib_max":..}}, "records": [ {variant, seed, batch, per_type, full, value_subset} ]}`.
  Print a compact table: rows = layer type + full + value_subset, columns = variants, `cos mean±std`.

## Read (only these)

- `dust.py`: `Recipe`, `recipe()`, `DUST.__init__`, `autocast`, `estimate`, `step`, `make_model`, `load_sequences`.
- `model.py`: `GPT.__init__` and `forward` (parameter names, `loss_reduction`).
- `diagnostics/tiny.py`, `tests/conftest.py`.

## Rules that apply (CLAUDE.md)

- No changes to `dust.py` or `model.py`. Note that DUST divides by `valid_tokens`, i.e. it estimates the gradient of the
  **mean** loss, the same as autograd with the default `reduction='mean'`. Compare like with like.
- Tests run on CPU with the tiny config, no network, no data files (use random token batches), fixed
  seeds, and finish in minutes. Expected values come from math or autograd, never from the code under test.
- Make the two stochastic knobs explicit: the DUST noise seed and the batch.

## Acceptance (test-writer: turn into tests in `tests/test_cosine.py`)

1. `compare` on hand-built dicts: identical → cos 1, ratio 1; `est = 2·true` → cos 1, ratio 2;
   `est = −true` → cos −1; per_type averages per-parameter cos correctly for a two-parameter type;
   `full` is the cosine of the concatenation, which in general is **not** the mean of per-param cos (build a case
   where they differ); a zero tensor gives nan, not an exception.
2. `layer_type` maps every parameter of the tiny model (`n_layer=2` and `4`) to one of the 11 types, and
   raises ValueError on an unknown name.
3. `autograd_gradients` equals a hand-written autograd of `model(x, y)` (mean CE) on the tiny model, and
   leaves `requires_grad` False on every parameter and `.grad` None afterwards.
4. **Paired variants share what they should** (tiny, same seed, same batch, via `dust_gradients`): A and B
   give **bitwise equal** estimates for every parameter computed before the local-credit loop
   (`attn.c_proj`, `mlp.c_proj`, `mlp.c_fc`, `wte`, `lm_head`) and for `attn.c_q` (kind `'q'` precedes `'v'`);
   B and C are **bitwise equal** on every parameter except `attn.c_v` and `value_embeds.*`, and differ on those.
   (This is PLAN §3.3's "every other parameter unchanged" check, made exact.)
5. End-to-end tiny: `main(['--config', 'tiny', '--seeds', '0,1', '--batches', '2', '--device', 'cpu',
   '--out', str(tmp_path/'c.json')])` finishes in < 2 min, writes JSON with variants A, B, C, finite
   `full` cosines, and per_type keys covering all 11 types. Positive sanity: `full` cos mean > 0 for every variant.
