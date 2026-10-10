# E2-WP2 Staleness ladder: delayed gradients (+ E2-WP2t its tests)
owner: dust-dev (code), dust-test-writer (tests, in parallel)
interpreter: D:/Micromamba/micromambaenv/envs/dust/python.exe
test cmd: `D:/Micromamba/micromambaenv/envs/dust/python.exe -m pytest tests/test_delay.py -q`

## Goal

Stage 2.1 asks what a late gradient costs. Add to the backprop baseline: apply each step's gradient `d` steps late,
optionally compute it at extrapolated ("predicted") weights, and optionally make the delay depend on depth as in a pipeline.
With the defaults (`--grad-delay 0`), training is bitwise identical to the E2-WP1 code for every optimizer.

## Files

dust-dev writes: `baselines/decouple.py` (new) and minimal edits to `baselines/backprop.py` (flags and wiring through `train_step`).
test-writer writes: `tests/test_delay.py`.
do not touch: `baselines/optim.py` (E2-WP1, reviewed), `dust.py`, `model.py`, `diagnostics/`, other tests, docs, `data/`.

## Semantics (the contract)

Step t (t = 1, 2, …) uses batch t. Let `w_t` be the weights before step t's update, and `Δ_t = w_t − w_{t−1}` the last applied
update (`Δ_1 = 0`).

```
evaluation point  e_t = w_t + k · Δ_t     with k = d if --weight-prediction else 0   (per parameter, using that parameter's delay d)
g_t               = ∇L(batch t) at e_t    (one forward + backward; the reported train loss is the loss at e_t)
applied at step t : the gradient g_{t−d} computed d steps earlier; if t ≤ d, nothing is applied for that parameter
                    (its .grad is None, so the optimizer skips it; Muon, SGD and AdamW all skip None grads)
```

After computing g_t, the weights are restored to `w_t` before the optimizer step (weight prediction only moves the evaluation point).
Delays are per parameter, `delay_map: dict[name, int]`:
- `--delay-mode uniform`: every parameter has delay d.
- `--delay-mode depth` (pipeline-like: lower stages are staler): block i of L gets `round(d · (L − 1 − i) / (L − 1))`;
  `transformer.wte` and `value_embeds.{i}` follow their stage (wte: d; value_embeds.{i}: block i's delay);
  `lm_head`, `resid_lambdas` and `x0_lambdas` get 0.

```python
# baselines/decouple.py
def delay_map(model, d: int, mode: str) -> dict[str, int]
class DelayedGradients:
    def __init__(self, model, delays: dict[str, int], weight_prediction: bool)
    def evaluation_point(self) -> contextmanager   # inside: params moved to e_t (in place, no_grad); restored on exit
    def exchange(self) -> None                     # after backward: stash each param's .grad (clone) into its FIFO, then
                                                   # set .grad to the gradient due now (or None); records Δ bookkeeping
    def after_step(self) -> None                   # after optimizer.step(): Δ_{t+1} = w_{t+1} − w_t
```

`train_step(model, optimizers, x, y, delayed=None)`: if `delayed is None`, the code path is exactly the E2-WP1 one (no clone,
no extra ops), so d = 0 without weight prediction is bitwise unchanged. Otherwise, run forward/backward inside
`delayed.evaluation_point()`, then `delayed.exchange()`, `optimizer.step()` for each optimizer, and `delayed.after_step()`.
When `--grad-delay 0` and no `--weight-prediction` are given, `main()` passes `delayed=None`.
CLI: `--grad-delay INT` (default 0, ≥ 0), `--delay-mode {uniform,depth}` (default uniform), `--weight-prediction` (flag).
Record all three in `config.json` (new keys only). Memory: d gradient copies per parameter (fine at this model size).

## Read (only these)

- `baselines/backprop.py` (whole file), `baselines/optim.py` (signatures only), `model.py` `GPT.__init__` (parameter names),
  `diagnostics/tiny.py`, `tests/conftest.py`, `tests/test_optimizers.py` (style, and how modules are loaded by path).

## Rules that apply (CLAUDE.md)

- With default flags behaviour must be identical: same results bitwise for the same seed. `dust.py` and `model.py` are not touched.
- New mechanisms live in new modules (`baselines/decouple.py`), not inline sprawl.
- Expected values in tests come from math, an autograd reference or the unmodified baseline, never the code under test.

## Acceptance (test-writer: turn into tests; dev: make them pass)

1. **Default unchanged:** 3 steps of `train_step(..., delayed=None)` vs `DelayedGradients(d=0, uniform, no prediction)`, both bitwise
   equal to the E2-WP1 `train_step` without the `delayed` argument, for sgd, adamw and muon on the tiny CPU model.
2. **FIFO semantics against a hand reference:** tiny model, plain SGD (momentum 0, lr 0.1), d = 2, 5 steps on 5 different random batches.
   Reference: an explicit loop that computes g_t with autograd at the current weights, keeps a list, and applies `w -= lr · g_{t−2}`
   from step 3. Parameters match to 1e-6 after every step. Steps 1–2 leave the weights unchanged.
3. **Weight prediction:** SGD momentum 0, d = 2, prediction on. The forward of step t sees `w_t + 2·Δ_t` (capture it with a forward
   pre-hook on a block), the weights after `train_step` don't contain the extrapolation (restored), and the parameters match a hand
   reference that evaluates gradients at the extrapolated point.
4. **Depth map:** `delay_map(tiny_config(n_layer=4) model, d=6, 'depth')`: blocks 0..3 get 6, 4, 2, 0; wte 6; value_embeds.{1,3} get
   4 and 0; lm_head and both lambdas 0. Uniform mode gives d everywhere. Every parameter name has an entry.
5. **Mixed delays in one step:** depth mode with d = 2 on n_layer = 2: block 1 (delay 0) updates from step 1, while block 0 and wte
   (delay 2) stay unchanged until step 3.
6. **CLI:** a negative `--grad-delay` is a parse error. The flags reach `parse_args`.
