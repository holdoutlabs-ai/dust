# Experiment 1 — Analytic local value credit in DUST

Status: plan, not yet implemented. Target repo: `holdoutlabs-ai/dust` (fork of `qlabs-eng/dust`).
Work on a feature branch (e.g. `exp/value-credit`); do not touch `main`.

## 0. Goal and hypothesis

DUST estimates the error at each attention block's **output** `ĝ_s` by perturbation, then estimates the error at the **values** `v_t` by a second round of perturbation inside the block (`DUST.local_errors`, kind `'v'`), crediting output position `s` to value position `t` with `γ^(s−t)`.

In exact arithmetic the current value estimator has expectation

```
E[ê_t] = Σ_{s≥t} γ^(s−t) · A[s,t] · ĝ_s          (per head)
```

where `A[s,t]` is the attention weight from query position `s` to key position `t`. That expression is a single matmul over cached activations. Computing it directly removes the sampling noise of the inner perturbation (the paper's Appendix G shows V and VE are among the worst-estimated layer types: fitted `c` = 174k and 495k draws) and frees the draws that currently pay for it.

**Hypothesis:** replacing the sampled value/value-embedding credit with the deterministic expression (a) does not change what is being estimated, (b) raises cosine-to-backprop for `c_v` and `value_embeds` at fixed compute, and (c) gives equal or lower loss in short training runs.

Three variants, everything else in DUST untouched:

| Variant | Value & VE credit | Purpose |
|---|---|---|
| **A** | existing sampled estimator, `γ` as in recipe (0.98) | baseline |
| **B** | deterministic `Σ_s γ^(s−t) A[s,t] ĝ_s`, same `γ` | removes inner sampling noise only; its mean must equal A's → regression test |
| **C** | deterministic with `γ = 1`: `Aᵀ ĝ` | also removes the decay |

Queries, keys, gates stay exactly as they are. (Gate would need `σ'(z)`; keys/queries need the softmax Jacobian — separate experiments.)

Name this honestly in code and write-ups: "DUST with analytic local value credit". It still uses perturbation for all global credit; it replaces one *local* sampled estimate with the derivative of a linear map, which is the same category of step the paper already takes in Eq. 3 (`Ĝ_W = Σ_t ĝ_t x_tᵀ`).

## 1. The math to implement

Per block `i`, per batch element `b`, per head `h`. Notation follows `DUST.local_errors`:

- `q, k, v` — cached outputs of `attn.c_q/c_k/c_v`, shape `[b, t, h, d]`
- `qp, kp` — `norm(apply_rotary_emb(·))` of `q`, `k` (exactly as `processed()` does)
- `z` — cached `ve_gate` output `[b, t, h]`, only on blocks where `has_ve(i, n_layer)`
- `ve` — cached `value_embeds.{i}` output `[b, t, h, d]`
- `gate = 2·sigmoid(z)` `[b, t, h]`
- `values = v + gate[..., None] · ve`
- `ĝ` — `targets[i]` from `DUST.step`, the attention-output error at the `o@{i}` site, viewed `[b, t, h, d]`
- `window` — `model.window_sizes[i][0]`

Attention matrix (must match `model.attention` exactly — `F.scaled_dot_product_attention` default scale `1/√d`, boolean mask `0 ≤ s−t ≤ window`):

```
S[b,h,s,t] = (qp[b,s,h,:] · kp[b,t,h,:]) / sqrt(d)
S[mask == False] = -inf
A = softmax(S, dim=-1)                       # [b, h, s, t]
```

Decay matrix (the code already builds this as `credit[t, s]`; note the transposed index convention):

```
Γ[s,t] = γ^(s−t) if 0 ≤ s−t ≤ window else 0
```

Errors:

```
M      = A * Γ[None, None]                                        # [b, h, s, t]
e_v    = einsum('bhst,bshd->bthd', M, ĝ)                          # [b, t, h, d]
e_ve   = e_v * gate[..., None]                                    # only where VE exists
```

Variant C is the same with `Γ` replaced by the plain window mask (γ=1).

Return `e_v.reshape(b, t, n_embd)` under site key `transformer.h.{i}.attn.c_v` and `e_ve.reshape(b, t, n_embd)` under `value_embeds.{i}`, so the existing `DUST.reassemble` turns them into weight gradients unchanged.

**Scale check.** `local_errors` for `'v'` does *not* center rewards and divides by `draws` and `σ`; its expectation is exactly the expression above with no shrinkage factor. So B must match A's mean in **both direction and norm**. If the norms disagree by a constant factor, the implementation is wrong, not the theory.

## 2. Code changes (keep them minimal and flag-gated)

### 2.1 `dust.py`

1. Add a setting to `Recipe`: `value_credit: str = 'sampled'` with values `'sampled' | 'analytic'`, and `value_gamma: float | None = None` (None → use `settings.gamma`; for variant C pass `1.0`).
2. Add `DUST.analytic_value_errors(targets, gamma) -> dict[site, error]` implementing §1. Build `A` one block at a time, free it before the next block. On GPU materialize in fp32 (`8×8×2048×2048×4 B = 1 GiB` per block at the paper config); if that is too much for the card, loop over heads.
3. In `DUST.step`, where `r.local` is iterated:
   - if `value_credit == 'analytic'`: call `analytic_value_errors` for sites `c_v` and `value_embeds.*`, and **skip** kinds `'v'` and `'ve'` in the `local_errors` loop.
   - Guard `local_errors` against `draws == 0` (currently `errors[i] / draws` would divide by zero).
4. In `recipe()`, when `value_credit == 'analytic'`, set `local['v'] = local['ve'] = 0`. Do **not** redistribute those draws elsewhere in this experiment — the comparison is at *reduced* compute for B/C; redistribution is a follow-up.
5. Expose `--value-credit {sampled,analytic}` and `--value-gamma FLOAT` in `main()`, record them in `config.json`.

Everything else in `dust.py` stays byte-identical. Diff should be small enough to review in one sitting.

### 2.2 New `diagnostics/` package (no changes to training code)

```
diagnostics/
  __init__.py
  tiny.py          # small GPTConfig + matching Recipe for CPU tests
  cosine.py        # per-parameter cosine of DUST estimate vs autograd on one batch
  regression.py    # A-mean ≈ B test
```

**`tiny.py`** — `GPTConfig(sequence_len=128, vocab_size=4096, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)`. Keep `vocab_size=4096` so real token data and the 256-column head slabs work unchanged (`4096 % 256 == 0`). Provide `tiny_recipe(draws_scale)` with every count a multiple of `nrep` (use `nrep=2`) and `head` a multiple of `nrep·16` slabs; e.g. writers/hidden `[32, 16]`, hubs `[16, 8]`, embedding 16, head 512, local `q=4, v=32, gate=8, k=16, ve=16`.

**`cosine.py`** — given a model, a batch `(x, y)`, and a `DUST` instance:

1. Run the DUST estimate but **capture `parameter.grad` instead of stepping** (easiest: pass a dummy optimizer whose `step()` is a no-op, or refactor `DUST.step` to return the `gradients` dict before the optimizer call — a tiny refactor, acceptable).
2. `model.requires_grad_(True)`, zero grads, `loss = model(x, y)` under the same autocast, `loss.backward()`. Note DUST divides by `valid_tokens`, i.e. it estimates the gradient of the **mean** loss — same as autograd on the default `reduction='mean'`. Compare like with like.
3. Report per parameter: `cos`, `‖ĝ‖/‖g‖`, and also per layer-type means (MLP in/out, attn out, q, k, v, gate, ve, wte, lm_head, lambdas) to match the paper's Appendix G table.
4. Also report cosine of the **full concatenated** gradient and of the subset `{c_v, value_embeds}`.
5. CLI: `python -m diagnostics.cosine --config {tiny,paper} --checkpoint PATH --value-credit ... --value-gamma ... --population N --seed S --data data/train_1M.pt --out results/*.json`.

Make the two stochastic knobs explicit: the DUST noise seed and the batch. Cosine on one batch is noisy; average over ≥4 batches and ≥3 noise seeds and report mean ± std.

**`regression.py`** — the B-vs-A-mean test (§3.1). Runs on CPU with the tiny config in well under a minute.

### 2.3 `model.py`

No changes. If `attention()` is ever modified, the analytic `A` must be kept in sync; add a one-line comment in `attention()` pointing at `analytic_value_errors`.

## 3. Tests, in order. Do not proceed to the next stage until the previous one passes.

### 3.1 Regression: B equals A's mean (CPU, tiny config, minutes)

For fixed model, batch, and `targets`:

- Run variant **A**'s `local_errors('v', draws=K)` with K ∈ {64, 256, 1024, 4096} (several noise seeds each).
- Run **B** once.
- Assert, for both `c_v` and `value_embeds` sites, as K grows: `cos(A_K, B) → 1` (expect > 0.99 at K=4096 on the tiny config) and `‖A_K‖/‖B‖ → 1` (within ~3%). Also check `‖A_K − B‖²` scales ≈ `1/K`.
- Repeat with `γ ∈ {0.98, 0.9, 0.5}` to make sure the decay is applied on the correct axis (a transposed `Γ` passes at γ≈1 and fails at γ=0.5).
- Repeat with `window = sequence_len // 2` on a block with the short window.

If this fails, nothing downstream is meaningful. Fix before continuing.

**Amendment (2026-10-10, approved by Noah).** The first full run met every bias-sensitive check but missed the
absolute per-seed thresholds above at γ ∈ {0.98, 0.9} (min cos ≈ 0.965, ratio ≈ 1.035 at K=4096): the sampled
estimator's variance grows with the number of lags the decay sums, so at γ near 1 K=4096 is not enough draws for
cos > 0.99 regardless of correctness. Since `E‖A_K‖² = ‖B‖² + E‖A_K − B‖²`, noise alone predicts
`ratio² − rel = 1` and `cos = 1/√(1 + rel)` (rel = `‖A_K − B‖²/‖B‖²`). (Clarified the same day after review:
since `rel = ratio² − 2·cos·ratio + 1` identically, the first check is `|p − 1| < 0.025` for the projection
coefficient `p = cos·ratio = ⟨A_K, B⟩/‖B‖²`, i.e. the component along B is unbiased; it trips on a ≥3% scale error.
The second is a weaker function of the same numbers. A bias orthogonal to B leaves p = 1 and is caught by the
slope/rel-decreasing checks, as an error floor. 3.1 cannot resolve the mask edge (one key at lag = window), which
3.2 pins.)
The 3.1 pass rule is therefore, for every (config, γ, site), per seed at the largest K:
`|ratio² − rel − 1| < 0.05` and `|cos − 1/√(1+rel)| < 0.003`; the slope of log rel vs log K in (−1.25, −0.75);
rel decreasing in K; and the wrong-γ control (γ=0.5 data vs B at γ=1) has rel > 2× rel vs B.
Raw cos and ratio are still reported, and the original thresholds are recorded as informational checks.

### 3.2 Autograd sanity for the analytic path (CPU, tiny)

Independent of DUST: set `targets[i]` to the **true** `∂L/∂o_i` from autograd (hook on `attn.c_proj` input). Then variant **C** must reproduce autograd's `∂L/∂v` and `∂L/∂ve` to float precision (`cos > 0.9999`, norms equal). This pins down the `A` construction, scale, mask, and the gate multiplier with no sampling noise anywhere.

### 3.3 Cosine ladder at the paper config (GPU)

Hardware: a 24 GB card is comfortable. On the **3080 8 GB** first run `--steps 1 --population 256` under `torch.cuda.max_memory_allocated()` logging; if it OOMs, drop `nrep` to 1 for the diagnostic (it only changes chunking, not the estimator). Do **not** assume it fits.

Checkpoints: (i) initialization (`make_model(seed=42)`), and (ii) a backprop-trained checkpoint — run `python baselines/backprop.py --tokens 1M --output runs/bp-1m` (61 steps, cheap) and use its `best.pt`. The paper measured on backprop-trained checkpoints; both are informative.

Data: `python prepare_data.py --tokens 1M` (streams FineWeb from Hugging Face; needs network and a while). Use batches from `train_1M.pt`.

For each checkpoint × variant {A, B, C} × population {256, 1024, 4096 if affordable} × ≥3 noise seeds × ≥4 batches, record the Appendix-G-style table. Primary readouts:

- `cos` for `c_v` and `value_embeds`: expect **B > A**, and **B ≈ A's limit**. Whether **C > B** or **C < B** is the actual scientific question — it tells us whether the decay has been compensating for something or just damping.
- `cos` for *every other* parameter: must be **unchanged** between A/B/C within noise (they share `targets`; if anything else moves, there's a bug or an unintended coupling).
- Wall-clock per `DUST.step` for A vs B/C at matched population (B/C skip the `v`/`ve` draws; report the saving as measured, not estimated).

### 3.4 Short training comparison (GPU, the real test)

Cosine is a candidate filter, not the objective — the paper notes higher cosine on one batch does not always lower loss.

- Config: `--tokens 1M --population 256` (single GPU recipe), 3 seeds × 3 variants = 9 runs. Time one run with `--steps 2` first and extrapolate before launching all nine.
- Keep the tuned `lr/momentum` for population 256 from `DEFAULT_CONFIGS`; do **not** retune per variant in this round (that's a confound; note it as a limitation).
- Readouts: `result.json` → `best_val`, `test_at_best_val`; the val curve from `metrics.jsonl`. Report mean ± std over seeds and the paired per-seed differences (B−A, C−A, C−B), since seeds share the data order and model init.
- Decision rule: B should be ≥ A (no worse) — it's the same estimator with less noise. If B is clearly worse than A, suspect the implementation before the theory. C vs B decides whether γ<1 earns its keep.

If results are promising at 256, repeat at 1024 on a 24 GB card (friend's 3090s or a rental). Only then consider the paper's 16k ladder.

## 4. What NOT to do in this experiment

- Don't touch gate, key, or query credit.
- Don't reallocate the freed `v`/`ve` draws to other sites (follow-up experiment).
- Don't change σ, centering, noise distribution, or the writer-pair grouping (separate experiments on "ceilings").
- Don't retune `lr`/`momentum` per variant in round one.
- Don't rent GPUs before 3.1–3.3 are done and the per-step time at pop 256 has been measured on the 3080.
- Don't add partial-forward caching here (its measured saving for this allocation is ~15–20%, and it would confound timing comparisons).

## 5. Deliverables

1. Branch with the `dust.py` flag + `diagnostics/` package, each stage's results committed as JSON under `results/exp1-value-credit/` with the exact command in a `README.md` next to them.
2. A short write-up (`results/exp1-value-credit/REPORT.md`) with: the regression plot/table (3.1), the autograd pin (3.2), the cosine tables at init and the backprop checkpoint (3.3), the training table with paired differences (3.4), measured step time A vs B/C, and peak memory on the 3080.
3. A one-paragraph conclusion answering: Is B ≡ A in the mean? Does C beat B? Does either beat A in training at pop 256? What would we try next (reallocate freed draws / gate with `σ'` / keys)?

## 6. Reference: where things live in the current code

- `dust.py :: DUST.local_errors` — the sampled inner estimator; `kind in ('v','ve')` is what B/C replace. Its `credit` tensor is `Γᵀ`.
- `dust.py :: DUST.step` — builds `targets[layer]` (the `o@{layer}` errors, averaged over ranks) and then loops `r.local`.
- `dust.py :: DUST.reassemble` — error × cached input → weight gradient; reuse as is.
- `dust.py :: recipe()` — per-population draw counts; `local = {'q','v','gate','k','ve'}`.
- `model.py :: attention` — SDPA with boolean window mask; the analytic `A` must match it.
- `model.py :: CausalSelfAttention.forward` — shows `values = v + gate·ve` and that RoPE+norm are applied to `q,k` *before* attention.
- `baselines/backprop.py` — produces the backprop-trained checkpoint for 3.3.
