# Experiment 1 results: analytic local value credit

One subdirectory or JSON file per stage (PLAN §3). Each entry records the exact command,
the commit hash, the hardware, and measured wall-clock / peak memory where relevant.
Interpreter: `D:/Micromamba/micromambaenv/envs/dust/python.exe` (torch 2.10.0+cu128), run from the repo root.

| Stage | File | Command | Code commit | Result |
|---|---|---|---|---|
| 3.1 regression | `3.1-regression.json`, `.log` | `$PY -m diagnostics.regression --out results/exp1-value-credit/3.1-regression.json` | 1f3b2e8 | **pass**, 117/117 gating checks |
| 3.2 autograd pin | `3.2-autograd-pin.json`, `.log` | `$PY results/exp1-value-credit/stage_3_2.py` (and `$PY -m pytest tests/test_analytic_autograd.py -q`: 6 passed) | 1f3b2e8 | **pass** |
| 3.3 cosine ladder | `3.3-cos-{init,bp1m}-pop{256,1024}.json` (+ `.log`) | see §3.3 below | 72d06bf (256), 3d55c10 (1024) | **done**: pop 256 on the 3080, pop 1024 on the rental |
| 3.4 training | | | | not run (gated) |

## 3.1: B equals A's mean (2026-10-10)

Hardware: CPU, Intel i7-11800H, float32, tiny config (`diagnostics/tiny.py`), n_layer ∈ {2, 4}, nrep = 8
(chunking only), K ∈ {64, 256, 1024, 4096}, 3 seeds, γ ∈ {0.98, 0.9, 0.5}. Wall-clock 121.7 s.

Pass rule as amended in PLAN §3.1 on 2026-10-10 (approved by Noah): per seed at K = 4096,
`|ratio² − rel − 1| < 0.05` (≡ projection `p = cos·ratio` within 0.025 of 1) and `|cos − 1/√(1+rel)| < 0.003`;
slope of log rel vs log K in (−1.25, −0.75); rel decreasing in K; wrong-γ control rejected.

- Max `|p − 1|` over all 81 (config, γ, site, seed) entries at K = 4096: **0.012**. Slopes: **−1.008 … −0.990**.
  Note that the 81 entries share three noise seeds per (seed, K, γ, kind) cell, so they are not 81 independent samples.
- Control (γ = 0.5 data against B at γ = 1): cos ≈ 0.57–0.62, rel ≈ 0.62–0.69, against cos ≈ 0.99 and rel ≈ 0.02 for the correct γ.
- The original absolute thresholds (cos > 0.99, |ratio − 1| < 3%) are met in 23/54 cases: all fail at γ = 0.98 and γ = 0.9,
  because the sampled estimator's variance grows with the number of lags the decay sums (rel ≈ 0.07 at γ = 0.98 vs 0.02 at γ = 0.5).
- Limits: 3.1 cannot resolve the mask edge (one key at lag = window carries ~γ^window/window weight); 3.2 pins it.
  Reviewer mutation tests on this rule: a ≥3% scale error, a missing or doubled ve gate, and a transposed contraction all fail it;
  a ≤2% scale error may pass.

## 3.2: autograd pin (2026-10-10)

Hardware: CPU, float32, tiny config with n_layer = 2 and 4, randomized `ve_gate`. Variant C (γ = 1) fed the true
∂L/∂o_i (summed loss) from autograd. Every `c_v` and `value_embeds` site, including VE on a short-window block (L4 b1):
error cos = 1.00000000 and ratio within 4e-8 of 1, for the per-token errors and the reassembled weight gradients.
Criterion: cos > 0.9999, |ratio − 1| < 1e-4. The window-edge and γ-orientation cases are pinned in
`tests/test_analytic_autograd.py::test_gamma_orientation_against_dense_loops` (t = 8, window 3, γ = 0.5, dense loops).

## 3.3: cosine ladder at population 256 (2026-10-10, RTX 3080 Laptop)

```
$PY -m diagnostics.cosine --config paper --variants A,B,C --population 256 --seeds 0,1,2 --batches 4 \
    --data data/train_1M.pt --device cuda --out results/exp1-value-credit/3.3-cos-init-pop256.json
$PY -m diagnostics.cosine --config paper --checkpoint runs/bp-1m/best.pt --variants A,B,C --population 256 \
    --seeds 0,1,2 --batches 4 --data data/train_1M.pt --device cuda --out results/exp1-value-credit/3.3-cos-bp1m-pop256.json
```

Batches are the first 32 rows of `train_1M.pt` (4 × 8). The noise seeds are 0, 1, 2. Each estimate is paired across variants, so 12 records per variant.
The backprop checkpoint is `runs/bp-1m/best.pt` (val 5.9395; see `3080-smoke/`). Wall-clock: 1486 s (init) and 1320 s (bp).

Cosine to autograd, mean ± sample std over 12 records:

| type | init A | init B | init C | bp A | bp B | bp C |
|---|---|---|---|---|---|---|
| v | 0.024 ± 0.007 | 0.131 ± 0.009 | **0.247 ± 0.010** | 0.005 ± 0.006 | 0.047 ± 0.017 | **0.100 ± 0.022** |
| ve | 0.003 ± 0.003 | 0.048 ± 0.006 | **0.169 ± 0.009** | 0.001 ± 0.004 | 0.022 ± 0.005 | **0.067 ± 0.014** |
| {v, ve} concat | 0.019 ± 0.006 | 0.130 ± 0.010 | **0.255 ± 0.010** | 0.006 ± 0.004 | 0.075 ± 0.012 | **0.184 ± 0.029** |
| full | 0.925 | 0.926 | 0.925 | 0.402 | 0.416 | 0.415 |
| gate | 0.038 ± 0.086 | 0.001 ± 0.065 | = B | −0.001 ± 0.056 | 0.027 ± 0.036 | = B |
| k | 0.001 ± 0.002 | 0.001 ± 0.003 | = B | 0.001 ± 0.004 | −0.000 ± 0.004 | = B |
| lambdas | 0.509 ± 0.134 | 0.579 ± 0.097 | = B | 0.346 ± 0.145 | 0.481 ± 0.155 | = B |
| wte, lm_head, q, attn_out, mlp_in, mlp_out | identical across A/B/C (bitwise; computed before the v draws) | | | | | |

- **B > A** on `v` and `ve` at both checkpoints, by a 5–20× margin. At pop 256 the sampled `v` estimate uses 20 draws per block
  and `ve` uses 8, so A is far from its limit B (3.1 shows B is that limit).
- **C > B** at both checkpoints, roughly 2× on `v` and 2–3× on `ve`. The γ = 0.98 decay *lowers* alignment with the true
  gradient at fixed (exact) local credit. Whether that translates into loss is what 3.4 tests.
- Other parameters: only `gate`, `k` and `lambdas` differ between A and B. Their noise draws shift because A consumes the `v` draws first.
  All three differences are within noise (largest: bp lambdas, ~2 SE with 12 records; a 12-seed tiny check of gate showed A 0.091 vs B 0.064, SE 0.037).
- Step time (`estimate()` only, discarded warm-up, throttled laptop): init A 41.9 s, B 39.1 s, C 39.4 s; bp A 36.9 s, B 34.9 s,
  C 34.9 s, which puts B/C about 5–7% below A. Peak 6.40 GiB, the same for all variants (this includes the autograd gradients held for comparison).

## 3.3: population 1024 (2026-10-10, rented 4× RTX 3090 24 GB, Ubuntu 24.04, torch 2.10.0+cu128, code 3d55c10)

Same commands as at pop 256 with `--population 1024` (nrep 8, the paper's chunking); job list in `rental/queue-all.txt`.
Peak 11.98 GiB per variant. `estimate()`: init A 54.2 s, B 52.7 s, C 52.7 s; bp A 55.9 s, B 54.3 s, C 54.3 s (−3%).

| type | init A | init B | init C | bp A | bp B | bp C |
|---|---|---|---|---|---|---|
| v | 0.098 ± 0.009 | 0.311 ± 0.014 | **0.525 ± 0.015** | 0.033 ± 0.011 | 0.125 ± 0.031 | **0.251 ± 0.042** |
| ve | 0.012 ± 0.003 | 0.123 ± 0.010 | **0.405 ± 0.016** | 0.006 ± 0.003 | 0.056 ± 0.007 | **0.167 ± 0.024** |
| {v, ve} concat | 0.077 ± 0.008 | 0.302 ± 0.014 | **0.532 ± 0.016** | 0.043 ± 0.010 | 0.196 ± 0.035 | **0.440 ± 0.054** |
| full | 0.974 | 0.974 | 0.974 | 0.669 | 0.674 | 0.674 |
| gate | 0.135 ± 0.046 | 0.077 ± 0.050 | = B | 0.024 ± 0.055 | 0.014 ± 0.050 | = B |
| k | 0.009 ± 0.002 | 0.009 ± 0.003 | = B | 0.009 ± 0.006 | 0.006 ± 0.006 | = B |
| lambdas | 0.503 ± 0.163 | 0.641 ± 0.102 | = B | 0.560 ± 0.183 | 0.530 ± 0.171 | = B |

**Caveat on gate / k / lambdas (all of 3.3).** DUST's noise stream depends only on the seed, so for a given seed the 4
batches reuse identical draws. For the noise-dominated parameters (gate: 8 local draws; lambdas: 8 ES draws) the 12 records
are effectively 3 independent noise realizations, and the ± std understates the uncertainty. The A–B differences in these
rows flip sign across checkpoints (lambdas: B > A at init, A > B on bp at pop 1024) and are consistent with noise. The estimator
for these parameters is identical in A and B with independent draws (bitwise-verified on CPU), and a 12-independent-seed tiny check
of gate gave A 0.091 vs B 0.064 (SE 0.037).

## Step time on the rental (`rental/step_time-*.json`)

`DUST.step` at pop 256, paper config, A/B/C each on its own 3090 at the same time, 1 discarded warm-up + 5 timed steps:
A **14.90 s**, B **14.87 s**, C **14.80 s** (spread within a variant ≤ 0.1 s); peak 6.50 / 6.49 / 6.49 GiB. **B/C save < 1%** per step:
the skipped `v`/`ve` draws only recompute one block's attention. The laptop's apparent 5–7% saving was thermal noise.
`dust.py --steps 2` gate run: 45 s wall, step 1 at 14.9 s, peak 7.18 GiB (`rental/smoke-A.log`).
