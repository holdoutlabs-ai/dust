# Experiment 1 results: analytic local value credit

One subdirectory or JSON file per stage (PLAN §3). Each entry records the exact command,
the commit hash, the hardware, and measured wall-clock / peak memory where relevant.
Interpreter: `D:/Micromamba/micromambaenv/envs/dust/python.exe` (torch 2.10.0+cu128), run from the repo root.

| Stage | File | Command | Code commit | Result |
|---|---|---|---|---|
| 3.1 regression | `3.1-regression.json`, `.log` | `$PY -m diagnostics.regression --out results/exp1-value-credit/3.1-regression.json` | 1f3b2e8 | **pass**, 117/117 gating checks |
| 3.2 autograd pin | `3.2-autograd-pin.json`, `.log` | `$PY results/exp1-value-credit/stage_3_2.py` (and `$PY -m pytest tests/test_analytic_autograd.py -q`: 6 passed) | 1f3b2e8 | **pass** |
| 3.3 cosine ladder | | | | not run (gated: data download, backprop checkpoint, GPU) |
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
