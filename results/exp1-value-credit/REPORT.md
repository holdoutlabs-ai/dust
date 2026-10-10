# Experiment 1: DUST with analytic local value credit (report, 2026-10-10)

Plan: `docs/PLAN-value-credit.md`. Branch `exp/value-credit`. Raw results, exact commands and hardware: `README.md` in
this directory. Variants: **A** = upstream sampled value / value-embedding credit; **B** = its exact expectation
`e_v[t] = Σ_s γ^(s−t) A[s,t] ĝ_s` with the recipe γ = 0.98 (`--value-credit analytic`); **C** = the same with γ = 1
(`--value-gamma 1.0`). B/C drop the `v`/`ve` local draws and do not redistribute them; everything else in DUST is unchanged.

## Conclusion

**B is A's mean**, to within sampling noise and to float precision where the true target is fed in (3.1, 3.2). At fixed
population, B raises cosine-to-backprop for `c_v` and `value_embeds` by 3–20×, and **C raises it a further ~2×** at both
checkpoints and both populations (3.3). Dropping the γ = 0.98 decay therefore does not just remove damping: it measurably
improves alignment of the value credit with the true gradient. No other parameter's estimate changes. In short training
at pop 256 (3.4), **B and C are no worse than A** (the plan's decision rule for B is met). Both have lower mean loss than A
(−0.015 and −0.017 val; −0.013 and −0.014 test), but with 3 seeds the difference is not significant: it is driven by one
seed (43) where A recovered worst from an early instability. **C vs B is a tie in loss** (−0.002 ± 0.011). The large cosine gain on
`v`/`ve` does not show up as a loss gain at this scale, plausibly because value parameters are a small share of the update
and the run is short. The compute saving is negligible: **< 1% per step**, because the skipped draws only recompute one block's attention.

What next, in order of expected value: (1) **reallocate the freed `v`/`ve` draws** (PLAN §4 follow-up; small at 28/step, so
pair it with the next item); (2) **gate credit with σ'(z)**, the same analytic step for the next-worst local site; (3) more seeds or
pop 1024 training to resolve the ~0.015 A-vs-B/C loss difference, if it matters for the decision; (4) keys/queries via the
softmax Jacobian, a larger change.

## 3.1: B equals A's mean (CPU, tiny config) — pass

K ∈ {64, 256, 1024, 4096} × 3 seeds × γ ∈ {0.98, 0.9, 0.5} × n_layer ∈ {2, 4}, covering short/full window and with/without VE.
- Squared error vs B falls as 1/K: slopes **−1.008 … −0.990**, with no error floor.
- Projection coefficient `p = ⟨A_K, B⟩/‖B‖² = cos·ratio`: max |p − 1| = **0.012** over all entries at K = 4096. This is direction and
  norm with no constant factor, as required by the uncentred `/draws /σ` scale.
- Wrong-γ control (γ = 0.5 draws vs B at γ = 1): rel ≈ 0.65 vs ≈ 0.02, cos ≈ 0.6 vs ≈ 0.99, so a transposed or absent decay is detectable.
- **Plan amendment (Noah, 2026-10-10):** the original absolute thresholds (cos > 0.99, |ratio − 1| < 3% at K = 4096) fail
  at γ ∈ {0.98, 0.9} from variance alone (rel ≈ 0.07 at γ = 0.98). The gate became the noise-corrected projection check.
  117/117 gating checks pass; the original thresholds are recorded as informational (23/54 met).
- 3.1 cannot resolve the mask edge (one key at lag = window); 3.2 pins it.

## 3.2: autograd pin (CPU, tiny config) — pass

C fed the true ∂L/∂o_i reproduces autograd's ∂L/∂v, ∂L/∂ve and the `c_v` / `value_embeds` weight gradients on every block
of n_layer = 2 and 4 (including VE on a short-window block): **cos = 1.00000000, |ratio − 1| ≤ 4e-8**. The Γ orientation at
γ = 0.5 and a window of 3 at t = 8 match an explicit-loop reference to 1e-5. Mutation tests (transposed Γ, flipped decay,
scale, gate, mask edge, missing norm/rotary) all fail the suite.

## 3.3: cosine to backprop at the paper config

Mean over 3 noise seeds × 4 batches of `train_1M.pt`. Init = `make_model(seed=42)`; bp = 61-step backprop checkpoint (val 5.94).

| cos | pop | A | B | C |
|---|---|---|---|---|
| `v`, init | 256 | 0.024 | 0.131 | **0.247** |
| `v`, init | 1024 | 0.098 | 0.311 | **0.525** |
| `v`, bp | 256 | 0.005 | 0.047 | **0.100** |
| `v`, bp | 1024 | 0.033 | 0.125 | **0.251** |
| `ve`, init | 256 | 0.003 | 0.048 | **0.169** |
| `ve`, init | 1024 | 0.012 | 0.123 | **0.405** |
| `ve`, bp | 256 | 0.001 | 0.022 | **0.067** |
| `ve`, bp | 1024 | 0.006 | 0.056 | **0.167** |

- `wte`, `lm_head`, `q`, `attn_out`, `mlp_in` and `mlp_out` are **bitwise identical** across A/B/C. Their estimates are drawn before
  the `v` draws.
- `gate`, `k` and `lambdas` differ only through shifted noise draws and are within noise. For these noise-dominated rows the 12
  records are only ~3 independent realizations, because the noise stream depends on the seed only (see README caveat).
- Whole-model cosine moves little (bp pop 256: 0.402 → 0.416), because `v`/`ve` are a small part of the gradient norm.

## 3.4: training, 1M tokens, pop 256, 3 seeds × 3 variants (rented 4× RTX 3090)

Tuned lr/momentum from `DEFAULT_CONFIGS` were not retuned per variant (a limitation, by design in round one). Seeds
42/43/44 share init and data order across variants. Every run's best validation loss is at its final step 61.

| variant | best val | test at best val |
|---|---|---|
| A | 6.0933 ± 0.0395 | 6.0794 ± 0.0378 |
| B | 6.0782 ± 0.0168 | 6.0668 ± 0.0141 |
| C | 6.0766 ± 0.0211 | 6.0657 ± 0.0197 |

| paired difference (val) | mean ± std | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| B − A | −0.0151 ± 0.0301 | −0.0025 | −0.0495 | +0.0068 |
| C − A | −0.0167 ± 0.0199 | −0.0111 | −0.0388 | −0.0001 |
| C − B | −0.0016 ± 0.0107 | −0.0086 | +0.0107 | −0.0069 |

(Test-set differences have the same signs: B − A −0.0126, C − A −0.0136, C − B −0.0010.) Seed 43 has an early excursion
in all variants (val at step 20: A 7.16, B 7.00, C 7.01), and A recovered least from it. Run wall-clock: A 1034–1050 s,
B/C 995–1026 s, on GPUs that shared the box with other jobs. Summary: `3.4-summary.json`; per-run files: `3.4/<variant>-s<seed>/`.

## Cost: time and memory

| | A | B | C |
|---|---|---|---|
| `DUST.step`, pop 256, 3090 (5 steps after warm-up) | 14.90 s | 14.87 s | 14.80 s |
| `estimate()`, pop 1024, 3090 | 54.2–55.9 s | 52.7–54.3 s | 52.7–54.3 s |
| peak, pop 256 (estimator / with `dust.py` eval) | 6.50 / 7.18 GiB | 6.49 / 7.18 GiB | same |
| peak, pop 1024, nrep 8 (incl. autograd grads for the comparison) | 11.98 GiB | 11.98 GiB | 11.98 GiB |

On the RTX 3080 Laptop (8 GB), pop 256 fits, at 6.5 GiB for the estimator and 7.18 GiB with eval, but the card throttles thermally
(84–86 °C, 42–87 s/step), which makes step-time comparisons there meaningless. The analytic path's own overhead at the paper config is
0.49 s and +0.9 GiB per step (reviewer measurement).

## Limitations

- 3 seeds, 1M tokens, one population for training; lr/momentum not retuned per variant.
- On CUDA the analytic path computes rotary+norm in fp32, while the model's forward runs in bf16. Measured effect: cos 0.999998.
- The cosine noise for `gate`/`lambdas` is shared across batches per seed (≈3 effective records).
- The freed draws are not reallocated, so B/C ran at slightly *lower* compute than A (28 fewer local draws per block per step).
