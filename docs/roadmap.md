# Roadmap: Experiment 1, analytic local value credit

Plan: `docs/PLAN-value-credit.md`. Branch: `exp/value-credit`.
Status values: `todo | in-progress | review | done`.
Interpreter: `D:/Micromamba/micromambaenv/envs/dust/python.exe`.

## Build

| WP | Name | Owner | Depends | Files | Status | Review / rounds |
|---|---|---|---|---|---|---|
| 0 | Setup: env, CLAUDE.md, agents, tiny config, conftest, smoke test | dust-lead | – | CLAUDE.md, .claude/agents/*, diagnostics/{__init__,tiny}.py, tests/{conftest,test_smoke}.py | done | – |
| 1 | `dust.py` flag: Recipe fields, `analytic_value_errors`, step wiring, `draws==0` guard, `recipe()`, CLI + config.json, `estimate()` split for cosine; `model.py` sync comment | dust-dev | 0 | dust.py, model.py | done | pass, 1 round (reviewer: in-place scores, lead applied) |
| 1t | Tests for WP1: autograd pin (stage 3.2), sampled path unchanged, flag/CLI plumbing | dust-test-writer | 0 | tests/test_analytic_autograd.py, tests/test_value_credit_flag.py | done | pass (mutation-checked: Γᵀ, flipped γ, scale, gate, mask, norm, rotary all caught) |
| 2 | `diagnostics/regression.py` (stage 3.1) + pytest wrapper | dust-dev / dust-test-writer | 1 | diagnostics/regression.py, tests/test_regression.py | done | pass, 1 round; 3.1 pass rule amended (Noah, 2026-10-10); lead applied: evaluate() + gate test, K%nrep guard, independent noise per (seed,K,γ,kind) |
| 3 | `diagnostics/cosine.py` (per-parameter cosine vs autograd, layer-type table, CLI) | dust-dev | 1 | diagnostics/cosine.py, tests/test_cosine.py | done | pass, 1 round (lead applied: CUDA warm-up before timing, arg choices/validation, std/peak notes) |

Waves: {1, 1t} → review → {2, 3} → review.

## Stages (PLAN §3), hard-gated in order

| Stage | What | Where | Gate | Status | Result |
|---|---|---|---|---|---|
| 3.1 | B equals A's mean (K ∈ {64…4096}, γ ∈ {0.98, 0.9, 0.5}, short window) | CPU | – | todo | |
| 3.2 | C with true ∂L/∂o reproduces autograd ∂L/∂v, ∂L/∂ve (cos > 0.9999) | CPU | – | todo | |
| 3.3 | Cosine ladder, init + backprop checkpoint, A/B/C × pop × seeds × batches | GPU 3080 8 GB | Noah: data download, bp checkpoint run | todo | |
| 3.4 | Training 3 variants × 3 seeds, 1M tokens, pop 256 | GPU | Noah: after `--steps 2` timing | todo | |

## Notes

- Tiny config (`n_layer=2`): block 0 has the short window (64) and no VE; block 1 has the full window
  (128) and VE. No block has both. Tests that need VE on a short-window block use
  `tiny_config(n_layer=4)` (blocks 1 and 3 have VE; block 1 is short).
- `ve_gate.weight` is zero at init, so gate = 1 everywhere. The `tiny_model` fixture randomizes it;
  any test of the `ve` gate multiplier must do the same.
- Windows: no NCCL, single GPU only. Upstream targets Linux.
- WP1 (2026-10-10): on CUDA the analytic path computes rotary+norm on fp32 q/k (the brief requires fp32);
  `local_errors.processed` rounds qp/kp to bf16, and the model's forward is bf16 throughout. Measured
  at the paper config on the 3080 (reviewer): cos 0.999998 and norm within 1e-5 against the bf16-rounded version. Negligible;
  keep this in mind when reading the GPU cosine numbers in 3.3. `analytic_value_errors` at the paper config:
  0.49 s, peak +917 MiB above capture (384 MiB of it the returned errors); capture itself holds 3.75 GiB.
- Sampled-mean margins at K=4096, γ=0.5 (WP1t, fixed seed): cos 0.988/0.990/0.991, ratio 1.016/1.008/1.028.
- WP3 (2026-10-10): tiny CPU cosines over 2 seeds × 2 batches: v 0.08 (A) → 0.26 (B) → 0.27 (C), ve 0.05 → 0.22 → 0.23;
  other types identical or within noise. Gate cos has per-seed sd ≈ 0.13 on tiny (12-seed check: A 0.091, B 0.064, SE 0.037),
  so 3.3 needs many seeds before reading anything into gate/k differences between A and B.
