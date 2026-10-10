# Roadmap: Experiment 1, analytic local value credit

Plan: `docs/PLAN-value-credit.md`. Branch: `exp/value-credit`.
Status values: `todo | in-progress | review | done`.
Interpreter: `D:/Micromamba/micromambaenv/envs/dust/python.exe`.

## Build

| WP | Name | Owner | Depends | Files | Status | Review / rounds |
|---|---|---|---|---|---|---|
| 0 | Setup: env, CLAUDE.md, agents, tiny config, conftest, smoke test | dust-lead | – | CLAUDE.md, .claude/agents/*, diagnostics/{__init__,tiny}.py, tests/{conftest,test_smoke}.py | done | – |
| 1 | `dust.py` flag: Recipe fields, `analytic_value_errors`, step wiring, `draws==0` guard, `recipe()`, CLI + config.json, `estimate()` split for cosine; `model.py` sync comment | dust-dev | 0 | dust.py, model.py | todo (brief ready: `docs/briefs/WP1.md`) | |
| 1t | Tests for WP1: autograd pin (stage 3.2), sampled path unchanged, flag/CLI plumbing | dust-test-writer | 0 | tests/test_analytic_autograd.py, tests/test_value_credit_flag.py | todo (same brief) | |
| 2 | `diagnostics/regression.py` (stage 3.1) + pytest wrapper | dust-dev / dust-test-writer | 1 | diagnostics/regression.py, tests/test_regression.py | todo | |
| 3 | `diagnostics/cosine.py` (per-parameter cosine vs autograd, layer-type table, CLI) | dust-dev | 1 | diagnostics/cosine.py, tests/test_cosine.py | todo | |

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
