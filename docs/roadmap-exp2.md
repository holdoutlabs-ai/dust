# Roadmap: Experiment 2, late and predicted gradients

Plan: `docs/PLAN-decoupling.md` (approved 2026-10-10). Branch: `exp/decoupling`.
Status values: `todo | in-progress | review | done`. Interpreter: `D:/Micromamba/micromambaenv/envs/dust/python.exe`.

## Build

| WP | Name | Owner | Depends | Files | Status | Review / rounds |
|---|---|---|---|---|---|---|
| E2-1 | `--optimizer {sgd,adamw,muon}` in the backprop baseline, Muon (Newton–Schulz), CPU-testable train step | dust-dev | – | baselines/backprop.py, baselines/optim.py | in-progress | |
| E2-1t | Tests: SGD path bitwise = main, NS orthogonality vs SVD, Muon step, param partition | dust-test-writer | – | tests/test_optimizers.py | in-progress | |
| E2-2 | `--grad-delay d`, weight-prediction flavour, per-block depth delay | dust-dev | E2-1 | baselines/decouple.py, baselines/backprop.py | todo | |
| E2-3 | Synthetic-gradient predictor, measure-only phase, then DNI training | dust-dev | E2-2 | baselines/decouple.py, diagnostics/ | todo | |
| E2-4 | Gradient cache + data-conditioned predictor (measure phase) | dust-dev | E2-3 | TBD | todo | |

## Stages (PLAN §2), hard-gated

| Stage | What | Where | Gate | Status | Result |
|---|---|---|---|---|---|
| 2.0 | Strong baseline: lr sweep, then 3 seeds, SGD/AdamW/Muon at 10M | rental | Noah: rental + batch | todo | |
| 2.1 | Staleness ladder d ∈ {0,1,2,4,8} × {plain, weight prediction} | rental | Noah | todo | |
| 2.2 | Synthetic gradients: cosine phase, then training | rental | Noah | todo | |
| 2.3 | Data-conditioned predictor vs stale cache | rental | Noah | todo | |

## Notes

- 10M data download approved and started 2026-10-10 (`prepare_data.py --tokens 10M`, HF cache on `E:/hf_cache`).
