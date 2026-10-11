# Roadmap: Experiment 2, late and predicted gradients

Plan: `docs/PLAN-decoupling.md` (approved 2026-10-10). Branch: `exp/decoupling`.
Status values: `todo | in-progress | review | done`. Interpreter: `D:/Micromamba/micromambaenv/envs/dust/python.exe`.

## Build

| WP | Name | Owner | Depends | Files | Status | Review / rounds |
|---|---|---|---|---|---|---|
| E2-1 | `--optimizer {sgd,adamw,muon}` in the backprop baseline, Muon (Newton–Schulz), CPU-testable train step | dust-dev | – | baselines/backprop.py, baselines/optim.py | done | pass, 1 round; lead applied: fp32 embeddings for adamw/muon (bf16 Adam steps round away), adamw rejects --momentum |
| E2-1t | Tests: SGD path bitwise = main, NS orthogonality vs SVD, Muon step, param partition | dust-test-writer | – | tests/test_optimizers.py | done | pass (NS range restricted to σ ≥ 5e-3·‖G‖_F for square G, justified) |
| E2-2 | `--grad-delay d`, weight-prediction flavour, per-block depth delay | dust-dev | E2-1 | baselines/decouple.py, baselines/backprop.py | done | pass, 1 round (default path bitwise vs 5b8633e verified for sgd/adamw/muon); lead applied: --weight-prediction requires delay > 0 |
| E2-3 | Synthetic-gradient predictor, measure-only phase, then DNI training | dust-dev | E2-2 | baselines/decouple.py, diagnostics/ | todo | |
| E2-4 | Gradient cache + data-conditioned predictor (measure phase) | dust-dev | E2-3 | TBD | todo | |

## Stages (PLAN §2), hard-gated

| Stage | What | Where | Gate | Status | Result |
|---|---|---|---|---|---|
| 2.0 | Strong baseline: lr sweep, then 3 seeds, SGD/AdamW/Muon at 10M | rented 4×3090 | Noah: approved | done | reference Muon lr 0.02 / adam_lr 0.008: val 4.348 ± 0.026; AdamW 4.509; SGD 4.998 |
| 2.1 | Staleness ladder d ∈ {0,1,2,4,8} × {plain, weight prediction} + depth mode + 2.1b lr retune | rented 4×3090 | Noah: approved | done | d=1 +0.40, d=2 +0.65 (= SGD); prediction ≈ halves d; depth d=4 ≈ uniform d=1; retune recovers ≤ 0.05 |
| 2.2 | Synthetic gradients: cosine phase, then training | rental | Noah | todo | |
| 2.3 | Data-conditioned predictor vs stale cache | rental | Noah | todo | |

## Notes

- 10M data download approved and started 2026-10-10 (`prepare_data.py --tokens 10M`, HF cache on `E:/hf_cache`).
- E2-WP1 review: on CUDA, wte/value_embeds are bf16 (model.init_weights); AdamW's ~lr-sized steps round away there (at lr 1e-3
  ~60% of entries never move, CPU simulation). For adamw/muon, backprop.py now casts them to fp32 (`fp32_embeddings` in config.json);
  sgd is untouched. **Verify in the first GPU smoke run** that config.json says fp32_embeddings true and wte.dtype is float32.
- Muon's AdamW subset (embeddings, head, scalars) uses one lr (`--adam-lr`, default 0.008). modded-nanogpt uses a much larger embedding lr;
  include adam_lr in the 2.0 sweep.
- E2-WP2 readout caveats for 2.1 (reviewer): with --weight-prediction, train_loss is L at the extrapolated point; compare val_loss
  (evaluated at restored weights). A delay-d run applies only steps − d updates (d=8: 602 of 610). In depth mode, AdamW step counts
  differ per parameter for the first d steps. The FIFO holds d+1 gradient copies momentarily (~1.3 GiB at d=8; fine on a 3090).
- Smoke on the 3080 (2026-10-10, 10M, --steps 2): sgd 77.5 s wall / 4.88 GiB; muon (lr 0.02) 78.0 s / 5.69 GiB, val after 2 steps
  6.98 vs sgd 7.80; adamw (lr 1e-3) 5.78 GiB (wall-clock invalid: ran during CPU test runs). fp32_embeddings true for adamw/muon on CUDA ✓.
