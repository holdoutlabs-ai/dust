# Experiment 2 results: late and predicted gradients

Plan: `docs/PLAN-decoupling.md`. Branch `exp/decoupling`. Interpreter at home: `D:/Micromamba/micromambaenv/envs/dust/python.exe`;
rental: `/root/venv/bin/python` (torch 2.10.0+cu128).

| Stage | Files | Command | Code commit | Result |
|---|---|---|---|---|
| 2.0 strong baseline | `2.0/<run>/`, `2.0-summary.json` | job lists `rental/queue-2.0A.txt`, `rental/queue-2.0B.txt`; `summarize.py results/exp2-decoupling/2.0 --reference muon-lr0.02-s` | e574e8c | **reference: Muon lr 0.02, adam_lr 0.008: val 4.348 ± 0.026** |
| 2.1 staleness ladder + 2.1b lr retune | `2.1/<run>/`, `2.1b/<run>/`, `2.1-summary.json` | job lists `rental/queue-2.1*.txt`; summary built from 2.1 + 2.1b + the three 2.0 reference runs as `d0-s<seed>` | e574e8c | **done**: see §2.1 |

## 2.0: strong backprop baseline (2026-10-10, rented 4× RTX 3090, 10M tokens = 610 steps, batch 8 × 2048)

All runs use `baselines/backprop.py --tokens 10M` with a constant lr (no schedule), weight decay 0, and eval every 10 steps.
AdamW and Muon keep fp32 master copies of the embeddings (`fp32_embeddings: true`). Seeds share init and data order.

| configuration | best val | n | paired vs reference |
|---|---|---|---|
| **Muon lr 0.02, adam_lr 0.008 (reference)** | **4.3481 ± 0.0262** | 3 | – |
| Muon lr 0.02, adam_lr 0.003 | 4.3401 | 1 | +0.010 (seed 42) |
| Muon lr 0.01 | 4.3598 | 1 | +0.030 |
| Muon lr 0.04 | 4.4170 | 1 | +0.087 |
| Muon lr 0.02, adam_lr 0.03 | 4.4365 | 1 | +0.107 |
| AdamW lr 1e-3 | 4.5093 ± 0.0147 | 3 | +0.161 ± 0.039 |
| AdamW lr 3e-3 | 4.5465 | 1 | |
| Muon lr 0.08 | 4.5586 | 1 | |
| AdamW lr 3e-4 | 4.8991 | 1 | |
| SGD lr 0.2, momentum 0.95 (repo's tuned baseline) | 4.9980 ± 0.0134 | 3 | +0.650 ± 0.035 |
| AdamW lr 1e-2 | 5.3013 | 1 | |
| Muon lr 0.02, adam_lr 0.1 | 6.7784 (diverged) | 1 | |

- Both optima lie inside the swept ranges (Muon lr 0.01–0.08; AdamW 3e-4–1e-2). adam_lr 0.003 vs 0.008 is within seed noise
  (one seed, +0.010 vs the reference's std of 0.026), so the default 0.008 is kept.
- The repo's SGD baseline is **0.65 nats** behind Muon at 10M tokens. Experiment 1's DUST comparison was against this weaker baseline.
- Step time: ~0.3–0.6 s per training step, ~4.5 s per eval. Wall-clock per run was 480–700 s, with 4 jobs sharing the box.

## 2.1: staleness ladder (2026-10-10/11, same rental, Muon lr 0.02 / adam_lr 0.008 unless noted, 3 seeds per cell)

Gradient computed at step t is applied at step t + d (`--grad-delay d`). Weight prediction evaluates the gradient at
`w_t + d·Δ_t`, where Δ is the last applied update. Depth mode gives block i of 8 a delay of round(d·(7 − i)/7), with wte at d and lm_head and the scalars at 0.
Paired differences are against the d = 0 reference (the three 2.0 runs `muon-lr0.02-s4{2,3,4}`).

| delay | plain | with weight prediction |
|---|---|---|
| 0 | 4.3481 ± 0.0262 (reference) | – |
| 1 | 4.7451 ± 0.0203 (**+0.397 ± 0.011**) | 4.4819 ± 0.0155 (**+0.134 ± 0.013**) |
| 2 | 4.9988 ± 0.0030 (+0.651 ± 0.027) | 4.7307 ± 0.0110 (+0.383 ± 0.037) |
| 4 | 5.2159 ± 0.0054 (+0.868 ± 0.026) | 4.9947 ± 0.0054 (+0.647 ± 0.028) |
| 8 | 5.4321 ± 0.0090 (+1.084 ± 0.035) | 5.2167 ± 0.0076 (+0.869 ± 0.030) |
| 4, depth mode | 4.7529 ± 0.0167 (+0.405 ± 0.022) | – |

- **A stale gradient is expensive.** One step of delay costs 0.40 nats. Two steps erase Muon's whole lead over the repo's SGD (4.998).
  After the first step, every doubling of d costs about +0.22.
- **Weight prediction is worth one halving of the delay**, to within 0.02 at every d: predicted d = 2k ≈ plain d = k.
- **Pipeline-shaped staleness is cheaper:** depth mode at d = 4 (mean block delay ≈ 2) costs the same as a uniform delay of 1.
- With delay d, the last d gradients are never applied (d = 8: 602 of 610 updates). With prediction, train_loss in metrics.jsonl is
  measured at the extrapolated weights; compare val only.

### 2.1b: learning-rate retune (PLAN amendment 2026-10-10)

Seed 42 at lr ∈ {0.005, 0.01, 0.02}, then 3 seeds at lr 0.01 for the cells where it won:

| cell | lr 0.005 (s42) | lr 0.01 | lr 0.02 |
|---|---|---|---|
| d = 1 | 4.7973 | **4.7040 ± 0.0272** (3 seeds, +0.356 vs ref) | 4.7451 ± 0.0203 (+0.397) |
| d = 1, prediction | 4.5763 | 4.4742 (s42) | **4.4819 ± 0.0155** (+0.134) |
| d = 2 | 5.0321 | 5.0047 (s42) | **4.9988 ± 0.0030** (+0.651) |
| d = 2, prediction | 4.7727 | **4.6835 ± 0.0189** (3 seeds, +0.335 vs ref) | 4.7307 ± 0.0110 (+0.383) |

Retuning recovers only **0.04–0.05 nats** of the 0.35–0.40 cost (d = 1 plain, d = 2 predicted) and nothing elsewhere. The cost is
staleness, not an lr mismatch.
