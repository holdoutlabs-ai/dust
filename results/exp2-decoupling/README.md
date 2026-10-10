# Experiment 2 results: late and predicted gradients

Plan: `docs/PLAN-decoupling.md`. Branch `exp/decoupling`. Interpreter at home: `D:/Micromamba/micromambaenv/envs/dust/python.exe`;
rental: `/root/venv/bin/python` (torch 2.10.0+cu128).

| Stage | Files | Command | Code commit | Result |
|---|---|---|---|---|
| 2.0 strong baseline | `2.0/<run>/`, `2.0-summary.json` | job lists `rental/queue-2.0A.txt`, `rental/queue-2.0B.txt`; `summarize.py results/exp2-decoupling/2.0 --reference muon-lr0.02-s` | e574e8c | **reference: Muon lr 0.02, adam_lr 0.008: val 4.348 ± 0.026** |
| 2.1 staleness ladder | | | e574e8c | running |

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
