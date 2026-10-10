# 3080 Laptop smoke measurements (2026-10-10)

Hardware: RTX 3080 Laptop 8 GB (≈0.9 GB VRAM held by desktop apps), i7-11800H, Windows 11, torch 2.10.0+cu128.
Paper config (8 layers, 512 wide, seq 2048, batch 8), population 256, nrep 2, bf16. Run from the repo root.

| What | Command | Result |
|---|---|---|
| `DUST.step` time, A/B interleaved (A, B, A, B), 1 discarded warm-up + 2 timed steps each | `$PY results/exp1-value-credit/3080-smoke/step_time.py {A,B} 2` | `step_time.jsonl`: steps 42–87 s (A: 52, 86, 62, 54; B: 88, 59, 75, 43); warm-up 34–42 s; peak 6.5 GiB |
| `dust.py --steps 2` smoke, A and B | `$PY results/exp1-value-credit/3080-smoke/smoke_wrap.py --tokens 1M --population 256 --steps 2 [--value-credit analytic] --output runs/smoke-{A,B}` | wall 240 s (A), 172 s (B), including 3 val evals; peak 7.18 GiB (eval at batch 32 dominates) |
| One cosine estimate per variant at init, random batch | `$PY -m diagnostics.cosine --config paper --variants A,B,C --population 256 --seeds 0 --batches 1 --device cuda --out runs/smoke/cos-paper-init-smoke.json` | `cos-paper-init-smoke.json`: peak 6.4 GiB; timings (327/34/49 s) **invalid**: ran during FineWeb tokenization and near the VRAM limit |

The GPU ran at 84–86 °C with SM clocks of 1335–1800 MHz (thermal throttling). Step-time variance is far larger than
any A-vs-B difference, so no step-time saving is claimed from this machine; PLAN §3.3's timing readout moves to
the rented desktop GPUs. Losses after 2 steps match between A and B to 3 decimals (7.7940 vs 7.7958), as expected.

## Backprop checkpoint for 3.3 (`runs/bp-1m/best.pt`, not committed)

`$PY results/exp1-value-credit/3080-smoke/bp_wrap.py --tokens 1M --output runs/bp-1m` (wraps `baselines/backprop.py`):
61 steps, best val 5.9395, test 5.9257, wall 564 s, peak 4.88 GiB. Metadata in `bp-1m/`.
