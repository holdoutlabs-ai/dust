# CLAUDE.md

Guidance for Claude Code in this repo.

## What this is

`holdoutlabs-ai/dust`, a fork of `qlabs-eng/dust`: a minimal implementation of DUST, which trains a
transformer with forward passes and SGD only (no backprop). `model.py` is the model, `dust.py` the
estimator and training loop, `baselines/backprop.py` the backprop baseline. Paper:
<https://qlabs.sh/research/dust>.

## Current work: Experiment 2, late and predicted gradients (branch `exp/decoupling`)

Plan: `docs/PLAN-decoupling.md` (source of truth; lead edits only, with date and reason). Roadmap:
`docs/roadmap-exp2.md`. Briefs: `docs/briefs/E2-WP<n>.md`. Results: `results/exp2-decoupling/`.
Stages 2.0 strong baseline (AdamW/Muon) → 2.1 staleness ladder → 2.2 synthetic gradients → 2.3 data-conditioned
gradient predictor, hard-gated in order. Same build team, gates and test rules as below.

Experiment-2 code rules:
- `baselines/backprop.py` gets minimal, flag-gated changes. With default flags (`--optimizer sgd`, no delay, no predictor)
  behaviour must be identical to `main`: same results bitwise for the same seed. `dust.py` and `model.py` are not touched.
- New mechanisms (delays, predictors) live in new modules (e.g. `baselines/decouple.py`), not inline sprawl.
- Expected values in tests come from math, an SVD/autograd reference, or the unmodified baseline, never the code under test.

## Previous work: Experiment 1, analytic local value credit (done; `results/exp1-value-credit/REPORT.md`)

Plan: `docs/PLAN-value-credit.md` (the source of truth; only `dust-lead` edits it, with the date and reason).
Roadmap and status: `docs/roadmap.md`. Briefs: `docs/briefs/WP<n>.md`.
Branch: `exp/value-credit`. **Never commit to `main`.**

In short: replace DUST's sampled value / value-embedding credit (`DUST.local_errors`, kinds `'v'`,
`'ve'`) with its exact expectation `e_v[t] = Σ_s γ^(s−t) A[s,t] ĝ_s`, behind `--value-credit analytic`.
Variants: A = sampled (baseline), B = analytic with the same γ, C = analytic with γ = 1.

## Commands

Env: micromamba `dust` (Python 3.12, torch 2.10.0+cu128). Always call the interpreter by full path;
in Bash, bare `python` is the system 3.13 without torch.

```bash
PY=D:/Micromamba/micromambaenv/envs/dust/python.exe
$PY -m pytest -q                                   # CPU tests (tiny config)
$PY -m pytest tests/test_analytic_autograd.py -q -x
$PY -m diagnostics.regression                       # stage 3.1, CPU, < 1 min (WP2, not built yet)
$PY -m diagnostics.cosine --config tiny ...         # stage 3.3 tool, PLAN §2.2 (WP3, not built yet)
$PY prepare_data.py --tokens 1M                     # network: streams FineWeb (gate: ask Noah)
$PY dust.py --tokens 1M --population 256 --steps 2 --output runs/smoke   # GPU
$PY baselines/backprop.py --tokens 1M --output runs/bp-1m               # GPU, backprop checkpoint
```

## Hardware and platform

- Home box: Windows 11, **RTX 3080 Laptop, 8 GB**. The paper config (8 layers, 512 wide, seq 2048,
  batch 8) may not fit; measure with `torch.cuda.max_memory_allocated()` before assuming.
  Measured 2026-10-10: one clean bf16 forward at the paper config peaks at 0.93 GiB (DUST adds
  `nrep` copies of the batch plus caches; not yet measured).
- Upstream README says Linux. On Windows, single-GPU runs only (no NCCL / `torchrun`).
- CPU tests use the tiny config in `diagnostics/tiny.py` and float32 (DUST picks fp32 on CPU, bf16 on CUDA).

## Rules for all code

- **Minimal, flag-gated diff to `dust.py`.** With `--value-credit sampled` (the default) behaviour must
  be identical to upstream: same draws, same RNG consumption order, same result. Everything outside the
  plan's §2.1 list stays byte-identical. `model.py` only gets the one-line sync comment in `attention()`.
- The analytic attention matrix must match `model.attention` exactly: SDPA default scale `1/√d`, mask
  `0 ≤ s−t ≤ window`, `qp, kp = norm(apply_rotary_emb(·))` as in `local_errors.processed`.
- Index conventions are where this goes wrong: `local_errors`' `credit[t, s]` is `Γᵀ`. Tests must use
  γ well below 1 (e.g. 0.5) so a transposed decay fails.
- Scale: the sampled `'v'` estimator is uncentred and divides by `draws` and `σ`, so B must match A's
  mean in **direction and norm**. A constant-factor mismatch is a bug.
- Don't touch gate/key/query credit, σ, centring, noise, writer grouping, or lr/momentum per variant
  (PLAN §4). Don't redistribute freed `v`/`ve` draws.
- Tests run on CPU with the tiny config, no network, no data files (use random token batches), fixed seeds,
  and finish in minutes. Expected values come from math or autograd, never from the code under test.
- Results are JSON under `results/exp1-value-credit/` with the exact command in the README beside them.
  `data/` and `runs/` are gitignored.

## Gates: ask Noah first

- Network downloads (`prepare_data.py` streams FineWeb).
- Any GPU run longer than a smoke test (`--steps 1/2`), and the 9-run training comparison (PLAN §3.4):
  time one `--steps 2` run first and report the extrapolated total.
- Renting GPUs or using a friend's machine. Not before 3.1–3.3 pass (PLAN §4).
- `git push`, PRs, anything touching `main`.

## Build team

Start a build session with `claude --agent dust-lead`. Definitions are in `.claude/agents/`.

| Agent | Model | Owns |
|---|---|---|
| `dust-lead` | opus | `docs/`, briefs, glue, running stages, results, REPORT.md, git commits |
| `dust-dev` | opus | the code files a brief assigns (`dust.py`, `diagnostics/*.py`) |
| `dust-test-writer` | sonnet | test files named in a brief, written from the math, not the implementation |
| `dust-reviewer` | fable (high) | read-only review: hunts index/scale/mask bugs and unintended changes to the sampled path |

Builders read only their brief and the files it lists. Only the lead commits.
