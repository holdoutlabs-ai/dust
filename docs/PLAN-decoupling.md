# Experiment 2: how much does a late or predicted gradient cost?

Status: approved by Noah as written, 2026-10-10 (10M data download approved the same day). Branch `exp/decoupling` (from `exp/value-credit`, to reuse `diagnostics/cosine.py` and the
gated workflow). Never commit to `main`. Only the lead edits this plan, with the date and reason.

## 0. Goal and hypotheses

Backprop locks each layer's update to the full forward + backward round trip: a layer needs (a) its stored input
activations and (b) the error from everything above it. Every way of breaking that lock replaces the true, current
gradient with a **late** one (pipelining with stale weights) or a **predicted** one (synthetic gradients, local losses,
and Noah's data-conditioned predictor). Before building any of them, measure what they would have to beat:

- **H1 (staleness):** at this scale, gradients delayed by a few steps cost little loss, and momentum-based weight prediction
  recovers most of the cost.
- **H2 (synthetic gradients):** a small per-block predictor of ∂L/∂h reaches a useful cosine to the true gradient, and
  training on it for the lower blocks costs a bounded amount of loss.
- **H3 (Noah's predictor):** a per-example gradient cache plus a learned, data-conditioned first-order correction predicts the
  current gradient better than the plain stale cache does.

All three are judged against a **strong backprop baseline** in loss at fixed tokens (and wall-clock where a method
claims a speedup). Cosine is a filter only: Experiment 1 showed that 2× higher cosine can give zero loss gain.

## 1. Setup

- Model, data and evaluation as in `baselines/backprop.py` (8 layers, 512 wide, seq 2048, batch 8, FineWeb, 4096 vocab).
- Token budget: **10M** (610 steps). 1M is only 61 steps, too short for delays of 4–8 steps. Needs
  `prepare_data.py --tokens 10M` (network: gate).
- Seeds 42/43/44, paired across variants (same init and data order). CPU tests use the tiny config, as in Experiment 1.

## 2. Stages (hard-gated in order; each commits JSON + README with the exact command)

### 2.0 Strong baseline

Add `--optimizer {sgd,adamw,muon}` to `baselines/backprop.py` (default `sgd` = byte-identical behaviour). Muon is
Newton–Schulz orthogonalized momentum on 2-D hidden matrices, with AdamW for embeddings, head and scalars. Do a short lr sweep
at 10M for each (one seed, ~5 lrs), then 3 seeds at the best lr. **Reference = best of the three.** Report loss and wall-clock.
- Test gate: SGD path unchanged (bitwise vs `main`); Muon's orthogonalization is checked against an SVD reference on CPU.

### 2.1 Staleness ladder (H1)

`--grad-delay d`: the gradient computed at step t is applied at step t+d (a FIFO of d gradients); d ∈ {0, 1, 2, 4, 8}.
Two flavours: plain delay, and delay with weight prediction (compute the gradient at `w_t + d·Δ_momentum`, the extrapolated
weights). One more pipeline-realistic variant: per-block delay proportional to depth from the top.
- Test gate: d = 0 is bitwise the baseline; a d = 2 run on CPU applies exactly the gradients of steps t−2.
- Readout: loss at 10M vs d, paired differences against d = 0, for the reference optimizer.

**Amendment (2026-10-10, approved by Noah): 2.1b lr retune.** At the reference lr (0.02, tuned for d = 0), a 1-step delay cost
+0.40 and a 2-step delay +0.65 val (the latter erases Muon's whole lead over SGD). To separate staleness from an lr that is too aggressive
for stale gradients, re-run d ∈ {1, 2} × {plain, weight prediction} at lr ∈ {0.005, 0.01} (seed 42), then 3 seeds of the best lr per cell.
The ladder's fixed-lr results stay as run.

### 2.2 Synthetic gradients (H2)

A small per-block predictor (linear, then a 2-layer MLP) maps a block's output h (+ optional token embedding) to ∂L/∂h,
trained online by regression on the true ∂L/∂h. Phase 1, measure only: cosine of predicted vs true ∂L/∂h per block over
training, using the `diagnostics/cosine.py` approach. Phase 2, only if phase 1 cosine is useful: blocks below k update from
predicted gradients (DNI style), with k ∈ {2, 4, 6}.
- Test gate: with the predictor replaced by an oracle (true ∂L/∂h), training is bitwise the baseline.

### 2.3 Data-conditioned gradient predictor (H3, Noah's idea)

1. **Cache:** at refresh steps, store each training example's gradient, projected to a low dimension with a fixed random
   projection (TRAK-style), plus a precomputed embedding of the example's content.
2. **Null model:** predict the current gradient by the cached one (pure staleness). Measure its cosine to the true gradient as a
   function of steps since refresh.
3. **Predictor:** a small model takes (example embedding, cached projected gradient, summary of the weight change since
   refresh) → current projected gradient. First-order structure: predict the correction term `H·Δw` rather than the whole gradient.
4. **Gate to training:** only if the predictor beats the null by a clear cosine margin at a cost below one backward pass.
   Then train with predicted updates between refreshes and compare loss with the reference at matched compute.

## 3. Decision rules

- An idea is interesting only if it **matches the reference loss within noise while removing a lock**, or beats it.
- If 2.1 shows that even d = 1 costs more than the gap between SGD and the reference optimizer, decoupling at this scale is
  expensive, and 2.2/2.3 must beat the stale baseline, not just the true gradient.
- If 2.0 shows the reference far ahead of SGD, Experiment 1's DUST comparison should be re-read against it.

## 4. Not in this experiment

Multi-GPU pipeline systems engineering, forward-forward, feedback alignment, model sizes beyond the paper config (a scale
ladder comes after a positive result), retuning per variant beyond the documented lr sweep.

## 5. Budget and hardware

Backprop at the paper config is ~1–2 s/step on a 3090 (to be measured in 2.0), so a 10M run is roughly 15–25 min.
- 2.0: about 15 sweep runs plus 9 seed runs.
- 2.1: about 27 runs.
- 2.2/2.3 phase 1: measurement runs.

Estimate: **15–25 GPU-hours**, about **$3–5 per hour of 4× 3090** (≈ $0.70/h for the box), so roughly **$10–20 in total**.
Gates as in Experiment 1: data download, every rental, and each training batch are approved by Noah with a timed `--steps 2` estimate first.

## 6. Deliverables

Per stage: JSON + README under `results/exp2-decoupling/`, then `REPORT.md` answering: what does staleness cost, can a predictor
do better than staleness, and is any decoupled variant within noise of a strong backprop baseline?
