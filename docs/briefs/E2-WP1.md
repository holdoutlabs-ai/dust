# E2-WP1 Optimizer flag for the backprop baseline (+ E2-WP1t its tests)
owner: dust-dev (code), dust-test-writer (tests, in parallel)
interpreter: D:/Micromamba/micromambaenv/envs/dust/python.exe
test cmd: `D:/Micromamba/micromambaenv/envs/dust/python.exe -m pytest tests/test_optimizers.py -q`

## Goal

Stage 2.0 needs a strong backprop reference. Add `--optimizer {sgd,adamw,muon}` to `baselines/backprop.py`.
`sgd` (the default) must reproduce `main` bitwise. AdamW and Muon get their own lrs.

## Files

dust-dev writes: `baselines/optim.py` (new: Muon, Newton–Schulz, the parameter partition, `build_optimizer`) and minimal edits
to `baselines/backprop.py` (flags, call `build_optimizer`, record the choice in `config.json`).
test-writer writes: `tests/test_optimizers.py`.
do not touch: `dust.py`, `model.py`, `diagnostics/`, other tests, docs. Nothing else is being written now (a data download runs
in the background into `data/`; don't touch `data/`).

## Contract

```python
# baselines/optim.py
def zeropower_newton_schulz(G: Tensor, steps: int = 5) -> Tensor
    # G 2-D. X = G / (‖G‖_F + 1e-7), in float32 (bf16 on CUDA is acceptable later; tests use fp32 on CPU).
    # If rows > cols, work on Xᵀ and transpose back. Repeat `steps` times, with (a, b, c) = (3.4445, -4.7750, 2.0315):
    #     A = X @ X.T ;  B = b * A + c * A @ A ;  X = a * X + B @ X
    # Result ≈ U · f(S) · Vᵀ for G = U S Vᵀ, with f(S) in roughly [0.68, 1.13] (quintic NS does not converge exactly).

class Muon(torch.optim.Optimizer):
    # params: 2-D tensors only (ValueError otherwise). defaults: lr, momentum=0.95, nesterov=True, ns_steps=5.
    # per step, for each p with grad g:
    #     buf = momentum * buf + g            (buf starts at zero)
    #     u   = g + momentum * buf  if nesterov else buf
    #     O   = zeropower_newton_schulz(u, ns_steps)
    #     p  -= lr * max(1, rows / cols) ** 0.5 * O      (rows, cols = p.shape)

def partition(model) -> tuple[list[Parameter], list[Parameter]]
    # (muon_params, adam_params). muon_params = every 2-D weight under transformer.h.* EXCEPT attn.ve_gate
    # (c_q, c_k, c_v, attn.c_proj, mlp.c_fc, mlp.c_proj). adam_params = everything else: transformer.wte,
    # value_embeds.*, lm_head, resid_lambdas, x0_lambdas, attn.ve_gate. Every parameter in exactly one list.

def build_optimizer(model, name, lr, momentum, adam_lr=None) -> list[torch.optim.Optimizer]
    # 'sgd'  : exactly upstream make_optimizer(model, lr, momentum) (same two groups, same order, wte lr 1000.0) -> [sgd]
    # 'adamw': torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0) -> [adamw]
    # 'muon' : [Muon(muon_params, lr=lr, momentum=momentum), AdamW(adam_params, lr=adam_lr or 0.008, betas=(0.9, 0.95),
    #           eps=1e-8, weight_decay=0.0)]
```

`train_step(model, optimizers, x, y)` takes a **list** of optimizers (zero_grad all, backward, step all) and stays CPU-runnable
(autocast is already disabled when not on CUDA). `main()` gains `--optimizer` (default `sgd`) and `--adam-lr FLOAT`.
For adamw and muon, `--lr` and `--momentum` have no upstream defaults; require `--lr` explicitly (`parser.error` otherwise)
and use momentum 0.95 for Muon when not given. Record `optimizer`, `adam_lr` in `config.json` metadata (new keys only;
existing keys and values unchanged for sgd). Constant lr for all optimizers (no schedule), as upstream.

## Read (only these)

- `baselines/backprop.py` (whole file, 149 lines), `model.py` `GPT.__init__` (parameter names), `diagnostics/tiny.py`,
  `tests/conftest.py`, `tests/test_value_credit_flag.py` lines 1–60 (how the upstream module is loaded from `git show main:...`).

## Rules that apply (CLAUDE.md)

- With default flags behaviour must be identical to `main`: same results bitwise for the same seed. `dust.py` and `model.py` are not touched.
- Tests: CPU, tiny config, no network or data files, fixed seeds, minutes. Expected values come from math, an SVD reference or
  the unmodified upstream code, never the code under test.

## Acceptance (test-writer: turn into tests; dev: make them pass)

1. **SGD unchanged:** load upstream `baselines/backprop.py` from `git show main:baselines/backprop.py` (skip if git is unavailable).
   On two identical tiny CPU models (`make_model(tiny_config(), cpu, 42).requires_grad_(True)`, random token batch), 3 steps of
   upstream `train_step(model, make_optimizer(model, lr, mom), x, y)` and 3 steps of the new `train_step(model,
   build_optimizer(model, 'sgd', lr, mom), x, y)` give bitwise-equal parameters and losses. The param groups match too
   (lrs, momentum, parameter identity and order).
2. **Newton–Schulz vs SVD:** for random fp32 matrices of shapes (64, 64), (64, 256) and (256, 64), with G = U S Vᵀ from
   `torch.linalg.svd`: `M = Uᵀ · NS(G) · V` is diagonal (off-diagonal Frobenius < 2% of diagonal) with diagonal entries in [0.6, 1.25].
   NS is scale-invariant: NS(10·G) ≈ NS(G) (rel 1e-5). Also test an ill-conditioned G (singular values spanning 1e-3 … 1):
   the diagonal still stays in range for singular values above ~1e-2.
3. **Muon step math:** single 2-D parameter, known gradient, momentum 0 → `p_new = p − lr·max(1, r/c)^0.5·NS(g)` (rel 1e-6), for a tall
   and a wide shape. With momentum 0.5 and nesterov, two steps with gradients g1, g2 reproduce the hand-computed buffers. A 1-D parameter → ValueError.
4. **Partition:** on the tiny model (n_layer 2 and 4), every parameter appears in exactly one list; muon_params are exactly the
   2-D block weights except `ve_gate`; `build_optimizer('muon')` returns two optimizers covering all parameters once.
5. **AdamW wiring:** `build_optimizer(model, 'adamw', 1e-3, None)` has one group covering every parameter with betas (0.9, 0.95)
   and weight_decay 0. A few CPU train steps with each optimizer decrease the training loss on a fixed tiny batch (sanity, not a benchmark).
6. **CLI:** `--optimizer muon` without `--lr` errors. (Parse-level test only: call the argparse path, or factor a `parse_args(argv)`.)
