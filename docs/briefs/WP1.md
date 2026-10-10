# WP1 `dust.py` value-credit flag (+ WP1t its tests)
owner: dust-dev (code), dust-test-writer (tests, in parallel)
interpreter: D:/Micromamba/micromambaenv/envs/dust/python.exe
test cmd: `D:/Micromamba/micromambaenv/envs/dust/python.exe -m pytest -q`

## Goal

Add "DUST with analytic local value credit": behind a flag, replace the sampled value and
value-embedding credit (`DUST.local_errors` kinds `'v'`, `'ve'`) with its exact expectation, a
matmul over cached activations. Default behaviour (`'sampled'`) is unchanged.

## Files

dust-dev writes: `dust.py`, `model.py` (one comment line only).
dust-test-writer writes: `tests/test_analytic_autograd.py`, `tests/test_value_credit_flag.py`.
do not touch: `diagnostics/`, `tests/conftest.py`, `tests/test_smoke.py`, `baselines/`, docs.

## Math

Per block `i`, batch `b`, head `h`, as in `DUST.local_errors`:

- `q, k, v` = `self.outputs[f'transformer.h.{i}.attn.c_{q,k,v}']`, viewed `[b, t, h, d]`, in fp32
- `qp, kp = norm(apply_rotary_emb(·, cos, sin))` of `q`, `k`, exactly as `processed()` (cos/sin fp32)
- on blocks with `has_ve(i, n_layer)`: `z = outputs[prefix+'ve_gate']` `[b,t,h]`, `gate = 2·sigmoid(z)`
- `ĝ = targets[i]` (the `o@{i}` error from `DUST.step`), viewed `[b, t, h, d]`
- `window = model.window_sizes[i][0]`

```
S[b,h,s,t] = (qp[b,s,h,:] · kp[b,t,h,:]) / sqrt(d)        # SDPA default scale
S[~mask]   = -inf,   mask[s,t] = 0 ≤ s−t ≤ window
A          = softmax(S, dim=-1)                            # [b, h, s, t]
Γ[s,t]     = γ^(s−t) on mask, else 0                       # local_errors' credit[t,s] is Γᵀ
e_v        = einsum('bhst,bshd->bthd', A * Γ, ĝ)           # [b, t, h, d]
e_ve       = e_v * gate[..., None]                         # VE blocks only
```

Return `e_v.reshape(b, t, n_embd)` under `transformer.h.{i}.attn.c_v` and `e_ve.reshape(b, t, n_embd)`
under `value_embeds.{i}`, so `DUST.reassemble` is reused unchanged. The scale matches
`local_errors('v')` exactly (it is uncentred and divides by draws and σ), so no extra factor.
γ = `settings.value_gamma` if not None else `settings.gamma`; variant C passes 1.0 (Γ = plain mask).

Compute in fp32 regardless of `self.dtype`. Build `A` one block at a time and free it before the
next; loop over heads inside a block so peak extra memory is one `[b, t, t]` (paper config:
8×2048×2048×4 B = 128 MiB per head).

## Contract

1. `Recipe` gains, after `gamma`: `value_credit: str = 'sampled'` (`'sampled' | 'analytic'`) and
   `value_gamma: float | None = None`. Validate the value (ValueError otherwise).
2. `DUST.analytic_value_errors(self, targets, gamma) -> dict[str, Tensor]`, as above.
3. `DUST.step`: if `value_credit == 'analytic'`, merge `analytic_value_errors` results into
   `gradients` via `reassemble`, and skip kinds `'v'`, `'ve'` in the `r.local` loop.
4. `local_errors`: return zeros (no division by zero) when `draws == 0`.
5. `recipe(tokens, population, value_credit='sampled', value_gamma=None)`: when analytic, set
   `local['v'] = local['ve'] = 0`. Do not redistribute those draws.
6. Split `DUST.step` into `estimate(x, y) -> dict[param_name, Tensor]` (everything up to and including
   `scalar_gradients`, returning `gradients` on the summed-loss scale, before `/ valid_tokens` and
   rank averaging) and `step(x, y, optimizer)` which calls `estimate` and keeps the remaining code
   as is. Diagnostics will call `estimate` to compare with autograd.
7. `main()`: `--value-credit {sampled,analytic}` (default sampled) and `--value-gamma FLOAT`
   (default None), passed to `recipe()`. They reach `config.json` through `asdict(settings)`.
8. `model.py :: attention()`: add one comment line saying `DUST.analytic_value_errors` rebuilds this
   attention matrix and must be kept in sync.

Everything else in `dust.py` stays byte-identical. With `'sampled'`, the RNG consumption order and the
estimate must be bit-for-bit the same as upstream for the same seed.

## Read (only these)

- `dust.py`: `Recipe` and `recipe()` (top ~90 lines), `DUST.__init__`, `local_errors`, `reassemble`,
  `step`, `main()` argument parsing.
- `model.py`: `attention`, `apply_rotary_emb`, `norm`, `has_ve`, `CausalSelfAttention.forward`.
- `diagnostics/tiny.py`, `tests/conftest.py`, `tests/test_smoke.py` (fixtures and style).

## Rules that apply (CLAUDE.md)

- Minimal, flag-gated diff to `dust.py`. With `--value-credit sampled` behaviour must be identical
  to upstream.
- The analytic attention matrix must match `model.attention` exactly.
- `credit[t, s]` is `Γᵀ`. Tests must use γ well below 1 so a transposed decay fails.
- B must match A's mean in direction **and** norm.
- Don't touch gate/key/query credit, σ, centring, noise, writer grouping.
- Tests: CPU, tiny config, no network or data, fixed seeds, expected values from math or autograd.

## Acceptance (test-writer: turn each into tests; dev: make them pass)

1. **Autograd pin, stage 3.2** (`test_analytic_autograd.py`). Tiny model with randomized `ve_gate`
   (fixture). Run a DUST `capture(x, y)`; separately compute with autograd on `L = sum of per-token CE`
   (`model(x, y, loss_reduction='none').sum()`) the true `∂L/∂o_i` (hook on `attn.c_proj` input),
   `∂L/∂v_i` (hook on `attn.c_v` output) and `∂L/∂ve_i` (hook on `value_embeds.{i}` output). Feed
   `targets = {i: true ∂L/∂o_i}` to `analytic_value_errors(targets, gamma=1.0)`:
   `cos > 0.9999` and norm ratio within 1e-4 of 1, for every block and both sites, and the
   reassembled `c_v.weight` / `value_embeds.{i}.weight` gradients match autograd's `.grad` likewise.
   Also run with `tiny_config(n_layer=4)` so a VE block with a short window is covered.
2. **Γ orientation.** A dense reference written with explicit Python loops over `s, t` on a very
   small case (e.g. t=8, window=3) at γ=0.5 equals `analytic_value_errors` to 1e-5. A transposed Γ
   must fail it.
3. **Mean of the sampled estimator** (cheap version; the full ladder is WP2). With `nrep=2`, fixed
   random `targets`, γ=0.5, `local_errors('v', K)` averaged over K=4096 draws has `cos > 0.98` and norm
   ratio within 5% of `analytic_value_errors(targets, 0.5)` for both blocks; same for `'ve'` on block 1.
4. **Sampled path unchanged.** With `value_credit='sampled'`, `estimate()` on a fixed seed returns the
   same tensors (bitwise) as running the upstream `step` logic. Practical check: two `DUST` instances
   with the same seed, one calling `estimate` and one calling `step` with a no-op optimizer, give equal
   `parameter.grad * valid_tokens` vs `gradients`; plus recipe values for every population equal upstream's.
5. **Flag plumbing.** `recipe('1M', 256, 'analytic')` has `local['v'] == local['ve'] == 0` and other
   counts unchanged; `analytic` `estimate()` returns finite gradients for every parameter name;
   `local_errors(kind, 0, targets)` returns zeros without error; bad `value_credit` raises ValueError;
   `value_gamma=None` uses `settings.gamma`.

## Clarifications (lead, 2026-10-10, before dispatch)

- **`estimate` keeps `@torch.no_grad()`**, like `step`. `step(x, y, optimizer)` becomes: `gradients =
  self.estimate(x, y)`, then the existing `valid_tokens` / `zero_grad` / per-parameter loop /
  `optimizer.step()` / return lines, verbatim. Validate `value_credit` in `Recipe.__post_init__`
  (`ValueError` unless `'sampled'` or `'analytic'`).
- **Analytic call placement** in `estimate`: after the `r.local` loop (which `continue`s on `'v'`, `'ve'`
  when analytic), before `scalar_gradients`. It uses no RNG, so the placement doesn't change draws.
  In `'sampled'` mode the loop must be the exact upstream loop: no extra RNG calls, same order.
- **`analytic_value_errors` inputs**: it reads `self.outputs`, `self.batch`, `self.length`, so it needs
  a prior `capture(x, y)`. `targets[i]` is `[b, t, n_embd]` (fp32 in `step`; cast to fp32 inside).
- **Autograd pin (acceptance 1), test-writer notes**: `make_model` returns `requires_grad_(False)`;
  call `model.requires_grad_(True)` for the autograd pass. Run `estimator.capture(x, y)` under
  `torch.no_grad()`, separate from the autograd forward. The DUST hooks stay registered during the
  autograd forward; they are no-ops when `capturing` is False and `jitter` is empty, so that's fine.
  In `CausalSelfAttention.forward`, the `c_v` output gradient equals `∂L/∂values` and
  `∂L/∂ve = gate · ∂L/∂values`. Use the `tiny_model` fixture (randomized `ve_gate`); for
  `n_layer=4` build `make_model(tiny_config(n_layer=4), cpu, 42)` and randomize `ve_gate` the same way.
  Compare reassembled weight gradients with `.grad` of `L = model(x, y, loss_reduction='none').sum()`
  (summed-loss scale, same as `reassemble`).
- **Γ orientation (acceptance 2)**: get a small case with
  `make_model(tiny_config(sequence_len=8), cpu, 42)` (windows: block 0 = 4, block 1 = 8) and a
  `[1, 8]` token batch, and also set `model.window_sizes[0] = (3, 0)` to cover a window shorter than
  t−1. The dense reference: explicit loops over `b, h, s, t`, computing `S`, the masked softmax and
  `Σ_s γ^(s−t) A[s,t] ĝ_s` directly from the cached `q, k, v` outputs, with `norm`/`apply_rotary_emb`
  from `model.py`. Include an assertion that the transposed-Γ version of the reference *differs* from
  the analytic output by more than 1e-2 relative (proves the test can fail).
- **Mean of sampled (acceptance 3)**: if cos > 0.98 / 5% isn't met at K=4096 with a fixed seed, don't
  loosen it silently. Report the measured cos/ratio in your final message (WP2 does the full ladder).
  Keep it under ~60 s on CPU.
- **Sampled path unchanged (acceptance 4), make it an upstream comparison**: in the test, write
  `git show main:dust.py` (via `subprocess`, cwd = repo root) to `tmp_path/dust_upstream.py`, import it with
  `importlib.util.spec_from_file_location` (it imports `model` from the repo root, which is on
  `sys.path` via conftest), and `pytest.skip` if git is unavailable. Build two identical tiny models
  (same `make_model` seed and the same `ve_gate` randomization), and run upstream `DUST.step` and new
  `DUST.step` with the same seed and a no-op optimizer (`zero_grad(set_to_none=True)` and `step()` do
  nothing). Assert `torch.equal` on every `parameter.grad` and equal returned losses. Also assert
  new `estimate(x, y)[name] / valid_tokens` equals the new step's `.grad` bitwise (same seed, fresh
  instance). Check `asdict(recipe(tok, pop))` for every tokens × population in `DEFAULT_CONFIGS` equals
  upstream's, apart from the two new keys (`value_credit='sampled'`, `value_gamma=None`).
- **Concurrent work**: nothing else is being written during WP1.
