"""WP1 acceptance 1-3: analytic value credit vs autograd, a dense loop reference, and the sampled mean.

Expected values come from autograd on the model, explicit Python loops, or an averaged sampled estimator.
"""
import math

import pytest
import torch

from diagnostics.tiny import tiny_config, tiny_recipe
from dust import DUST, make_model
from model import apply_rotary_emb, has_ve, norm

CPU = torch.device('cpu')


def randomize_gates(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for block in model.transformer.h:
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight.copy_(torch.randn(block.attn.ve_gate.weight.shape, generator=g) * 0.5)


def cos_and_ratio(a, b):
    """cos(a, b) and |a| / |b| on flattened tensors."""
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item(), (a.norm() / b.norm()).item()


def autograd_truth(model, x, y):
    """True dL/do_i, dL/dv_i, dL/dve_i and parameter grads for L = sum of per-token CE."""
    store = {'o': {}, 'v': {}, 've': {}}
    hooks = []
    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    def pre(i):
        def hook(module, args):
            args[0].retain_grad()
            store['o'][i] = args[0]
        return hook

    def post(kind, i):
        def hook(module, args, output):
            output.retain_grad()
            store[kind][i] = output
        return hook

    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.c_proj.register_forward_pre_hook(pre(i)))
        hooks.append(block.attn.c_v.register_forward_hook(post('v', i)))
        if str(i) in model.value_embeds:
            hooks.append(model.value_embeds[str(i)].register_forward_hook(post('ve', i)))
    try:
        loss = model(x, y, loss_reduction='none').sum()
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
    truth = {kind: {i: t.grad.detach().clone() for i, t in d.items()} for kind, d in store.items()}
    weight_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    model.requires_grad_(False)
    return truth, weight_grads


def run_pin(model, batch):
    x, y = batch
    est = DUST(model, tiny_recipe(), seed=0)
    try:
        with torch.no_grad():
            est.capture(x, y)
        truth, weight_grads = autograd_truth(model, x, y)
        n_layer = model.config.n_layer
        with torch.no_grad():
            out = est.analytic_value_errors({i: truth['o'][i].clone() for i in range(n_layer)}, gamma=1.0)
            for i in range(n_layer):
                site = f'transformer.h.{i}.attn.c_v'
                cos, ratio = cos_and_ratio(out[site], truth['v'][i])
                assert cos > 0.9999 and abs(ratio - 1) < 1e-4, (i, 'v', cos, ratio)
                w = est.reassemble(site, out[site])
                cos, ratio = cos_and_ratio(w, weight_grads[site + '.weight'])
                assert cos > 0.9999 and abs(ratio - 1) < 1e-4, (i, 'c_v.weight', cos, ratio)
                if has_ve(i, n_layer):
                    site = f'value_embeds.{i}'
                    cos, ratio = cos_and_ratio(out[site], truth['ve'][i])
                    assert cos > 0.9999 and abs(ratio - 1) < 1e-4, (i, 've', cos, ratio)
                    w = est.reassemble(site, out[site])
                    cos, ratio = cos_and_ratio(w, weight_grads[site + '.weight'])
                    assert cos > 0.9999 and abs(ratio - 1) < 1e-4, (i, 've.weight', cos, ratio)
                else:
                    assert f'value_embeds.{i}' not in out
    finally:
        est.close()


def test_autograd_pin_two_layers(tiny_model, tiny_batch):
    """At gamma=1 with the true dL/do, analytic e_v / e_ve / weight grads equal autograd (cos>0.9999, norm within
    1e-4). Catches a wrong mask or window, wrong scale (1/sqrt(d)), missing rotary/norm, missing ve gate
    multiplier (gate randomized, so != 1), sum-vs-mean mismatch. Block 0 has no VE, block 1 does."""
    run_pin(tiny_model, tiny_batch)


def test_autograd_pin_four_layers_short_window_ve_block(tiny_batch):
    """Same pin with n_layer=4: blocks 1 (VE, window 64 < 128 tokens) and 3 (VE, full). Catches the window
    being ignored or mis-indexed per block, and VE handling on a short-window block."""
    model = make_model(tiny_config(n_layer=4), CPU, 42)
    randomize_gates(model)
    assert model.window_sizes[1][0] == 64 and has_ve(1, 4)
    run_pin(model, tiny_batch)


def dense_reference(est, model, i, ghat, gamma, variant='correct'):
    """Explicit loops over b, h, s, t from the cached q, k. Returns e_v [b, t, h, d] in float64."""
    b, t, h, d = est.batch, est.length, model.config.n_head, model.config.n_embd // model.config.n_head
    pre = f'transformer.h.{i}.attn.'
    q = est.outputs[pre + 'c_q'].float().view(b, t, h, d)
    k = est.outputs[pre + 'c_k'].float().view(b, t, h, d)
    cos, sin = model.cos[:, :t].float(), model.sin[:, :t].float()
    qp = norm(apply_rotary_emb(q, cos, sin)).double()
    kp = norm(apply_rotary_emb(k, cos, sin)).double()
    window = model.window_sizes[i][0]
    g = ghat.double().view(b, t, h, d)
    out = torch.zeros(b, t, h, d, dtype=torch.float64)
    for bi in range(b):
        for hi in range(h):
            for s in range(t):
                ts = [u for u in range(t) if 0 <= s - u <= window]
                scores = torch.stack([(qp[bi, s, hi] * kp[bi, u, hi]).sum() / math.sqrt(d) for u in ts])
                a = torch.softmax(scores, 0)
                for u, au in zip(ts, a):
                    lag = s - u
                    if variant == 'correct':
                        w = gamma ** lag
                    elif variant == 'flipped':      # decay grows with lag instead of shrinking
                        w = gamma ** (-lag)
                    else:                           # 'diagonal': Gamma^T intersected with the causal mask
                        w = 1.0 if lag == 0 else 0.0
                    out[bi, u, hi] += w * au * g[bi, s, hi]
    return out


def test_gamma_orientation_against_dense_loops():
    """gamma=0.5, t=8, window 3 (block 0) and 8 (block 1, VE). The loop reference sum_s gamma^(s-t) A[s,t] g_s
    equals the analytic output to 1e-5; transposed/flipped decay variants of the same reference differ by
    >1e-2 relative, so a transposed Gamma in the code fails this test."""
    model = make_model(tiny_config(sequence_len=8), CPU, 42)
    randomize_gates(model)
    model.window_sizes[0] = (3, 0)
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, 4096, (1, 8), generator=g)
    y = torch.randint(0, 4096, (1, 8), generator=g)
    est = DUST(model, tiny_recipe(), seed=0)
    try:
        with torch.no_grad():
            est.capture(x, y)
            ghat = {i: torch.randn(1, 8, 64, generator=g) for i in range(2)}
            out = est.analytic_value_errors(ghat, 0.5)
        assert set(out) == {'transformer.h.0.attn.c_v', 'transformer.h.1.attn.c_v', 'value_embeds.1'}
        for i in range(2):
            got = out[f'transformer.h.{i}.attn.c_v'].double().view(1, 8, 4, 16)
            ref = dense_reference(est, model, i, ghat[i], 0.5)
            assert (got - ref).abs().max() < 1e-5, (i, (got - ref).abs().max())
            for variant in ('flipped', 'diagonal'):
                wrong = dense_reference(est, model, i, ghat[i], 0.5, variant)
                assert (got - wrong).norm() / got.norm() > 1e-2, (i, variant)
            if i == 1:
                z = est.outputs['transformer.h.1.attn.ve_gate'].double()
                gate = (2 * torch.sigmoid(z)).unsqueeze(-1)
                assert (gate - 1).abs().max() > 0.05
                got_ve = out['value_embeds.1'].double().view(1, 8, 4, 16)
                assert (got_ve - ref * gate).abs().max() < 1e-5
    finally:
        est.close()


@pytest.fixture(scope='module')
def sampled_mean():
    """K=4096 sampled 'v' and 've' estimates (nrep=2, gamma=0.5), b=1 to keep it fast."""
    model = make_model(tiny_config(), CPU, 42)
    randomize_gates(model)
    g = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, 4096, (1, 129), generator=g)
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    targets = {i: torch.randn(1, 128, 64, generator=g) for i in range(2)}
    est = DUST(model, tiny_recipe(gamma=0.5, nrep=2), seed=7)
    try:
        with torch.no_grad():
            est.capture(x, y)
            exact = est.analytic_value_errors(targets, 0.5)
            sampled_v = est.local_errors('v', 4096, targets)
            sampled_ve = est.local_errors('ve', 4096, targets)
    finally:
        est.close()
    return exact, sampled_v, sampled_ve


@pytest.mark.parametrize('site', ['transformer.h.0.attn.c_v', 'transformer.h.1.attn.c_v'])
def test_sampled_v_mean_matches_analytic(sampled_mean, site):
    """Mean of local_errors('v', 4096) at gamma=0.5, nrep=2 has cos > 0.98 and norm ratio within 5% of the
    analytic value. Catches a scale mismatch (e.g. missing 1/sigma or draws factor) or a transposed Gamma."""
    exact, sampled_v, _ = sampled_mean
    cos, ratio = cos_and_ratio(sampled_v[site], exact[site])
    print(f'MEASURED v {site}: cos={cos:.5f} ratio={ratio:.5f}')
    assert cos > 0.98 and abs(ratio - 1) < 0.05, (cos, ratio)


def test_sampled_ve_mean_matches_analytic(sampled_mean):
    """Same for 've' on block 1, which includes the gate multiplier (randomized, so a missing gate fails)."""
    exact, _, sampled_ve = sampled_mean
    cos, ratio = cos_and_ratio(sampled_ve['value_embeds.1'], exact['value_embeds.1'])
    print(f'MEASURED ve: cos={cos:.5f} ratio={ratio:.5f}')
    assert cos > 0.98 and abs(ratio - 1) < 0.05, (cos, ratio)
