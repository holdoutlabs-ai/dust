"""Tests for diagnostics/cosine.py (WP3), written from the brief and the math."""
import dataclasses
import json
import math

import pytest
import torch

from diagnostics.tiny import tiny_config, tiny_recipe
from dust import make_model
from diagnostics import cosine as C

TYPES = {'wte', 'lm_head', 'lambdas', 've', 'q', 'k', 'v', 'gate', 'attn_out', 'mlp_in', 'mlp_out'}

N0 = 'transformer.h.0.mlp.c_fc.weight'
N1 = 'transformer.h.1.mlp.c_fc.weight'
NO = 'transformer.h.0.mlp.c_proj.weight'
NV = 'transformer.h.0.attn.c_v.weight'
NVE = 'value_embeds.1.weight'


def _cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


def _g(seed, *shape):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- 1. compare

def test_compare_identical():
    """Identical est/true must give cos 1 and ratio 1 (catches wrong normalisation)."""
    t = {N0: _g(0, 4, 3), NO: _g(1, 3, 4)}
    r = C.compare({k: v.clone() for k, v in t.items()}, t)
    for n in t:
        assert r['per_param'][n]['cos'] == pytest.approx(1, abs=1e-6)
        assert r['per_param'][n]['ratio'] == pytest.approx(1, abs=1e-6)
    assert r['full'] == pytest.approx(1, abs=1e-6)


def test_compare_scaled_and_negated():
    """est=2*true -> cos 1, ratio 2 (ratio is est/true, not inverted); est=-true -> cos -1."""
    t = {N0: _g(2, 4, 3)}
    r = C.compare({N0: 2 * t[N0]}, t)
    assert r['per_param'][N0]['cos'] == pytest.approx(1, abs=1e-6)
    assert r['per_param'][N0]['ratio'] == pytest.approx(2, abs=1e-6)
    assert r['per_param'][N0]['type'] == 'mlp_in'
    r = C.compare({N0: -t[N0]}, t)
    assert r['per_param'][N0]['cos'] == pytest.approx(-1, abs=1e-6)


def test_compare_per_type_mean_of_cosines():
    """per_type must average per-parameter cos/ratio (two mlp_in params), checked by hand."""
    t = {N0: _g(3, 5), N1: _g(4, 7)}
    e = {N0: t[N0] + _g(5, 5), N1: 3 * t[N1] + _g(6, 7)}
    r = C.compare(e, t)
    c0, c1 = _cos(e[N0], t[N0]), _cos(e[N1], t[N1])
    r0, r1 = float(e[N0].norm() / t[N0].norm()), float(e[N1].norm() / t[N1].norm())
    assert r['per_type']['mlp_in']['cos_mean'] == pytest.approx((c0 + c1) / 2, abs=1e-5)
    assert r['per_type']['mlp_in']['ratio_mean'] == pytest.approx((r0 + r1) / 2, abs=1e-5)


def test_compare_full_is_concatenation_not_mean():
    """full = cos of the concatenation. Here a big param is aligned and a small one anti-aligned,
    so concat cos (~0.98) differs from mean of per-param cos (0). A mean-of-cos bug fails."""
    a = torch.tensor([10., 0, 0, 0])
    b = torch.tensor([0., 1., 0., 0.])
    t = {N0: a, NO: b}
    e = {N0: a.clone(), NO: -b}
    r = C.compare(e, t)
    want = _cos(torch.cat([e[N0], e[NO]]), torch.cat([t[N0], t[NO]]))
    assert want == pytest.approx(99 / 101, abs=1e-9)
    assert r['full'] == pytest.approx(want, abs=1e-6)
    assert r['full'] != pytest.approx((1 + -1) / 2, abs=0.1)


def test_compare_value_subset_only_v_and_ve():
    """value_subset is the concatenation of c_v and value_embeds params only: the other (anti-aligned)
    parameter must not leak in."""
    t = {NV: _g(7, 6), NVE: _g(8, 6), NO: _g(9, 6)}
    e = {NV: t[NV] + 0.3 * _g(10, 6), NVE: t[NVE] + 0.3 * _g(11, 6), NO: -t[NO]}
    r = C.compare(e, t)
    want = _cos(torch.cat([e[NV], e[NVE]]), torch.cat([t[NV], t[NVE]]))
    assert r['value_subset'] == pytest.approx(want, abs=1e-5)


def test_compare_zero_gives_nan_not_exception():
    """A zero-norm est or true yields nan cos instead of raising or returning 0/inf."""
    t = {N0: _g(12, 4), NO: _g(13, 4)}
    e = {N0: torch.zeros(4), NO: t[NO].clone()}
    r = C.compare(e, t)
    assert math.isnan(r['per_param'][N0]['cos'])
    assert r['per_param'][NO]['cos'] == pytest.approx(1, abs=1e-6)
    r = C.compare({N0: t[N0].clone(), NO: t[NO].clone()}, {N0: torch.zeros(4), NO: t[NO]})
    assert math.isnan(r['per_param'][N0]['cos'])


# ---------------------------------------------------------------- 2. layer_type

@pytest.mark.parametrize('n_layer', [2, 4])
def test_layer_type_covers_tiny_model(n_layer):
    """Every tiny parameter maps to one of the 11 types and all 11 appear (a dropped or renamed
    parameter class would fail). Also spot-checks the name rules from the brief."""
    model = make_model(tiny_config(n_layer=n_layer), torch.device('cpu'), 42)
    seen = {C.layer_type(n) for n, _ in model.named_parameters()}
    assert seen == TYPES
    assert C.layer_type('transformer.wte.weight') == 'wte'
    assert C.layer_type('lm_head.weight') == 'lm_head'
    assert C.layer_type('resid_lambdas') == 'lambdas'
    assert C.layer_type('x0_lambdas') == 'lambdas'
    assert C.layer_type('value_embeds.1.weight') == 've'
    assert C.layer_type('transformer.h.0.attn.c_q.weight') == 'q'
    assert C.layer_type('transformer.h.0.attn.c_k.weight') == 'k'
    assert C.layer_type('transformer.h.0.attn.c_v.weight') == 'v'
    assert C.layer_type('transformer.h.1.attn.ve_gate.weight') == 'gate'
    assert C.layer_type('transformer.h.0.attn.c_proj.weight') == 'attn_out'
    assert C.layer_type('transformer.h.0.mlp.c_fc.weight') == 'mlp_in'
    assert C.layer_type('transformer.h.0.mlp.c_proj.weight') == 'mlp_out'


def test_layer_type_unknown_raises():
    """An unknown parameter must raise ValueError, not be silently dropped."""
    with pytest.raises(ValueError):
        C.layer_type('transformer.h.0.something.new.weight')


# ---------------------------------------------------------------- 3. autograd

def test_autograd_gradients_match_manual_and_restore_state(tiny_model, tiny_batch):
    """Equals plain autograd of the mean-CE loss; afterwards requires_grad False and grad None
    (catches sum-vs-mean scaling and leaked autograd state that would break DUST's no_grad model)."""
    x, y = tiny_batch
    got = C.autograd_gradients(tiny_model, x, y)
    assert all(not p.requires_grad and p.grad is None for p in tiny_model.parameters())

    tiny_model.requires_grad_(True)
    loss = tiny_model(x, y)
    loss.backward()
    want = {n: p.grad.float().clone() for n, p in tiny_model.named_parameters()}
    tiny_model.zero_grad(set_to_none=True)
    tiny_model.requires_grad_(False)

    assert set(got) == set(want)
    assert any(w.abs().max() > 0 for w in want.values())
    for n in want:
        assert got[n].dtype == torch.float32
        torch.testing.assert_close(got[n], want[n], rtol=1e-5, atol=1e-8)
    assert all(not p.requires_grad and p.grad is None for p in tiny_model.parameters())


# ---------------------------------------------------------------- 4. pairing

@pytest.fixture(scope='module')
def paired():
    torch.manual_seed(0)
    model = make_model(tiny_config(), torch.device('cpu'), seed=42)
    with torch.no_grad():
        for b in model.transformer.h:
            if b.attn.ve_gate is not None:
                b.attn.ve_gate.weight.normal_(0, 0.5)
    tok = torch.randint(0, 4096, (2, 129), generator=torch.Generator().manual_seed(1))
    x, y = tok[:, :-1].contiguous(), tok[:, 1:].contiguous()
    base = tiny_recipe()
    out = {}
    for v in 'ABC':
        g, secs, peak = C.dust_gradients(model, C.variant_settings(base, v), x, y, 5)
        out[v] = g
        assert secs >= 0
    return out


def _is_shared_AB(n):
    return any(k in n for k in ('attn.c_proj', 'mlp.c_proj', 'mlp.c_fc', 'wte', 'lm_head', 'attn.c_q'))


def test_variant_settings():
    """B/C set analytic with local v=ve=0 (as recipe() does); C has gamma 1; A is unchanged sampled."""
    base = tiny_recipe(gamma=0.7)
    a, b, c = (C.variant_settings(base, v) for v in 'ABC')
    assert a.value_credit == 'sampled' and a.local == base.local
    for s in (b, c):
        assert s.value_credit == 'analytic' and s.local['v'] == 0 and s.local['ve'] == 0
        assert s.local['k'] == base.local['k'] and s.local['gate'] == base.local['gate']
    assert c.value_gamma == 1.0
    assert b.value_gamma in (None, 0.7)


def test_paired_A_vs_B_shared_params_bitwise(paired):
    """A and B must be bitwise equal on everything computed before the local-credit v loop
    (a changed RNG order or a changed draw count for other kinds would break this)."""
    A, B = paired['A'], paired['B']
    names = [n for n in A if _is_shared_AB(n)]
    assert names
    for n in names:
        assert torch.equal(A[n], B[n]), n


def test_paired_A_vs_B_value_params_differ(paired):
    """c_v and value_embeds estimates must differ between sampled and analytic."""
    for n in paired['A']:
        if 'attn.c_v' in n or n.startswith('value_embeds'):
            assert not torch.equal(paired['A'][n], paired['B'][n]), n


def test_paired_B_vs_C_bitwise_except_value_params(paired):
    """B and C (gamma 0.98 vs 1) differ only on c_v and value_embeds; all else bitwise equal."""
    B, Cc = paired['B'], paired['C']
    assert set(B) == set(Cc)
    n_diff = 0
    for n in B:
        if 'attn.c_v' in n or n.startswith('value_embeds'):
            assert not torch.equal(B[n], Cc[n]), n
            n_diff += 1
        else:
            assert torch.equal(B[n], Cc[n]), n
    assert n_diff >= 2


def test_dust_gradients_is_mean_loss_scale(paired, tiny_batch):
    """Estimates are divided by valid_tokens: norm of the full estimate must be of the same order as
    the autograd mean-loss gradient (a missing division would be ~256x too large)."""
    torch.manual_seed(0)
    model = make_model(tiny_config(), torch.device('cpu'), seed=42)
    x, y = tiny_batch
    true = C.autograd_gradients(model, x, y)
    g, _, _ = C.dust_gradients(model, C.variant_settings(tiny_recipe(), 'A'), x, y, 0)
    nt = torch.cat([true[n].flatten() for n in true]).norm()
    ne = torch.cat([g[n].flatten() for n in true]).norm()
    assert 0.01 < float(ne / nt) < 100


# ---------------------------------------------------------------- 5. end to end

def test_main_tiny_end_to_end(tmp_path):
    """CLI runs on tiny/CPU, writes JSON for A, B, C with finite full cosine > 0 and all 11 types."""
    out = tmp_path / 'c.json'
    C.main(['--config', 'tiny', '--seeds', '0,1', '--batches', '2', '--device', 'cpu',
            '--out', str(out)])
    d = json.loads(out.read_text())
    assert set(d['variants']) == {'A', 'B', 'C'}
    for v, r in d['variants'].items():
        assert math.isfinite(r['full']['mean']), v
        assert r['full']['mean'] > 0, v
        assert set(r['per_type']) == TYPES, v
        assert 'value_subset' in r
    assert len(d['records']) == 3 * 2 * 2
    assert 'command' in d and 'recipe' in d
