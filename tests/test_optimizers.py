"""E2-WP1t: optimizer flag for the backprop baseline (baselines/optim.py, baselines/backprop.py).

Expected values come from the unmodified upstream backprop.py (git show main:...), an SVD reference, a hand-written
Newton-Schulz and hand-computed momentum buffers, never from the code under test.
"""
import importlib.util
import io
import subprocess
import sys
from contextlib import redirect_stderr
from pathlib import Path

import pytest
import torch

from diagnostics.tiny import tiny_config
from dust import make_model

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device('cpu')
NS_COEF = (3.4445, -4.7750, 2.0315)


# ---------------------------------------------------------------- helpers

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def optim():
    return load_module('e2wp1_optim', ROOT / 'baselines' / 'optim.py')


@pytest.fixture(scope='module')
def new_bp():
    return load_module('e2wp1_backprop_new', ROOT / 'baselines' / 'backprop.py')


@pytest.fixture(scope='module')
def upstream_bp(tmp_path_factory):
    path = tmp_path_factory.mktemp('upstream_bp') / 'backprop_upstream.py'
    try:
        src = subprocess.run(['git', 'show', 'main:baselines/backprop.py'], cwd=ROOT,
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('git or main:baselines/backprop.py unavailable')
    path.write_bytes(src)
    return load_module('e2wp1_backprop_upstream', path)


def ref_ns(G, steps=5):
    """Hand-written quintic Newton-Schulz, straight from the brief's contract."""
    a, b, c = NS_COEF
    X = G.float()
    X = X / (X.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if tall else X


def tiny_batch():
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 4096, (2, 129), generator=g)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def fresh_model(n_layer=2):
    return make_model(tiny_config(n_layer=n_layer), CPU, 42).requires_grad_(True)


# ---------------------------------------------------------------- 1. SGD unchanged

def test_sgd_bitwise_equals_upstream(upstream_bp, new_bp, optim):
    """build_optimizer('sgd') + list-taking train_step reproduce upstream make_optimizer + train_step bitwise
    over 3 steps (params and losses), and the param groups match in lr, momentum, order and parameter identity.
    Catches a changed wte lr (1000), swapped group order, wrong momentum, a different update rule, or train_step
    not stepping every optimizer in the list."""
    lr, mom = 0.175, 0.98
    x, y = tiny_batch()
    m_old, m_new = fresh_model(), fresh_model()
    o_old = upstream_bp.make_optimizer(m_old, lr, mom)
    o_new = optim.build_optimizer(m_new, 'sgd', lr, mom)
    assert isinstance(o_new, list) and len(o_new) == 1

    names_old = {id(p): n for n, p in m_old.named_parameters()}
    names_new = {id(p): n for n, p in m_new.named_parameters()}
    assert len(o_new[0].param_groups) == len(o_old.param_groups) == 2
    for go, gn in zip(o_old.param_groups, o_new[0].param_groups):
        assert go['lr'] == gn['lr'] and go['momentum'] == gn['momentum']
        assert [names_old[id(p)] for p in go['params']] == [names_new[id(p)] for p in gn['params']]
    assert o_new[0].param_groups[1]['lr'] == 1000.0

    for _ in range(3):
        l_old = upstream_bp.train_step(m_old, o_old, x, y)
        l_new = new_bp.train_step(m_new, o_new, x, y)
        assert l_old == l_new
    for (n, p), (_, q) in zip(m_old.named_parameters(), m_new.named_parameters()):
        assert torch.equal(p, q), n


# ---------------------------------------------------------------- 2. Newton-Schulz vs SVD

@pytest.mark.parametrize('shape', [(64, 64), (64, 256), (256, 64)])
def test_ns_matches_svd_polar(optim, shape):
    """Uᵀ NS(G) V is diagonal (off-diagonal Frobenius < 2% of the diagonal's) with entries in [0.6, 1.25].
    Catches a wrong coefficient, a missing transpose for tall matrices (256,64 would give a non-diagonal or
    wrong-shaped result), a missing normalisation, or fewer than 5 iterations.
    Measured with a hand reference: off-diagonal ratio ~1.5e-6; wide/tall diag range [0.68, 1.13].
    Square Gaussian (64,64) has a few singular values < 1e-3*||G||_F (smallest ~1e-4), which quintic NS cannot
    lift to 0.6 (measured diag 0.04), so the range check there only covers sv >= 5e-3*||G||_F, as for the ill-conditioned
    case (the brief's blanket [0.6,1.25] fails for a random square matrix even for a correct NS)."""
    g = torch.Generator().manual_seed(0)
    G = torch.randn(*shape, generator=g)
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    M = U.T @ optim.zeropower_newton_schulz(G) @ Vh.T
    d = M.diagonal()
    keep = S >= 5e-3 * G.norm()
    assert M.shape == (min(shape), min(shape))
    off = (M - torch.diag(d)).norm()
    assert off < 0.02 * d.norm()
    assert d[keep].min() >= 0.6 and d[keep].max() <= 1.25
    if shape[0] != shape[1]:
        assert keep.all()


def test_ns_output_shape_and_dtype(optim):
    """Output has G's shape (tall and wide) and is float32. Catches forgetting to transpose back."""
    for shape in [(8, 24), (24, 8), (16, 16)]:
        out = optim.zeropower_newton_schulz(torch.randn(*shape))
        assert out.shape == shape and out.dtype == torch.float32


def test_ns_scale_invariant(optim):
    """NS(10 G) == NS(G) to rel 1e-5 (normalisation by the Frobenius norm; eps 1e-7 shifts it ~1e-8).
    Catches a missing normalisation (the quintic diverges for large inputs)."""
    g = torch.Generator().manual_seed(2)
    for shape in [(64, 64), (64, 256), (256, 64)]:
        G = torch.randn(*shape, generator=g)
        a, b = optim.zeropower_newton_schulz(G), optim.zeropower_newton_schulz(10 * G)
        assert (a - b).norm() / a.norm() < 1e-5


def test_ns_matches_hand_reference(optim):
    """NS equals the contract's loop written out by hand (rel 1e-5), and `steps` is honoured: steps=1 differs
    from steps=5 and matches the one-step hand reference. Catches a hard-coded iteration count."""
    g = torch.Generator().manual_seed(4)
    for shape in [(32, 96), (96, 32)]:
        G = torch.randn(*shape, generator=g)
        for steps in (1, 5):
            a, r = optim.zeropower_newton_schulz(G, steps), ref_ns(G, steps)
            assert (a - r).norm() / r.norm() < 1e-5
        assert (ref_ns(G, 1) - ref_ns(G, 5)).norm() > 0.1 * ref_ns(G, 5).norm()


def test_ns_ill_conditioned(optim):
    """G with singular values logspace(1e-3, 1): the diagonal in the SVD basis stays in [0.6, 1.25] for sv >= 1e-2.
    Measured with a hand reference: in range down to sv ~3e-3 (||G||_F ~ 2.3), so 1e-2 has margin.
    Catches too few iterations or wrong coefficients (small singular values not amplified)."""
    g = torch.Generator().manual_seed(1)
    n = 64
    U, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    V, _ = torch.linalg.qr(torch.randn(n, n, generator=g))
    S = torch.logspace(-3, 0, n)
    G = U @ torch.diag(S) @ V.T
    U2, S2, Vh2 = torch.linalg.svd(G)
    d = (U2.T @ optim.zeropower_newton_schulz(G) @ Vh2.T).diagonal()
    keep = S2 >= 1e-2
    assert keep.sum() > 20
    assert d[keep].min() >= 0.6 and d[keep].max() <= 1.25


# ---------------------------------------------------------------- 3. Muon step math

@pytest.mark.parametrize('shape', [(96, 32), (32, 96), (48, 48)])
def test_muon_step_momentum_zero(optim, shape):
    """momentum 0: p_new = p - lr*max(1, r/c)**0.5 * NS(g). Compared on the update (p - p_new) with rel 1e-5
    (lr=0.5 so the update is not lost in fp32 rounding of p). Catches a missing/inverted aspect-ratio factor
    (tall gets sqrt(3), wide gets 1), wrong sign, wrong lr use, or a missing nesterov term at momentum 0."""
    g = torch.Generator().manual_seed(5)
    p = torch.nn.Parameter(torch.randn(*shape, generator=g))
    grad = torch.randn(*shape, generator=g)
    p0 = p.detach().clone()
    lr = 0.5
    opt = optim.Muon([p], lr=lr, momentum=0.0)
    p.grad = grad.clone()
    opt.step()
    expected = lr * max(1, shape[0] / shape[1]) ** 0.5 * ref_ns(grad)
    delta = p0 - p.detach()
    assert (delta - expected).norm() / expected.norm() < 1e-5


@pytest.mark.parametrize('nesterov', [True, False])
def test_muon_two_steps_momentum(optim, nesterov):
    """momentum 0.5, two steps with g1, g2. buf1 = g1; buf2 = 0.5*buf1 + g2; u = g + 0.5*buf (nesterov) or buf.
    Compared on the total update with rel 1e-5. Catches nesterov applied to the wrong
    quantity, buffer reset each step, or the nesterov flag ignored."""
    g = torch.Generator().manual_seed(6)
    shape = (64, 32)
    p = torch.nn.Parameter(torch.randn(*shape, generator=g))
    g1, g2 = torch.randn(*shape, generator=g), torch.randn(*shape, generator=g)
    p0 = p.detach().clone()
    lr, m = 0.5, 0.5
    opt = optim.Muon([p], lr=lr, momentum=m, nesterov=nesterov)
    for gr in (g1, g2):
        p.grad = gr.clone()
        opt.step()
    buf1 = g1
    buf2 = m * buf1 + g2
    u1 = g1 + m * buf1 if nesterov else buf1
    u2 = g2 + m * buf2 if nesterov else buf2
    scale = lr * max(1, shape[0] / shape[1]) ** 0.5
    expected = scale * (ref_ns(u1) + ref_ns(u2))
    delta = p0 - p.detach()
    assert (delta - expected).norm() / expected.norm() < 1e-5


def test_muon_defaults_and_validation(optim):
    """Defaults momentum=0.95, nesterov=True, ns_steps=5; a 1-D parameter raises ValueError (so biases/lambdas can
    never silently go through NS); a param without grad is left alone."""
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = optim.Muon([p], lr=0.1)
    gp = opt.param_groups[0]
    assert gp['momentum'] == 0.95 and gp['nesterov'] is True and gp['ns_steps'] == 5
    before = p.detach().clone()
    opt.step()
    assert torch.equal(p.detach(), before)
    with pytest.raises(ValueError):
        optim.Muon([torch.nn.Parameter(torch.randn(8))], lr=0.1)


# ---------------------------------------------------------------- 4. Partition

def expected_muon_names(model):
    return {n for n, p in model.named_parameters()
            if n.startswith('transformer.h.') and p.ndim == 2 and 've_gate' not in n}


@pytest.mark.parametrize('n_layer', [2, 4])
def test_partition_exact(optim, n_layer):
    """Every parameter is in exactly one list; muon_params is exactly the 2-D block weights (c_q, c_k, c_v,
    attn.c_proj, mlp.c_fc, mlp.c_proj) minus ve_gate; the rest (wte, value_embeds, lm_head, lambdas, ve_gate) is Adam.
    Catches a duplicated or dropped parameter, ve_gate or embeddings or lm_head sent to Muon, lambdas (1-D) to Muon."""
    model = fresh_model(n_layer)
    muon, adam = optim.partition(model)
    ids_m, ids_a = [id(p) for p in muon], [id(p) for p in adam]
    all_ids = [id(p) for p in model.parameters()]
    assert len(ids_m) == len(set(ids_m)) and len(ids_a) == len(set(ids_a))
    assert not set(ids_m) & set(ids_a)
    assert set(ids_m) | set(ids_a) == set(all_ids)
    by_id = {id(p): n for n, p in model.named_parameters()}
    assert {by_id[i] for i in ids_m} == expected_muon_names(model)
    assert all(p.ndim == 2 for p in muon)
    adam_names = {by_id[i] for i in ids_a}
    for must in ('transformer.wte.weight', 'lm_head.weight', 'resid_lambdas', 'x0_lambdas'):
        assert must in adam_names
    assert any(n.startswith('value_embeds.') for n in adam_names)
    assert {n for n in adam_names if 've_gate' in n}, 'tiny model has no ve_gate; precondition of this test'
    assert all(not n.startswith('transformer.h.') or 've_gate' in n for n in adam_names)


@pytest.mark.parametrize('n_layer', [2, 4])
def test_build_muon(optim, n_layer):
    """'muon' returns [Muon, AdamW] covering every parameter exactly once; Muon gets lr and momentum, AdamW gets
    adam_lr (default 0.008, override honoured), betas (0.9, 0.95), eps 1e-8, weight_decay 0.
    Catches wrong lr routing (lr to Adam), a dropped adam_lr override, or non-zero weight decay."""
    model = fresh_model(n_layer)
    opts = optim.build_optimizer(model, 'muon', 0.02, 0.9)
    assert len(opts) == 2
    assert isinstance(opts[0], optim.Muon) and isinstance(opts[1], torch.optim.AdamW)
    ids = [id(p) for o in opts for g in o.param_groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    assert opts[0].param_groups[0]['lr'] == 0.02 and opts[0].param_groups[0]['momentum'] == 0.9
    ga = opts[1].param_groups[0]
    assert ga['lr'] == 0.008 and tuple(ga['betas']) == (0.9, 0.95) and ga['eps'] == 1e-8 and ga['weight_decay'] == 0
    opts2 = optim.build_optimizer(model, 'muon', 0.02, 0.9, adam_lr=0.003)
    assert opts2[1].param_groups[0]['lr'] == 0.003 and opts2[0].param_groups[0]['lr'] == 0.02


# ---------------------------------------------------------------- 5. AdamW wiring, training sanity

def test_build_adamw(optim):
    """'adamw': one AdamW, one group with every parameter once, betas (0.9, 0.95), eps 1e-8, weight_decay 0, lr as
    given (momentum None accepted). Catches default betas (0.999), default weight_decay 0.01, or a split into groups."""
    model = fresh_model()
    opts = optim.build_optimizer(model, 'adamw', 1e-3, None)
    assert len(opts) == 1 and isinstance(opts[0], torch.optim.AdamW)
    assert len(opts[0].param_groups) == 1
    g = opts[0].param_groups[0]
    assert g['lr'] == 1e-3 and tuple(g['betas']) == (0.9, 0.95) and g['eps'] == 1e-8 and g['weight_decay'] == 0
    assert [id(p) for p in g['params']] == [id(p) for p in model.parameters()] or \
        {id(p) for p in g['params']} == {id(p) for p in model.parameters()}
    assert len(g['params']) == len(list(model.parameters()))


def test_build_unknown_name_raises(optim):
    """An unknown optimizer name must not silently fall back to sgd."""
    with pytest.raises((ValueError, KeyError, AssertionError)):
        optim.build_optimizer(fresh_model(), 'lion', 1e-3, 0.9)


@pytest.mark.parametrize('name,lr,mom,adam_lr', [
    ('sgd', 0.05, 0.9, None), ('adamw', 1e-3, None, None), ('muon', 0.02, 0.95, 0.008)])
def test_training_decreases_loss(optim, new_bp, name, lr, mom, adam_lr):
    """20 train_step calls on one fixed tiny batch drop the loss from ~ln(4096)=8.3 by more than 0.5 for each
    optimizer (hand-run reference: sgd 8.32->6.9, adamw ->6.6, muon ->0.55, so wide margin). Catches a wrong sign,
    or NaNs (a not-stepped optimizer is caught by the next test)."""
    x, y = tiny_batch()
    model = fresh_model()
    opts = optim.build_optimizer(model, name, lr, mom, adam_lr) if adam_lr is not None \
        else optim.build_optimizer(model, name, lr, mom)
    losses = [new_bp.train_step(model, opts, x, y) for _ in range(20)]
    assert all(l == l for l in losses)
    assert losses[-1] < losses[0] - 0.5


def test_train_step_steps_every_optimizer(optim, new_bp):
    """With Muon + AdamW, one train_step changes a Muon parameter (c_q) and Adam-only parameters (lm_head, lambdas).
    Also, with stub optimizers, zero_grad and step are each called exactly once on every optimizer in the list.
    Catches train_step stepping or zeroing only optimizers[0]."""
    x, y = tiny_batch()
    model = fresh_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    new_bp.train_step(model, optim.build_optimizer(model, 'muon', 0.02, 0.95), x, y)
    after = dict(model.named_parameters())
    for n in ('transformer.h.0.attn.c_q.weight', 'lm_head.weight', 'resid_lambdas'):
        assert not torch.equal(before[n], after[n]), n

    class Stub:
        def __init__(self):
            self.zeroed = self.stepped = 0

        def zero_grad(self, set_to_none=True):
            self.zeroed += 1

        def step(self):
            self.stepped += 1

    stubs = [Stub(), Stub()]
    new_bp.train_step(fresh_model(), stubs, x, y)
    assert [(s.zeroed, s.stepped) for s in stubs] == [(1, 1), (1, 1)]


# ---------------------------------------------------------------- 6. CLI

def run_cli(new_bp, argv, tmp_path):
    """Run the argparse path; return (exit_code, last stderr line). Prefers parse_args(argv) if the module has it."""
    err = io.StringIO()
    old = sys.argv
    sys.argv = ['backprop.py'] + argv + ['--data-dir', str(tmp_path / 'nodata'), '--output', str(tmp_path / 'out')]
    try:
        with redirect_stderr(err):
            try:
                if hasattr(new_bp, 'parse_args'):
                    new_bp.parse_args(sys.argv[1:])
                else:
                    new_bp.main()
                code = 0
            except SystemExit as e:
                code = e.code
    finally:
        sys.argv = old
    lines = [l for l in err.getvalue().strip().splitlines() if l.strip()]
    return code, (lines[-1] if lines else '')


def test_cli_muon_requires_lr(new_bp, tmp_path):
    """`--optimizer muon` without --lr exits with an argparse error that mentions lr (not the CUDA or data error);
    `adamw` likewise. With --lr given the error (if any, on a CPU box) is not about lr.
    Catches silently falling back to the sgd default lr 0.175 for Muon/AdamW."""
    for name in ('muon', 'adamw'):
        code, last = run_cli(new_bp, ['--optimizer', name], tmp_path)
        assert code not in (0, None), name
        assert 'lr' in last.lower(), (name, last)
    code, last = run_cli(new_bp, ['--optimizer', 'muon', '--lr', '0.02'], tmp_path)
    assert 'lr' not in last.lower().replace('--adam-lr', ''), last


def test_cli_rejects_unknown_optimizer(new_bp, tmp_path):
    """`--optimizer lion` is rejected by argparse choices."""
    code, _ = run_cli(new_bp, ['--optimizer', 'lion', '--lr', '0.1'], tmp_path)
    assert code not in (0, None)
