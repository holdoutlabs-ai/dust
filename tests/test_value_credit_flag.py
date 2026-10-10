"""WP1 acceptance 4-5: the sampled path is unchanged from upstream, and the flag plumbing."""
import importlib.util
from dataclasses import asdict
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import dust as new_dust
from diagnostics.tiny import tiny_config, tiny_recipe
from dust import DEFAULT_CONFIGS, DUST, make_model, recipe

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device('cpu')


class NoOpOptimizer:
    def zero_grad(self, set_to_none=True):
        pass

    def step(self):
        pass


def fresh_model(n_layer=2):
    model = make_model(tiny_config(n_layer=n_layer), CPU, 42)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for block in model.transformer.h:
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight.copy_(torch.randn(block.attn.ve_gate.weight.shape, generator=g) * 0.5)
    return model


@pytest.fixture(scope='module')
def upstream(tmp_path_factory):
    path = tmp_path_factory.mktemp('upstream') / 'dust_upstream.py'
    try:
        src = subprocess.run(['git', 'show', 'main:dust.py'], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('git or main:dust.py unavailable')
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location('dust_upstream', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['dust_upstream'] = module
    spec.loader.exec_module(module)
    return module


def masked_batch():
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 4096, (2, 129), generator=g)
    x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    y[0, :5] = -1
    return x, y


def run_step(module, model, settings, x, y):
    est = module.DUST(model, settings, seed=123)
    try:
        loss = est.step(x, y, NoOpOptimizer())
    finally:
        est.close()
    return loss, {n: p.grad.clone() for n, p in model.named_parameters()}


def test_sampled_step_bitwise_equals_upstream(upstream):
    """New DUST.step (default value_credit) gives bitwise-equal parameter.grad and loss to upstream's step for
    the same seed. Catches any change of RNG consumption order, extra draws, or altered scaling in 'sampled'."""
    x, y = masked_batch()
    settings = tiny_recipe()
    loss_up, grads_up = run_step(upstream, fresh_model(), settings, x, y)
    loss_new, grads_new = run_step(new_dust, fresh_model(), settings, x, y)
    assert loss_up == loss_new
    assert grads_up.keys() == grads_new.keys()
    for name in grads_up:
        assert torch.equal(grads_up[name], grads_new[name]), name


@pytest.mark.parametrize('nrep', [1, 2])
def test_estimate_matches_step_and_upstream(upstream, nrep):
    """estimate(x, y)[name] / valid_tokens equals the step's .grad bitwise (summed-loss scale; y has -1 tokens
    so valid_tokens matters), and equals upstream's step too. nrep=1 and 2 both covered."""
    x, y = masked_batch()
    settings = tiny_recipe(nrep=nrep)
    _, grads_up = run_step(upstream, fresh_model(), settings, x, y)
    _, grads_new = run_step(new_dust, fresh_model(), settings, x, y)
    model = fresh_model()
    est = DUST(model, settings, seed=123)
    try:
        gradients = est.estimate(x, y)
    finally:
        est.close()
    valid = (y != -1).sum()
    assert set(gradients) == set(grads_new)
    for name, grad in grads_new.items():
        assert torch.equal(gradients[name] / valid, grad), name
        assert torch.equal(grads_up[name], grad), name


def test_recipes_equal_upstream(upstream):
    """For every tokens x population, asdict(recipe) equals upstream's apart from the two new keys."""
    assert upstream.DEFAULT_CONFIGS == DEFAULT_CONFIGS
    for tokens in DEFAULT_CONFIGS:
        for pop in DEFAULT_CONFIGS[tokens]['optimizer']:
            new = asdict(recipe(tokens, pop))
            assert new.pop('value_credit') == 'sampled'
            assert new.pop('value_gamma') is None
            assert new == asdict(upstream.recipe(tokens, pop)), (tokens, pop)


def test_recipe_analytic_zeroes_only_v_and_ve():
    """recipe('1M', 256, 'analytic') has local v == ve == 0 and every other count unchanged (no redistribution)."""
    base, analytic = recipe('1M', 256), recipe('1M', 256, 'analytic')
    assert analytic.value_credit == 'analytic'
    assert analytic.local['v'] == 0 and analytic.local['ve'] == 0
    for k in ('q', 'gate', 'k'):
        assert analytic.local[k] == base.local[k]
    for field in ('writers', 'hidden', 'hubs', 'embedding', 'head', 'nrep', 'lr', 'momentum', 'gamma'):
        assert getattr(analytic, field) == getattr(base, field), field
    assert recipe('1M', 256, value_gamma=1.0).value_gamma == 1.0
    assert recipe('1M', 256, 'analytic', value_gamma=1.0).value_gamma == 1.0
    assert recipe('1M', 256, 'sampled').local == base.local


def test_bad_value_credit_raises():
    """An unknown value_credit raises ValueError from recipe() and from Recipe construction."""
    with pytest.raises(ValueError):
        recipe('1M', 256, 'bogus')
    with pytest.raises(ValueError):
        tiny_recipe(value_credit='bogus')
    assert tiny_recipe(value_credit='analytic').value_credit == 'analytic'


@pytest.mark.parametrize('nrep', [1, 2])
@pytest.mark.parametrize('zero_draws', [False, True])
def test_analytic_estimate_finite_for_every_parameter(nrep, zero_draws):
    """Analytic estimate() returns a finite gradient of the right shape for every parameter name, with v/ve
    draws left in the recipe or set to 0 (as recipe() does)."""
    settings = tiny_recipe(nrep=nrep, value_credit='analytic')
    if zero_draws:
        settings.local['v'] = settings.local['ve'] = 0
    x, y = masked_batch()
    model = fresh_model()
    est = DUST(model, settings, seed=5)
    try:
        gradients = est.estimate(x, y)
    finally:
        est.close()
    for name, p in model.named_parameters():
        assert name in gradients, name
        assert gradients[name].shape == p.shape, name
        assert torch.isfinite(gradients[name]).all(), name
    assert gradients['transformer.h.0.attn.c_v.weight'].abs().sum() > 0
    assert gradients['value_embeds.1.weight'].abs().sum() > 0


def test_analytic_uses_analytic_errors_and_leaves_earlier_rng_alone():
    """In analytic mode the c_v / value_embeds gradients are reassemble(analytic_value_errors(targets, gamma))
    for the very targets used; everything computed before 'v' in the local loop (writers, hidden, embedding,
    head, q) is bitwise equal to sampled mode with the same seed, i.e. the analytic call consumes no RNG and
    the sampled path of the other kinds is untouched."""
    x, y = masked_batch()
    s_sampled, s_analytic = tiny_recipe(), tiny_recipe(value_credit='analytic')
    est = DUST(fresh_model(), s_sampled, seed=9)
    try:
        g_sampled = est.estimate(x, y)
    finally:
        est.close()
    est = DUST(fresh_model(), s_analytic, seed=9)
    seen = {}
    original = est.analytic_value_errors

    def spy(*args, **kwargs):
        seen['args'], seen['kwargs'] = args, kwargs
        seen['out'] = original(*args, **kwargs)
        return seen['out']

    est.analytic_value_errors = spy
    try:
        g_analytic = est.estimate(x, y)
        targets = seen['args'][0] if seen['args'] else seen['kwargs']['targets']
        gamma = seen['args'][1] if len(seen['args']) > 1 else seen['kwargs']['gamma']
        assert gamma == s_analytic.gamma
        expected = {s + '.weight': est.reassemble(s, e) for s, e in original(targets, gamma).items()}
    finally:
        est.close()
    assert 'value_embeds.1.weight' in expected and 'transformer.h.0.attn.c_v.weight' in expected
    for name, value in expected.items():
        assert torch.equal(g_analytic[name], value), name
    c_v = 'transformer.h.0.attn.c_v.weight'
    assert not torch.equal(g_sampled[c_v], g_analytic[c_v])
    same = [n for n in g_sampled if 'c_v' not in n and 'value_embeds' not in n and 've_gate' not in n
            and 'c_k' not in n and n not in ('resid_lambdas', 'x0_lambdas')]
    assert 'lm_head.weight' in same and 'transformer.h.0.attn.c_q.weight' in same
    for name in same:
        assert torch.equal(g_sampled[name], g_analytic[name]), name


def test_value_gamma_overrides_and_none_uses_gamma():
    """value_gamma=None passes settings.gamma to analytic_value_errors; value_gamma=1.0 / 0.4 pass those. A
    mixed-up gamma/value_gamma (variant B vs C) would be caught."""
    x, y = masked_batch()
    for value_gamma, expected in ((None, 0.7), (1.0, 1.0), (0.4, 0.4)):
        settings = tiny_recipe(gamma=0.7, value_credit='analytic', value_gamma=value_gamma)
        est = DUST(fresh_model(), settings, seed=9)
        seen = []
        original = est.analytic_value_errors

        def spy(*args, **kwargs):
            seen.append(args[1] if len(args) > 1 else kwargs['gamma'])
            return original(*args, **kwargs)

        est.analytic_value_errors = spy
        try:
            est.estimate(x, y)
        finally:
            est.close()
        assert seen and all(g == expected for g in seen), (value_gamma, seen)


@pytest.mark.parametrize('kind', ['q', 'v', 'gate', 'k', 've'])
def test_local_errors_zero_draws_returns_zeros(kind):
    """local_errors(kind, 0, targets) returns zeros of the right shape without ZeroDivisionError/NaN
    (the old code divided by draws)."""
    x, y = masked_batch()
    est = DUST(fresh_model(), tiny_recipe(), seed=1)
    try:
        with torch.no_grad():
            est.capture(x, y)
            g = torch.Generator().manual_seed(2)
            targets = {i: torch.randn(2, 128, 64, generator=g) for i in range(2)}
            out = est.local_errors(kind, 0, targets)
    finally:
        est.close()
    assert out
    for site, e in out.items():
        assert e.shape[:2] == (2, 128) and torch.isfinite(e).all(), site
        assert (e == 0).all(), site
