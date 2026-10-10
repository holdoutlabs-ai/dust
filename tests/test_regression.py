"""WP2 tests: diagnostics/regression.py (stage 3.1, analytic B is the mean of sampled A).

Expected values come from hand-built tensors and from the WP1 APIs used as the reference
(`analytic_value_errors` is the target the sampled estimator must converge to).
The module is imported lazily so the file collects before it exists.
"""
import importlib
import json

import pytest
import torch


def reg():
    return importlib.import_module('diagnostics.regression')


def flat_cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()))


# ---- 1. compare on hand-built tensors -------------------------------------------------------------

def test_compare_scaled_copy():
    """compare(2b, b): cos 1, ratio 2, rel 1. Catches ratio taken as ‖b‖/‖a‖ or rel not squared
    (rel = ‖a-b‖²/‖b‖² = 1 here; unsquared would also give 1, so the next test disambiguates)."""
    b = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(0))
    r = reg().compare(2 * b, b)
    assert r['cos'] == pytest.approx(1.0, abs=1e-5)
    assert r['ratio'] == pytest.approx(2.0, abs=1e-5)
    assert r['rel'] == pytest.approx(1.0, abs=1e-5)


def test_compare_rel_is_squared_and_ratio_direction():
    """compare(3b, b): ‖a-b‖²/‖b‖² = 4 (unsquared would give 2), ratio 3 (inverted would give 1/3)."""
    b = torch.randn(3, 7, generator=torch.Generator().manual_seed(1))
    r = reg().compare(3 * b, b)
    assert r['rel'] == pytest.approx(4.0, rel=1e-4)
    assert r['ratio'] == pytest.approx(3.0, rel=1e-4)


def test_compare_orthogonal_and_identical():
    """Orthogonal -> cos 0 (rel = 2 since ‖a-b‖² = ‖a‖²+‖b‖² with equal norms); identical -> rel 0, cos 1, ratio 1."""
    a = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    b = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    r = reg().compare(a, b)
    assert r['cos'] == pytest.approx(0.0, abs=1e-6)
    assert r['rel'] == pytest.approx(2.0, abs=1e-5)
    s = reg().compare(b, b)
    assert s['rel'] == pytest.approx(0.0, abs=1e-8)
    assert s['cos'] == pytest.approx(1.0, abs=1e-6)
    assert s['ratio'] == pytest.approx(1.0, abs=1e-6)


def test_compare_flattens_whole_tensor():
    """Metrics are over the flattened tensor, not per-row averages: a has one row equal to b, one row zero,
    so cos = 1/sqrt(2) and ratio = 1/sqrt(2); per-row averaging would give different numbers."""
    b = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    a = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    r = reg().compare(a, b)
    assert r['cos'] == pytest.approx(0.5 ** 0.5, abs=1e-5)
    assert r['ratio'] == pytest.approx(0.5 ** 0.5, abs=1e-5)
    assert r['rel'] == pytest.approx(0.5, abs=1e-5)


# ---- 2. quick run ---------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def quick():
    import time
    t0 = time.time()
    out = reg().run(gammas=(0.5,), draws=(64, 256, 1024), seeds=(0, 1), layers=(2,), nrep=8)
    return out, time.time() - t0


def find(out, site, gamma=0.5):
    hits = [r for r in out['results'] if r['site'] == site and r['gamma'] == gamma and r['n_layer'] == 2]
    assert len(hits) == 1, f'expected one result for {site}, got {len(hits)}'
    return hits[0]


SITES = ['transformer.h.0.attn.c_v', 'transformer.h.1.attn.c_v', 'value_embeds.1']


def test_quick_run_time_and_json(quick):
    """Quick run finishes < 60 s (brief) and json.dumps works (numpy/torch scalars leaking would raise)."""
    out, elapsed = quick
    assert elapsed < 60
    json.dumps(out)
    assert isinstance(out['passed'], bool)


def test_quick_run_sites_and_structure(quick):
    """Entries exist for both c_v blocks and block 1's value_embeds, none for block 0 VE (block 0 has no VE),
    with the metadata the brief promises: window 64 for block 0 (= sequence_len // 2), 128 for block 1."""
    out, _ = quick
    sites = {r['site'] for r in out['results'] if r['n_layer'] == 2}
    assert sites == set(SITES)
    assert find(out, SITES[0])['window'] == 64 and find(out, SITES[0])['has_ve'] is False
    assert find(out, SITES[1])['window'] == 128 and find(out, SITES[1])['has_ve'] is True
    assert find(out, SITES[2])['kind'] == 've' and find(out, SITES[0])['kind'] == 'v'
    for s in SITES:
        per_k = find(out, s)['per_K']
        assert [p['K'] for p in per_k] == [64, 256, 1024]
        for p in per_k:
            assert len(p['cos']) == len(p['ratio']) == len(p['rel']) == 2  # one per seed
            assert 'cos_mean' in p and 'ratio_mean' in p


@pytest.mark.parametrize('site', SITES)
def test_quick_run_slope_and_monotone(quick, site):
    """Squared error scales as 1/K: slope in (-1.4, -0.6). Catches an estimator whose noise does not
    average down (e.g. fixed draws reused across K) or a wrong draws divisor. Also rel(1024) < rel(64)."""
    r = find(quick[0], site)
    assert -1.4 < r['slope'] < -0.6
    rel = lambda p: sum(p['rel']) / len(p['rel'])
    assert rel(r['per_K'][-1]) < rel(r['per_K'][0])


@pytest.mark.parametrize('site', SITES)
def test_quick_run_converges_in_direction_and_norm(quick, site):
    """The seed-averaged estimate (2*1024 effective draws) must match B in direction and norm at gamma=0.5.
    Thresholds are looser than PLAN (K=4096): cos_mean > 0.95, |ratio_mean - 1| < 0.05 (correct code gives
    ~0.02). A transposed decay fails these at gamma=0.5; a scale error beyond ~5% fails the ratio. The 3%
    scale sensitivity is the stage gate's job (test_evaluate_gate_formula)."""
    p = find(quick[0], site)['per_K'][-1]
    assert p['cos_mean'] > 0.95
    assert abs(p['ratio_mean'] - 1) < 0.05


# ---- 3. nrep independence (built from setup + WP1 APIs) --------------------------------------------

@pytest.mark.parametrize('nrep', [1, 2, 8])
def test_nrep_does_not_change_the_mean(nrep):
    """Single-seed K=1024 sampled estimate at gamma=0.5 has cos > 0.9 vs analytic_value_errors for every
    nrep (nrep only chunks draws; 'v'/'ve' are uncentred). Catches chunk-dependent scaling or centring
    at nrep=1 / nrep=2, and a missing 1/K or sigma factor."""
    est, targets, info = reg().setup(n_layer=2, nrep=nrep, seed=0)
    try:
        est.settings.gamma = 0.5
        with torch.no_grad():
            b = est.analytic_value_errors(targets, 0.5)
            est.generator.manual_seed(123)
            a_v = est.local_errors('v', 1024, targets)
            est.generator.manual_seed(124)
            a_ve = est.local_errors('ve', 1024, targets)
        a = {**a_v, **a_ve}
        assert set(a) == set(SITES)
        for site in SITES:
            assert flat_cos(a[site], b[site]) > 0.9, site
    finally:
        est.close()


def test_setup_randomizes_ve_gate_and_has_expected_blocks():
    """ve_gate weights are non-zero (zero at init hides a missing gate multiplier); block 0 short window
    no VE, block 1 full window with VE."""
    est, targets, info = reg().setup(n_layer=2, nrep=8, seed=0)
    try:
        for blk in est.model.transformer.h:
            if blk.attn.ve_gate is not None:
                assert blk.attn.ve_gate.weight.abs().sum() > 0
        assert set(targets) == {0, 1}
        assert est.batch == 2 and est.length == 128
    finally:
        est.close()


# ---- 4. control discriminates ----------------------------------------------------------------------

def test_control_wrong_gamma_is_clearly_worse():
    """Direct from the WP1 APIs: at gamma=0.5, ê_4096 against B(0.5) has small rel, against B(1.0) rel is
    more than 2x larger, for every site. Proves a gamma mismatch (e.g. transposed or ignored decay) is
    detectable at this K. Expected: rel_vs_B ~ 1/4096 scale of noise; rel_vs_wrong is O(0.1+)."""
    est, targets, _ = reg().setup(n_layer=2, nrep=8, seed=0)
    try:
        est.settings.gamma = 0.5
        with torch.no_grad():
            b = est.analytic_value_errors(targets, 0.5)
            wrong = est.analytic_value_errors(targets, 1.0)
            est.generator.manual_seed(7)
            a = {**est.local_errors('v', 4096, targets), **est.local_errors('ve', 4096, targets)}
        for site in SITES:
            rel = lambda ref: float((a[site] - ref[site]).double().pow(2).sum() / ref[site].double().pow(2).sum())
            assert rel(wrong) > 2 * rel(b), (site, rel(wrong), rel(b))
    finally:
        est.close()


def test_run_controls_entries_discriminate():
    """If run() reports controls, each has rel_vs_wrong_gamma > 2*rel_vs_B and the four keys the brief names.
    run() must report controls for gamma=0.5 when K=4096 is among draws."""
    out = reg().run(gammas=(0.5,), draws=(64, 4096), seeds=(0,), layers=(2,), nrep=8)
    assert out['controls'], 'no controls reported for gamma=0.5'
    for c in out['controls']:
        for key in ('site', 'cos_vs_B', 'cos_vs_wrong_gamma', 'rel_vs_B', 'rel_vs_wrong_gamma'):
            assert key in c
        assert c['rel_vs_wrong_gamma'] > 2 * c['rel_vs_B']
        assert c['cos_vs_B'] > c['cos_vs_wrong_gamma']
    assert {c['site'] for c in out['controls']} == set(SITES)


# ---- 5. main ---------------------------------------------------------------------------------------

def test_main_quick_writes_json(tmp_path):
    """main(['--quick','--out',...]) writes valid JSON with bool 'passed' and dict 'checks'. main may
    return or raise SystemExit(1) when not passed; both are accepted, the file must exist either way."""
    path = tmp_path / 'r.json'
    try:
        reg().main(['--quick', '--out', str(path)])
    except SystemExit as e:
        assert e.code in (0, 1, None)
    data = json.loads(path.read_text())
    assert isinstance(data['passed'], bool)
    assert isinstance(data['checks'], dict) and data['checks']
    assert all(isinstance(v, bool) for v in data['checks'].values())
    assert data['passed'] == all(data['checks'].values())
    assert data['config']['nrep'] == 8


# ---- lead (2026-10-10): the amended 3.1 gate, on hand-built numbers -----------------------------------

def test_evaluate_gate_formula():
    """The amended pass rule on synthetic (cos, ratio, rel) triples consistent with rel = r^2 - 2 c r + 1.
    Pure noise at gamma=0.98 (projection p = c*r = 1) passes; a 5% short norm (p = 0.95) fails the
    norm check; a flat slope (bias floor) and a non-decreasing rel fail theirs. Catches a wrong formula
    in the gate (e.g. ratio - rel - 1), which would otherwise only show up as a stage failure."""
    def triple(c, r):
        return dict(cos=[c], ratio=[r], rel=[r * r - 2 * c * r + 1])
    r = 1.035
    good = triple(1 / r, r)          # p = 1, rel = r^2 - 1 ~ 0.071: noise only
    checks = reg().evaluate(good, -1.0, [1.0, 0.25, 0.07])
    assert all(checks.values()), checks
    short = triple(0.95 / 0.97, 0.97)  # p = 0.95: 5% scale bias
    checks = reg().evaluate(short, -1.0, [1.0, 0.25, 0.07])
    assert not checks['|ratio^2-rel-1|<0.05']
    checks = reg().evaluate(good, -0.4, [1.0, 0.25, 0.25])
    assert not checks['slope_in(-1.25,-0.75)'] and not checks['rel_decreasing']


def test_run_rejects_draws_not_multiple_of_nrep():
    """local_errors runs draws // nrep chunks but divides by draws, so K % nrep != 0 silently rescales the
    estimate (K=64, nrep=24 -> 0.75x). run() must refuse."""
    with pytest.raises(ValueError):
        reg().run(gammas=(0.5,), draws=(64,), seeds=(0,), layers=(2,), nrep=24)
