"""Setup check: upstream DUST runs one step on CPU with the tiny config."""
import math

import torch

from diagnostics.tiny import tiny_recipe
from dust import DUST, make_optimizer


def test_upstream_step_runs_on_cpu(tiny_model, tiny_batch):
    settings = tiny_recipe()
    estimator = DUST(tiny_model, settings, seed=0)
    try:
        before = tiny_model.transformer.h[0].attn.c_v.weight.clone()
        loss = estimator.step(*tiny_batch, make_optimizer(tiny_model, settings))
    finally:
        estimator.close()
    assert math.isfinite(loss)
    assert not torch.equal(before, tiny_model.transformer.h[0].attn.c_v.weight)


def test_tiny_config_has_short_window_and_ve_blocks(tiny_model):
    """Stage 3.1 needs a short-window block and a block without VE; n_layer=2 gives both."""
    assert tiny_model.window_sizes[0][0] == 64 and tiny_model.window_sizes[1][0] == 128
    assert set(tiny_model.value_embeds.keys()) == {'1'}
