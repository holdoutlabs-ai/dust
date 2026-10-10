"""A tiny model and matching DUST recipe for fast CPU tests.

vocab_size stays 4096 so real token data and the 256-column head slabs work unchanged.
Every draw count is a multiple of nrep, and head is a multiple of nrep * 16 slabs.
"""
from dust import Recipe
from model import GPTConfig


def tiny_config(**overrides):
    return GPTConfig(**{
        **dict(sequence_len=128, vocab_size=4096, n_layer=2, n_head=4, n_kv_head=4, n_embd=64),
        **overrides,
    })


def tiny_recipe(draws_scale=1, nrep=2, gamma=0.98, lr=0.25, momentum=0.95, **overrides):
    s = draws_scale
    settings = dict(
        writers=[32 * s, 16 * s], hidden=[32 * s, 16 * s], hubs=[16 * s, 8 * s],
        embedding=16 * s, head=512 * s,
        local=dict(q=4 * s, v=32 * s, gate=8 * s, k=16 * s, ve=16 * s),
        nrep=nrep, lr=lr, momentum=momentum, gamma=gamma,
    )
    settings.update(overrides)
    return Recipe(**settings)
