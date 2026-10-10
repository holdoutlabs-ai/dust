import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.tiny import tiny_config  # noqa: E402
from dust import make_model  # noqa: E402


@pytest.fixture
def tiny_model():
    """Tiny GPT on CPU in float32. ve_gate is randomized so gate != 1 (it is zero at init)."""
    torch.manual_seed(0)
    model = make_model(tiny_config(), torch.device('cpu'), seed=42)
    with torch.no_grad():
        for block in model.transformer.h:
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight.normal_(0, 0.5)
    return model


@pytest.fixture
def tiny_batch():
    """Random tokens [2, 128] for x and y; no data files needed."""
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 4096, (2, 129), generator=g)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
