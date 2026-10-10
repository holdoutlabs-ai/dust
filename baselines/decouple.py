"""Delayed gradients for the backprop baseline (Experiment 2, stage 2.1: the staleness ladder).

Step t computes g_t = grad L(batch t) at the evaluation point e_t = w_t + k * Delta_t (k = the parameter's delay with
--weight-prediction, else 0; Delta_t = w_t - w_{t-1}, Delta_1 = 0) and applies g_{t-d}, the gradient computed d steps
earlier. While t <= d nothing is applied for that parameter (.grad is None, so the optimizer skips it).
"""
from collections import deque
from contextlib import contextmanager

import torch


def delay_map(model, d: int, mode: str) -> dict[str, int]:
    """Per-parameter delay. 'uniform': d everywhere. 'depth': block i of L gets round(d * (L-1-i) / (L-1)),
    wte gets d, value_embeds.{i} follows block i, lm_head and the residual lambdas get 0."""
    if d < 0:
        raise ValueError('delay must be >= 0')
    names = [name for name, _ in model.named_parameters()]
    if mode == 'uniform':
        return {name: d for name in names}
    if mode != 'depth':
        raise ValueError(f'unknown delay mode: {mode}')
    n_layer = len(model.transformer.h)

    def block_delay(i):
        return d if n_layer == 1 else round(d * (n_layer - 1 - i) / (n_layer - 1))

    delays = {}
    for name in names:
        parts = name.split('.')
        if name.startswith('transformer.h.'):
            delays[name] = block_delay(int(parts[2]))
        elif name.startswith('value_embeds.'):
            delays[name] = block_delay(int(parts[1]))
        elif name.startswith('transformer.wte.'):
            delays[name] = d
        elif name.startswith('lm_head.') or name in ('resid_lambdas', 'x0_lambdas'):
            delays[name] = 0
        else:
            raise KeyError(f'no depth-mode delay rule for parameter {name}')
    return delays


class DelayedGradients:
    """Per-parameter gradient FIFOs plus optional weight prediction. Use around one train step:
    evaluation_point() for forward/backward, then exchange(), optimizer.step(), after_step()."""

    def __init__(self, model, delays: dict[str, int], weight_prediction: bool):
        self.params = dict(model.named_parameters())
        missing = set(self.params) - set(delays)
        if missing:
            raise KeyError(f'delays missing for parameters: {sorted(missing)}')
        self.delays = {name: int(delays[name]) for name in self.params}
        if any(k < 0 for k in self.delays.values()):
            raise ValueError('delays must be >= 0')
        self.weight_prediction = weight_prediction
        self.fifos = {name: deque() for name in self.params}
        # Only parameters that are actually extrapolated need Delta bookkeeping.
        self.predicted = [name for name in self.params if weight_prediction and self.delays[name] > 0]
        self.deltas = {name: None for name in self.predicted}  # None means Delta = 0 (step 1)
        self.before = {}  # w_t snapshots for the parameters in self.predicted

    @contextmanager
    def evaluation_point(self):
        """Move each predicted parameter to w_t + k * Delta_t in place; restore w_t exactly on exit."""
        with torch.no_grad():
            for name in self.predicted:
                p = self.params[name]
                self.before[name] = p.detach().clone()
                if self.deltas[name] is not None:
                    p.add_(self.deltas[name], alpha=self.delays[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name in self.predicted:
                    self.params[name].copy_(self.before[name])

    def exchange(self) -> None:
        """Stash each fresh gradient in its FIFO and set .grad to the gradient due now (g_{t-d}) or None."""
        for name, p in self.params.items():
            fifo = self.fifos[name]
            fifo.append(None if p.grad is None else p.grad.detach().clone())
            p.grad = fifo.popleft() if len(fifo) > self.delays[name] else None

    def after_step(self) -> None:
        """Delta_{t+1} = w_{t+1} - w_t for the predicted parameters."""
        with torch.no_grad():
            for name in self.predicted:
                self.deltas[name] = self.params[name].detach() - self.before[name]
        self.before = {}
