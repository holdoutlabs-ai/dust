"""Optimizers for the backprop baseline: upstream SGD, AdamW, and Muon (Newton-Schulz orthogonalized momentum)."""
import torch
from torch import Tensor


NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def zeropower_newton_schulz(G: Tensor, steps: int = 5) -> Tensor:
    """Approximate U V^T for G = U S V^T with the quintic Newton-Schulz iteration, in float32.

    The quintic does not converge exactly: singular values land roughly in [0.68, 1.13].
    Returns a float32 tensor of G's shape.
    """
    if G.ndim != 2:
        raise ValueError(f'Newton-Schulz needs a 2-D tensor, got shape {tuple(G.shape)}')
    a, b, c = NS_COEFFICIENTS
    X = G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """SGD momentum whose update is orthogonalized by Newton-Schulz. 2-D parameters only."""

    def __init__(self, params, lr, momentum=0.95, nesterov=True, ns_steps=5):
        if lr <= 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0 <= momentum < 1:
            raise ValueError(f'Invalid momentum: {momentum}')
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                if p.ndim != 2:
                    raise ValueError(f'Muon only takes 2-D parameters, got shape {tuple(p.shape)}')

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                u = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                O = zeropower_newton_schulz(u, group['ns_steps'])
                rows, cols = p.shape
                p.add_(O.to(p.dtype), alpha=-lr * max(1, rows / cols) ** 0.5)
        return loss


def partition(model):
    """(muon_params, adam_params): 2-D weights in transformer.h.* except attn.ve_gate go to Muon; the rest to AdamW."""
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if name.startswith('transformer.h.') and p.ndim == 2 and '.attn.ve_gate.' not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def build_optimizer(model, name, lr, momentum, adam_lr=None):
    """Return a list of optimizers. 'sgd' is exactly upstream make_optimizer."""
    if name == 'sgd':
        groups = [
            {'params': [p for n, p in model.named_parameters()
                        if n != 'transformer.wte.weight'], 'lr': lr},
            {'params': [model.transformer.wte.weight], 'lr': 1000.0},
        ]
        return [torch.optim.SGD(groups, lr=lr, momentum=momentum)]
    if name == 'adamw':
        return [torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)]
    if name == 'muon':
        muon_params, adam_params = partition(model)
        return [
            Muon(muon_params, lr=lr, momentum=0.95 if momentum is None else momentum),
            torch.optim.AdamW(adam_params, lr=0.008 if adam_lr is None else adam_lr,
                              betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
        ]
    raise ValueError(f'Unknown optimizer: {name}')
