"""Muon optimizer (Keller Jordan, 2024) — the optimizer behind the
modded-nanoGPT speedrun record. Pure PyTorch, no external deps.

Idea: for 2-D weight matrices, take the momentum update and *orthogonalise*
it with a few Newton-Schulz iterations before applying it. This spreads the
update energy evenly across singular directions and empirically converges
faster than Adam per step — exactly what a fixed 2000-step budget wants.

Non-2-D params (embeddings, the tied head, norms, biases) are NOT suited to
Muon and are handled by a separate AdamW in train.py.
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration -> approx orthogonalisation of G (2-D)."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom, lr = group["momentum"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # scale so the effective step size is shape-invariant
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-lr * scale)
