# muon.py
import torch

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "mom" not in state:
                    state["mom"] = torch.zeros_like(g)
                buf = state["mom"]
                buf.mul_(group["momentum"]).add_(g)
                upd = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd, steps=group["ns_steps"])
                # aspect-ratio scaling so tall matrices don't undertrain
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(upd, alpha=-group["lr"] * scale)