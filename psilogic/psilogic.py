import math
import torch
from torch.optim.optimizer import Optimizer

class PsiLogic(Optimizer):
    r"""
    ΨLogic v4 — Fast GPU-optimized version.
    
    Fully vectorized operations. Zero CPU-GPU syncs (no .item() calls).
    Gradient norms are normalized by parameter size for threshold stability.

    Args:
        params          : iterable of parameters or dicts defining parameter groups
        lr              : learning rate (default: 1e-3)
        betas           : (beta1, beta2) EMA coefficients (default: (0.9, 0.999))
        weight_decay    : AdamW L2 coefficient (default: 1e-4)
        gamma           : maximum active cancellation strength (default: 0.05)
        p_ext           : P degree (default: 1.2)
        quantum_decay   : Quantum Decay coefficient (default: 1e-3)
        eps             : numerical stability term (default: 1e-8)
        grad_centralize : subtract spatial mean from gradients (default: True)
        chaos_tau       : slow_ema threshold (default: 0.3)
        gamma_T_max     : total optimizer steps for cosine gamma decay (0 = disabled)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 1e-4,
        gamma: float = 0.05,
        p_ext: float = 1.2,
        quantum_decay: float = 1e-3,
        eps: float = 1e-8,
        grad_centralize: bool = True,
        chaos_tau: float = 0.3,
        gamma_T_max: int = 0,
    ):
        if lr < 0.0:             raise ValueError(f"Invalid lr: {lr}")
        if weight_decay < 0.0:   raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if gamma < 0.0:          raise ValueError(f"Invalid gamma: {gamma}")
        if quantum_decay < 0.0:  raise ValueError(f"Invalid quantum_decay: {quantum_decay}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta2: {betas[1]}")
        if chaos_tau < 0.0:      raise ValueError(f"Invalid chaos_tau: {chaos_tau}")

        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay,
            gamma=gamma, p_ext=p_ext, quantum_decay=quantum_decay,
            eps=eps, grad_centralize=grad_centralize,
            chaos_tau=chaos_tau, gamma_T_max=gamma_T_max,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            wd           = group["weight_decay"]
            gamma        = group["gamma"]
            p_ext        = group["p_ext"]
            qd           = group["quantum_decay"]
            eps          = group["eps"]
            gc           = group["grad_centralize"]
            tau          = group["chaos_tau"]
            T_max        = group["gamma_T_max"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                dev = p.device

                # Gradient Centralization
                if gc and g.dim() > 1:
                    g = g - g.mean(dim=tuple(range(1, g.dim())), keepdim=True)

                st = self.state[p]
                if not st:
                    st["t"]    = 0
                    st["m"]    = torch.zeros_like(p)
                    st["v"]    = torch.zeros_like(p)
                    st["fast"] = torch.zeros(1, device=dev, dtype=p.dtype)
                    st["slow"] = torch.zeros(1, device=dev, dtype=p.dtype)

                st["t"] += 1
                t = st["t"]

                # AdamW decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Moment estimates
                st["m"].mul_(beta1).add_(g, alpha=1.0 - beta1)
                st["v"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Dual EMA (normalized by parameter count for scale invariance)
                gn = g.norm() / math.sqrt(g.numel())
                
                if t == 1:
                    st["fast"].copy_(gn)
                    st["slow"].copy_(gn)
                else:
                    st["fast"].mul_(0.9).add_(gn, alpha=0.1)
                    st["slow"].mul_(0.99).add_(gn, alpha=0.01)

                slow, fast = st["slow"], st["fast"]

                # Cosine gamma schedule
                if T_max > 0:
                    cos_w = 0.5 * (1.0 + math.cos(math.pi * min(t / T_max, 1.0)))
                    g_base = gamma * cos_w
                else:
                    g_base = gamma

                # Chaos term & Active Cancellation
                ratio   = fast / (slow + eps)
                chaos   = torch.tanh(slow) * (1.0 + 0.5 * torch.tanh(torch.relu(ratio - 1.0)))
                
                c_coeff = torch.where(
                    slow < tau,
                    torch.zeros(1, device=dev, dtype=p.dtype),
                    chaos * (lr * g_base * p_ext)
                )
                p.mul_(1.0 - c_coeff)

                # Quantum Decay (fully vectorized)
                if qd != 0.0:
                    qd_w = torch.where(
                        slow >= tau,
                        torch.full((1,), lr * qd, device=dev, dtype=p.dtype),
                        torch.zeros(1, device=dev, dtype=p.dtype)
                    )
                    p.mul_(1.0 - qd_w * torch.tanh(g.abs()))

                # Bias-corrected Adam gradient step
                step_size  = lr / (1.0 - beta1 ** t)
                bias_corr2 = math.sqrt(1.0 - beta2 ** t)
                denom = st["v"].sqrt().div_(bias_corr2).add_(eps)
                
                p.addcdiv_(st["m"], denom, value=-step_size)

        return loss
