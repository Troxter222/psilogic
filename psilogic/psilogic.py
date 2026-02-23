"""
ΨLogic Optimizer
A quantum-inspired optimizer utilizing active gradient cancellation.
Formula: dΨ/dt = -iĤ·Ψ − γ·P·tanh(S_info)·Ψ
"""

import math
import torch
from torch.optim.optimizer import Optimizer

class PsiLogic(Optimizer):
    """
    ΨLogic Optimizer.
    
    Implements a quantum-inspired gradient descent with an active cancellation term.
    It stabilizes training by dynamically dampening gradient updates using a tanh
    non-linearity based on the exponential moving average (EMA) of the gradient norm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        momentum (float, optional): momentum factor (default: 0.9)
        gamma (float, optional): cancellation coefficient (default: 0.05)
        p_ext (float, optional): external power scale (default: 1.2)
        ema_alpha (float, optional): smoothing factor for gradient norm EMA (default: 0.05)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, gamma=0.05, p_ext=1.2, ema_alpha=0.05, eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = dict(lr=lr, momentum=momentum, gamma=gamma, 
                        p_ext=p_ext, ema_alpha=ema_alpha, eps=eps)
        super(PsiLogic, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            gamma = group['gamma']
            p_ext = group['p_ext']
            ema_alpha = group['ema_alpha']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buf'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['grad_norm_ema'] = 0.0 # EMA of gradient norm

                state['step'] += 1
                buf = state['momentum_buf']
                v = state['exp_avg_sq']

                # 1. Standard momentum step (The -iĤ·Ψ part)
                buf.mul_(beta).add_(grad, alpha=1 - beta)

                # 2. Adaptive variance tracking (Adam-style denominator)
                v.mul_(0.999).addcmul_(grad, grad, value=0.001)
                bias_corr = 1 - 0.999 ** state['step']
                v_hat = v / bias_corr
                
                base_update = buf / (v_hat.sqrt() + eps)

                # 3. Active Cancellation Term (The Quantum-inspired part)
                current_grad_norm = grad.norm().item()
                
                # Update EMA of the gradient norm (S_info)
                if state['step'] == 1:
                    state['grad_norm_ema'] = current_grad_norm
                else:
                    state['grad_norm_ema'] = (1 - ema_alpha) * state['grad_norm_ema'] + ema_alpha * current_grad_norm
                
                s_info = state['grad_norm_ema']
                
                # tanh(S_info): dynamic dampening
                active_cancel = math.tanh(s_info)
                cancel_update = gamma * p_ext * active_cancel * p

                # Apply updates
                p.add_(base_update, alpha=-lr)
                p.add_(cancel_update, alpha=-lr)

        return loss