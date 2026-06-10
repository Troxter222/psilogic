"""Chaos detector and decay schedule helpers for PsiLogic."""

from __future__ import annotations

import math

import torch


def resolve_warmup(warmup_cfg: int, gamma_t_max: int) -> int:
    """Return effective chaos warmup steps (auto-scales when ``warmup_cfg == -1``)."""
    auto_warmup = max(50, gamma_t_max // 10) if gamma_t_max > 0 else 200
    return warmup_cfg if warmup_cfg >= 0 else auto_warmup


def effective_gamma_and_qd(
    step: int,
    gamma_t_max: int,
    gamma: float,
    quantum_decay: float,
) -> tuple[float, float]:
    """Cosine schedule for gamma and quantum decay; constant when ``gamma_t_max == 0``."""
    if gamma_t_max > 0:
        cos_w = 0.5 * (1.0 + math.cos(math.pi * min(step / gamma_t_max, 1.0)))
        return gamma * cos_w, quantum_decay * cos_w
    return gamma, quantum_decay


def chaos_contribution(
    slow_t: torch.Tensor,
    fast_t: torch.Tensor,
    *,
    adaptive_tau: bool,
    chaos_tau: float,
    tau_scale: float,
    eps: float,
    lr: float,
    gamma_eff: float,
    p_ext: float,
    max_cancel: float,
    param_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spike mask and clamped active-cancellation coefficient.

    Returns ``(chaos_contrib, spike_mask)`` where ``spike_mask`` is 0/1 in
    ``param_dtype`` and ``chaos_contrib`` is the per-step shrinkage fraction.
    """
    if adaptive_tau:
        spike_mask = (fast_t > tau_scale * slow_t + eps).to(param_dtype)
    else:
        spike_mask = (slow_t >= chaos_tau).to(param_dtype)

    ratio = fast_t / (slow_t + eps)
    chaos = torch.tanh(slow_t) * (1.0 + 0.5 * torch.tanh(torch.clamp(ratio - 1.0, min=0.0)))
    raw_cc = chaos * lr * gamma_eff * p_ext
    return torch.clamp(raw_cc, max=max_cancel) * spike_mask, spike_mask


def update_gradient_norm_ema(
    grad_norm: torch.Tensor,
    numel: int,
    step: int,
    fast: torch.Tensor,
    slow: torch.Tensor,
    gn_avg: torch.Tensor,
    eps: float,
) -> None:
    """In-place dual EMA update of scale-normalized gradient norm."""
    gn_scaled = grad_norm / math.sqrt(max(numel, 1))
    if step == 1:
        gn_avg.fill_(gn_scaled)
        fast.fill_(1.0)
        slow.fill_(1.0)
    else:
        gn_avg.mul_(0.99).add_(gn_scaled, alpha=0.01)
        gn_norm = gn_scaled / (gn_avg + eps)
        fast.mul_(0.9).add_(gn_norm, alpha=0.1)
        slow.mul_(0.99).add_(gn_norm, alpha=0.01)
