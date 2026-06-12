"""Chaos detector and decay schedule helpers for PsiLogic."""

from __future__ import annotations

import math
from typing import Any

import torch

# Auto-warmup policy: when ``chaos_warmup == -1`` the warmup horizon scales
# with the training length so chaos never fires into raw from-scratch noise.
_AUTO_WARMUP_FLOOR = 500
_AUTO_WARMUP_DIVISOR = 20
_AUTO_WARMUP_FALLBACK = 200  # used when the training horizon is unknown

# Auto-gamma policy: when the slow EMA falls below this threshold the run is
# considered converged and gamma is reduced proportionally (with a floor).
_AUTO_GAMMA_THRESHOLD = 0.1
_AUTO_GAMMA_FLOOR = 0.1


def resolve_warmup(warmup_cfg: int, total_steps: int) -> int:
    """Return the effective number of chaos warmup steps.

    ``warmup_cfg >= 0`` is honoured verbatim. ``warmup_cfg == -1`` auto-scales
    with the training horizon: ``max(500, total_steps // 20)`` when
    ``total_steps`` is known, otherwise a conservative 200 steps.
    """
    if warmup_cfg >= 0:
        return warmup_cfg
    if total_steps > 0:
        return max(_AUTO_WARMUP_FLOOR, total_steps // _AUTO_WARMUP_DIVISOR)
    return _AUTO_WARMUP_FALLBACK


def effective_warmup(step: int, total_steps: int, base_warmup: int) -> float:
    """Return the chaos gain in ``[0, 1]`` for the current optimizer step.

    The gain is exactly ``0.0`` while ``step <= warmup`` and then ramps
    linearly to ``1.0`` over one quarter of the warmup window, so chaos
    "warms in" instead of switching on abruptly. With ``base_warmup == 0``
    the gain is ``1.0`` from the very first step.
    """
    warmup = resolve_warmup(base_warmup, total_steps)
    if step <= warmup:
        return 0.0
    ramp = max(1, warmup // 4)
    return min(1.0, (step - warmup) / ramp)


def auto_gamma(
    slow_t: torch.Tensor | float,
    step: int,
    gamma_base: float,
) -> float:
    """Convergence-aware gamma reduction.

    When the scale-normalized slow EMA drops below ``0.1`` the network is in
    a quiet, converged regime where active cancellation only hurts — gamma is
    reduced proportionally, floored at 10% of its base value.

    Note: calling ``float()`` on a CUDA tensor synchronizes the stream, so
    this helper is only invoked when ``gamma_auto=True`` is explicitly set.
    """
    if step <= 1 or gamma_base <= 0.0:
        return gamma_base
    slow = float(slow_t) if isinstance(slow_t, torch.Tensor) else float(slow_t)
    if slow >= _AUTO_GAMMA_THRESHOLD:
        return gamma_base
    return gamma_base * max(slow / _AUTO_GAMMA_THRESHOLD, _AUTO_GAMMA_FLOOR)


def get_chaos_metrics(state: dict[str, Any]) -> dict[str, float]:
    """Export human-readable chaos metrics from a per-parameter state dict.

    Returns ``step``, ``fast``, ``slow``, ``ratio`` (fast/slow) and
    ``gn_avg``. Safe to call on an empty (uninitialized) state.
    """
    if not state or "fast" not in state:
        return {"step": 0.0, "fast": 0.0, "slow": 0.0, "ratio": 0.0, "gn_avg": 0.0}
    fast = float(state["fast"])
    slow = float(state["slow"])
    return {
        "step": float(state.get("t", 0)),
        "fast": fast,
        "slow": slow,
        "ratio": fast / (slow + 1e-12),
        "gn_avg": float(state["gn_avg"]) if "gn_avg" in state else 0.0,
    }


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
