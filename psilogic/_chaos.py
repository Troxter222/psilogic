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

# Soft-gate policy: the fast/slow ratio becomes a continuous ``excess`` in
# ``(0, 1)`` instead of a hard ``fast > tau_scale * slow`` mask. ``tau_scale``
# keeps its meaning — "how far above 1.0 the ratio must climb before the gate
# is wide open" — because the sigmoid saturates (~0.98) at ``ratio ==
# tau_scale``. The hard mask left the gate shut for entire runs, which made
# gamma / max_cancel / p_ext bit-for-bit inert.
_EXCESS_SHARPNESS = 4.0
_MIN_TAU_MARGIN = 0.25

# Never let trust reach zero: ``max_cancel == 1.0`` is a legal setting and a
# zero trust would divide the Adam denominator by zero.
_MAX_DAMPING = 0.99


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

    Returns ``step``, ``fast``, ``slow``, ``ratio`` (fast/slow), ``gn_avg``
    and ``soft_chaos`` (the continuous level that actually drove the last
    update). Safe to call on an empty (uninitialized) state.
    """
    if not state or "fast" not in state:
        return {
            "step": 0.0,
            "fast": 0.0,
            "slow": 0.0,
            "ratio": 0.0,
            "gn_avg": 0.0,
            "soft_chaos": 0.0,
        }
    fast = float(state["fast"])
    slow = float(state["slow"])
    return {
        "step": float(state.get("t", 0)),
        "fast": fast,
        "slow": slow,
        "ratio": fast / (slow + 1e-12),
        "gn_avg": float(state["gn_avg"]) if "gn_avg" in state else 0.0,
        "soft_chaos": float(state["soft_chaos"]) if "soft_chaos" in state else 0.0,
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


def grad_momentum_disagreement(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    grad_norm: torch.Tensor,
    *,
    step: int,
    eps: float,
) -> torch.Tensor:
    """Return ``0.5 * (1 - cos(grad, momentum))`` as a 1-element fp32 tensor.

    ``0`` means the fresh gradient points the same way as accumulated
    momentum (the optimizer is making consistent progress) and ``1`` means it
    fully opposes it (the step is oscillating across a valley). ``momentum``
    must still hold the *pre-update* value, and ``grad_norm`` the norm of the
    same post-AGC/post-centralize gradient that feeds the chaos EMAs.

    Returns ``0`` on the first step, where momentum is still all zeros and
    the cosine carries no information.
    """
    if step <= 1:
        return torch.zeros(1, device=momentum.device, dtype=torch.float32)
    dot = (momentum * grad).sum()
    denom = momentum.norm() * grad_norm.to(torch.float32) + eps
    cos = torch.clamp(dot / denom, min=-1.0, max=1.0)
    return (0.5 * (1.0 - cos)).reshape(1).to(torch.float32)


def soft_chaos_signal(
    slow_t: torch.Tensor,
    fast_t: torch.Tensor,
    disagree: torch.Tensor,
    *,
    adaptive_tau: bool,
    chaos_tau: float,
    tau_scale: float,
    eps: float,
) -> torch.Tensor:
    """Continuous chaos level in ``[0, 1]``.

    Two independent symptoms of "the model is confused" are averaged:

    * ``excess`` — the fast EMA of the gradient norm running hot relative to
      the slow baseline, squashed through a sigmoid whose sharpness is set by
      ``tau_scale`` (or by ``chaos_tau`` when ``adaptive_tau=False``).
    * ``disagree`` — the gradient fighting its own momentum, from
      :func:`grad_momentum_disagreement`.

    ``tanh(slow)`` scales the whole thing, so the signal vanishes on its own
    once gradients go quiet at convergence.

    Every operation is elementwise, so this serves the per-parameter path
    (1-element tensors) and the batched path (stacked ``(n,)`` vectors)
    identically.
    """
    if adaptive_tau:
        sharpness = _EXCESS_SHARPNESS / max(tau_scale - 1.0, _MIN_TAU_MARGIN)
        excess = torch.sigmoid((fast_t / (slow_t + eps) - 1.0) * sharpness)
    else:
        excess = torch.sigmoid((slow_t / max(chaos_tau, eps) - 1.0) * _EXCESS_SHARPNESS)
    return torch.tanh(slow_t) * (0.5 * excess + 0.5 * disagree)


def trust_from_soft_chaos(
    soft_chaos: torch.Tensor,
    *,
    gamma_eff: torch.Tensor | float,
    p_ext: float,
    chaos_gain: float,
    max_cancel: float,
) -> torch.Tensor:
    """Return the ``trust`` factor the Adam/Lion update is scaled by.

    ``trust = 1 - min(soft_chaos * gamma * p_ext * warmup_gain, max_cancel)``,
    so ``gamma`` reads as "the largest fraction of a step we are willing to
    withhold when the model is maximally confused". ``gamma_eff`` may be a
    Python float (group-uniform) or a per-parameter ``(n,)`` tensor from
    :func:`auto_gamma_batched`.
    """
    cap = min(max_cancel, _MAX_DAMPING)
    damping = torch.clamp(soft_chaos * gamma_eff * p_ext * chaos_gain, max=cap)
    return 1.0 - damping


def update_gradient_norm_ema_batched(
    gn_scaled: torch.Tensor,
    step: int,
    fast_vec: torch.Tensor,
    slow_vec: torch.Tensor,
    gn_avg_vec: torch.Tensor,
    eps: float,
) -> None:
    """Vectorized dual-EMA update across an entire parameter group.

    Only valid when every parameter in the group is on the *same* step
    (checked by the caller). ``gn_scaled`` is the stacked, already
    ``||grad||/sqrt(numel)``-normalized per-parameter gradient norm.

    Every operation here is elementwise along the parameter axis — there is
    no reduction across parameters — so this is bit-for-bit identical to
    calling :func:`update_gradient_norm_ema` once per parameter in a loop;
    it is purely a batching of the same arithmetic into fewer, larger kernel
    launches instead of many size-1 ones.
    """
    if step == 1:
        gn_avg_vec.copy_(gn_scaled)
        fast_vec.fill_(1.0)
        slow_vec.fill_(1.0)
    else:
        gn_avg_vec.mul_(0.99).add_(gn_scaled, alpha=0.01)
        gn_norm = gn_scaled / (gn_avg_vec + eps)
        fast_vec.mul_(0.9).add_(gn_norm, alpha=0.1)
        slow_vec.mul_(0.99).add_(gn_norm, alpha=0.01)


def auto_gamma_batched(
    slow_vec: torch.Tensor,
    step: int,
    gamma_base: float,
) -> torch.Tensor:
    """Vectorized :func:`auto_gamma` — never calls ``.item()``/``float()``.

    The scalar version forces a device-to-host sync per parameter (its own
    docstring flags this). Doing the same convergence-aware reduction as one
    elementwise op over the stacked ``slow`` vector produces identical
    per-parameter values while keeping everything on-device, so a group of
    ``n`` parameters costs one sync-free kernel instead of ``n`` blocking
    syncs.
    """
    if step <= 1 or gamma_base <= 0.0:
        return torch.full_like(slow_vec, gamma_base)
    scaled = gamma_base * torch.clamp(slow_vec / _AUTO_GAMMA_THRESHOLD, min=_AUTO_GAMMA_FLOOR)
    return torch.where(
        slow_vec >= _AUTO_GAMMA_THRESHOLD, torch.full_like(slow_vec, gamma_base), scaled
    )


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
