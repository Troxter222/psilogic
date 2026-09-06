"""PsiLogic-specific diagnostics.

PsiLogic augments Adam with a chaos-conditioned damping term driven by a dual
EMA of the (scale-normalized) gradient norm:

    fast_t = 0.90 * fast + 0.10 * gn   (tau ~ 10 steps)
    slow_t = 0.99 * slow + 0.01 * gn   (tau ~ 100 steps)
    ratio  = fast_t / slow_t
    chaos_t = tanh(slow_t) * (1 + 0.5 * tanh(relu(ratio - 1)))

This module extracts those internal signals so the benchmark can report the
dynamics requested by the paper: ``chaos_t`` over time and the ``fast_t -
slow_t`` gap. It degrades gracefully for non-PsiLogic optimizers (returns an
empty dict), so the training loop can call it unconditionally.
"""

from __future__ import annotations

import math
from typing import Dict

from torch.optim.optimizer import Optimizer

from .optimizers import is_psilogic


def _chaos_from(slow: float, ratio: float) -> float:
    """Reproduce PsiLogic's scalar ``chaos_t`` from aggregate slow/ratio."""
    return math.tanh(slow) * (1.0 + 0.5 * math.tanh(max(ratio - 1.0, 0.0)))


def psilogic_chaos_metrics(optimizer: Optimizer) -> dict[str, float]:
    """Return aggregate PsiLogic chaos signals, or ``{}`` for other optimizers.

    Keys (all parameter-averaged across groups):
        ``psi/fast_t``    -- responsive gradient-norm EMA.
        ``psi/slow_t``    -- stable baseline EMA.
        ``psi/ratio_t``   -- fast/slow ratio (chaos detector).
        ``psi/chaos_t``   -- derived damping coefficient in [0, ~1.5].
        ``psi/fast_minus_slow`` -- the fast/slow gap requested by the paper.
        ``psi/spike_rate``-- fraction of parameter tensors currently gated.
    """
    if not is_psilogic(optimizer):
        return {}

    # Preferred path: the library's own diagnostics helper (averages per group).
    try:
        from psilogic import debug as psi_debug  # type: ignore

        stats = psi_debug.chaos_stats(optimizer)
        if not stats:
            return {}
        # Aggregate across param groups, weighting by parameter count.
        total = sum(max(s.get("n_params", 0), 0) for s in stats) or len(stats)
        fast = sum(s["fast_mean"] * max(s.get("n_params", 1), 1) for s in stats) / total
        slow = sum(s["slow_mean"] * max(s.get("n_params", 1), 1) for s in stats) / total
        ratio = sum(s["ratio_mean"] * max(s.get("n_params", 1), 1) for s in stats) / total
        spike = sum(s["spike_rate"] * max(s.get("n_params", 1), 1) for s in stats) / total
        soft = (
            sum(s.get("soft_chaos_mean", 0.0) * max(s.get("n_params", 1), 1) for s in stats) / total
        )
        return {
            "psi/fast_t": float(fast),
            "psi/slow_t": float(slow),
            "psi/ratio_t": float(ratio),
            "psi/chaos_t": float(soft if soft > 0 else _chaos_from(slow, ratio)),
            "psi/fast_minus_slow": float(fast - slow),
            "psi/spike_rate": float(spike),
            "psi/soft_chaos": float(soft),
        }
    except Exception:
        pass

    # Fallback: read per-parameter state directly (works across psilogic versions).
    try:
        from psilogic import get_chaos_metrics  # type: ignore
    except Exception:
        return {}

    fasts, slows, ratios = [], [], []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if not state or "fast" not in state:
                continue
            m = get_chaos_metrics(state)
            fasts.append(m["fast"])
            slows.append(m["slow"])
            ratios.append(m["ratio"])
    if not fasts:
        return {}
    fast = sum(fasts) / len(fasts)
    slow = sum(slows) / len(slows)
    ratio = sum(ratios) / len(ratios)
    return {
        "psi/fast_t": float(fast),
        "psi/slow_t": float(slow),
        "psi/ratio_t": float(ratio),
        "psi/chaos_t": float(_chaos_from(slow, ratio)),
        "psi/fast_minus_slow": float(fast - slow),
    }
