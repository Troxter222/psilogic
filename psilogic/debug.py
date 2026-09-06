"""Diagnostics for inspecting PsiLogic chaos state and weight dynamics."""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch
import torch.nn as nn

from ._chaos import get_chaos_metrics
from .optimizer import PsiLogic

__all__ = ["chaos_stats", "layer_norms", "norm_history", "NormHistory", "get_chaos_metrics"]


def chaos_stats(optimizer: PsiLogic) -> list[dict[str, Any]]:
    """Summarize fast/slow chaos signals and spike rate per parameter group.

    Returns one dict per param group::

        {
            "group": 0,
            "n_params": 12,
            "step": 240,
            "fast_mean": 1.03,
            "slow_mean": 0.98,
            "ratio_mean": 1.05,
            "soft_chaos_mean": 0.12,  # continuous trust-damping signal
            "spike_rate": 0.08,       # fraction with soft_chaos > 0.5
        }

    Uninitialized parameters (no step taken yet) are skipped.
    """
    if not isinstance(optimizer, PsiLogic):
        raise TypeError(f"chaos_stats expects a PsiLogic optimizer, got {type(optimizer).__name__}")

    summary: list[dict[str, Any]] = []
    for group_idx, group in enumerate(optimizer.param_groups):
        fasts: list[float] = []
        slows: list[float] = []
        ratios: list[float] = []
        softs: list[float] = []
        spikes = 0
        step = 0

        for param in group["params"]:
            state = optimizer.state.get(param)
            if not state or "fast" not in state:
                continue
            metrics = get_chaos_metrics(state)
            fasts.append(metrics["fast"])
            slows.append(metrics["slow"])
            ratios.append(metrics["ratio"])
            soft = metrics["soft_chaos"]
            softs.append(soft)
            step = max(step, int(metrics["step"]))
            # Diagnostic hard threshold on the continuous soft gate.
            spikes += int(soft > 0.5)

        n = len(fasts)
        summary.append(
            {
                "group": group_idx,
                "n_params": n,
                "step": step,
                "fast_mean": sum(fasts) / n if n else 0.0,
                "slow_mean": sum(slows) / n if n else 0.0,
                "ratio_mean": sum(ratios) / n if n else 0.0,
                "soft_chaos_mean": sum(softs) / n if n else 0.0,
                "spike_rate": spikes / n if n else 0.0,
            }
        )
    return summary


def layer_norms(model: nn.Module) -> dict[str, float]:
    """Return the current L2 norm of every named parameter in ``model``."""
    with torch.no_grad():
        return {name: float(param.detach().norm()) for name, param in model.named_parameters()}


class NormHistory:
    """Records layer-wise weight norms after every ``optimizer.step()``.

    Wraps ``optimizer.step`` transparently; call :meth:`close` (or use as a
    context manager) to restore the original step method.

    Attributes:
        history: ``{param_name: [norm_after_step_1, norm_after_step_2, ...]}``
        steps: number of recorded steps.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, model: nn.Module) -> None:
        self._optimizer = optimizer
        self._model = model
        self._orig_step = optimizer.step
        self.history: dict[str, list[float]] = {}
        self.steps: int = 0
        self._closed = False

        @functools.wraps(self._orig_step)
        def _wrapped_step(closure: Optional[Any] = None) -> Optional[torch.Tensor]:
            out = self._orig_step(closure)
            self._record()
            return out

        optimizer.step = _wrapped_step  # type: ignore[assignment]

    def _record(self) -> None:
        self.steps += 1
        for name, norm in layer_norms(self._model).items():
            self.history.setdefault(name, []).append(norm)

    def close(self) -> None:
        """Restore the original ``optimizer.step`` and stop recording."""
        if not self._closed:
            self._optimizer.step = self._orig_step  # type: ignore[assignment]
            self._closed = True

    def __enter__(self) -> NormHistory:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def norm_history(optimizer: torch.optim.Optimizer, model: nn.Module) -> NormHistory:
    """Attach a :class:`NormHistory` recorder to ``optimizer``.

    Example::

        tracker = norm_history(optimizer, model)
        for batch in loader:
            ...
            optimizer.step()
        tracker.close()
        plot(tracker.history["encoder.layer.0.attention.weight"])
    """
    return NormHistory(optimizer, model)
