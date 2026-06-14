"""PyTorch Lightning integration.

Two pieces, both importable without Lightning installed:

- :func:`configure_psilogic` — drop-in body for
  ``LightningModule.configure_optimizers``.
- :class:`ChaosMonitorCallback` — logs PsiLogic chaos health metrics during
  training (degrades to a plain object when Lightning is unavailable).

Usage::

    class LitModel(L.LightningModule):
        def configure_optimizers(self):
            return configure_psilogic(
                self, preset="auto", lr=3e-4,
                total_steps=self.trainer.estimated_stepping_batches,
            )

    trainer = L.Trainer(callbacks=[ChaosMonitorCallback(log_every_n_steps=50)])
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from ..optimizer import PsiLogic
from .hf import create_psilogic_optimizer

try:  # Lightning >= 2.0
    from lightning.pytorch.callbacks import Callback as _CallbackBase

    _LIGHTNING_AVAILABLE = True
except (ImportError, RuntimeError):  # pragma: no cover - optional / broken stacks
    try:  # legacy package name
        from pytorch_lightning.callbacks import Callback as _CallbackBase

        _LIGHTNING_AVAILABLE = True
    except (ImportError, RuntimeError):
        _CallbackBase = object  # type: ignore[assignment,misc]
        _LIGHTNING_AVAILABLE = False


def configure_psilogic(
    model: nn.Module,
    preset: str = "auto",
    lr: Optional[float] = None,
    total_steps: int = 0,
    **kwargs: Any,
) -> PsiLogic:
    """Build a PsiLogic optimizer for use in ``configure_optimizers``.

    ``total_steps`` should usually be
    ``self.trainer.estimated_stepping_batches`` so cosine γ decay and the
    chaos warmup auto-scale match the real training horizon.
    """
    return create_psilogic_optimizer(
        model, None, preset=preset, lr=lr, total_steps=total_steps, **kwargs
    )


class ChaosMonitorCallback(_CallbackBase):  # type: ignore[valid-type,misc]
    """Logs PsiLogic chaos statistics (fast/slow EMA, spike rate) to the
    Lightning logger every ``log_every_n_steps`` optimizer steps."""

    def __init__(self, log_every_n_steps: int = 50, prefix: str = "psilogic") -> None:
        if not _LIGHTNING_AVAILABLE:
            raise ImportError(
                "ChaosMonitorCallback requires lightning or pytorch_lightning. "
                "Install with: pip install lightning"
            )
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.prefix = prefix
        self._step_count = 0

    def on_before_optimizer_step(
        self,
        trainer: Any,
        pl_module: Any,
        optimizer: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not isinstance(optimizer, PsiLogic):
            return
        self._step_count += 1
        if self._step_count % self.log_every_n_steps != 0:
            return

        from ..debug import chaos_stats

        metrics: dict[str, float] = {}
        for group_stats in chaos_stats(optimizer):
            gid = group_stats["group"]
            metrics[f"{self.prefix}/group{gid}_fast"] = group_stats["fast_mean"]
            metrics[f"{self.prefix}/group{gid}_slow"] = group_stats["slow_mean"]
            metrics[f"{self.prefix}/group{gid}_spike_rate"] = group_stats["spike_rate"]
        if metrics:
            pl_module.log_dict(metrics, on_step=True, on_epoch=False)
