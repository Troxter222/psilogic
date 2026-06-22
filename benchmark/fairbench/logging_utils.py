"""Unified experiment logging: console, TensorBoard and Weights & Biases.

A single :class:`ExperimentLogger` fans scalar metrics out to every enabled
backend. Both TensorBoard and W&B are optional -- if the package is missing
the corresponding backend silently no-ops, so the benchmark always runs.

Runs are grouped for clean dashboards: TensorBoard uses a
``arena/optimizer/seed`` run-directory hierarchy and W&B uses ``group`` (the
optimizer) plus ``job_type`` (the arena) so curves can be faceted by optimizer
and seed exactly as the paper requires.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

LOGGER = logging.getLogger("fairbench")


def setup_console_logging(level: int = logging.INFO) -> None:
    """Configure a clean, single-handler console logger."""
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


class ExperimentLogger:
    """Fan-out scalar logger for one (arena, optimizer, seed/lr) run.

    Args:
        output_dir: Root results directory.
        arena, optimizer: Grouping keys.
        tag: A sub-identifier such as ``"seed0"`` or ``"sweep_lr1e-3"``.
        use_tb: Enable TensorBoard if ``tensorboard`` is importable.
        use_wandb: Enable W&B if ``wandb`` is importable.
        wandb_project / wandb_entity: W&B destination.
        config: Hyperparameter dict logged for provenance.
    """

    def __init__(
        self,
        output_dir: str,
        arena: str,
        optimizer: str,
        tag: str,
        use_tb: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "fairbench-optimizers",
        wandb_entity: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        device_info: Optional[dict[str, Any]] = None,
    ):
        self.arena = arena
        self.optimizer = optimizer
        self.tag = tag
        self._device_info = device_info or {}
        self._tb = None
        self._wandb_run = None

        if use_tb:
            self._init_tensorboard(output_dir, arena, optimizer, tag)
        if use_wandb:
            self._init_wandb(arena, optimizer, tag, wandb_project, wandb_entity, config)

    def _init_tensorboard(self, output_dir, arena, optimizer, tag) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(output_dir, "tensorboard", arena, optimizer, tag)
            self._tb = SummaryWriter(log_dir=log_dir)
            if self._device_info:
                gpu = self._device_info.get("gpu_name", "CPU")
                device = self._device_info.get("device", "cpu")
                vram = self._device_info.get("gpu_vram_gb", 0.0)
                self._tb.add_text(
                    "hardware/training_device",
                    f"{device} | {gpu} | {vram:.1f} GB VRAM" if vram else f"{device} | {gpu}",
                    0,
                )
        except Exception as exc:  # pragma: no cover - optional dep
            LOGGER.warning("TensorBoard disabled (%s).", exc)

    def _init_wandb(self, arena, optimizer, tag, project, entity, config) -> None:
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=project,
                entity=entity,
                group=optimizer,  # facet by optimizer
                job_type=arena,  # facet by arena
                name=f"{arena}-{optimizer}-{tag}",
                config=config or {},
                reinit=True,
            )
        except Exception as exc:  # pragma: no cover - optional dep
            LOGGER.warning("W&B disabled (%s).", exc)

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a flat dict of scalar metrics at the given global step."""
        if self._tb is not None:
            for key, value in metrics.items():
                try:
                    self._tb.add_scalar(key, value, step)
                except Exception:
                    pass
        if self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception:
                pass

    def close(self) -> None:
        if self._tb is not None:
            try:
                self._tb.flush()
                self._tb.close()
            except Exception:
                pass
            self._tb = None
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
            self._wandb_run = None

    def __enter__(self) -> ExperimentLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
