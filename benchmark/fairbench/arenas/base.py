"""Abstract arena interface shared by all four benchmark tasks.

An :class:`Arena` fully encapsulates a (dataset, model, objective) triple
behind a uniform API so the training engine in :mod:`fairbench.runner` is
completely task-agnostic. Concrete arenas only implement data preparation,
model construction, the per-batch loss and an evaluation pass.

Contract guarantees that keep the comparison fair:

* :meth:`build_model` must be deterministic given the global RNG state, so the
  runner can snapshot identical initial weights for every optimizer.
* :meth:`forward_loss` and :meth:`evaluate` must not depend on the optimizer.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Arena(abc.ABC):
    """Base class for a benchmark arena."""

    #: Short canonical name (matches the registry key).
    name: str = "base"
    #: The metric used to *select* the LR in the sweep and to rank runs.
    primary_metric: str = "val_loss"
    #: "min" if lower primary_metric is better, else "max".
    primary_mode: str = "min"
    #: Default decoupled/L2 weight decay for this task.
    default_weight_decay: float = 0.0

    def __init__(
        self,
        data_root: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        synthetic: bool = False,
        offline: bool = False,
        extra: Optional[dict[str, Any]] = None,
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.synthetic = synthetic
        self.offline = offline
        self.extra = dict(extra or {})
        self._prepared = False

    # ----------------------------- data --------------------------------- #

    def prepare(self) -> None:
        """Download / preprocess the dataset once (idempotent)."""
        if self._prepared:
            return
        self._prepare()
        self._prepared = True

    @abc.abstractmethod
    def _prepare(self) -> None:
        """Arena-specific one-time data preparation."""

    @abc.abstractmethod
    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Return ``(train_loader, val_loader)``."""

    # ----------------------------- model -------------------------------- #

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """Construct a fresh model. Determinism is the caller's responsibility
        (it seeds the RNG before calling), so two calls under the same seed
        yield identical initial weights."""

    # ----------------------------- objective ---------------------------- #

    @abc.abstractmethod
    def forward_loss(
        self, model: nn.Module, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        """Compute the training loss for one batch.

        Returns ``(loss, num_examples)`` where ``num_examples`` is used for
        throughput accounting.
        """

    @abc.abstractmethod
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        amp_ctx,
        max_batches: Optional[int] = None,
    ) -> dict[str, float]:
        """Run a validation pass and return a dict of metrics.

        Must include ``primary_metric``. ``amp_ctx`` is a zero-arg callable
        returning an autocast context manager (or nullcontext).
        """

    # ----------------------------- optimizer hints ---------------------- #

    def psilogic_kwargs(self) -> dict[str, Any]:
        """Arena-appropriate PsiLogic hyperparameters (besides lr/wd).

        These mirror the library's published presets per architecture. They
        are held fixed across the whole benchmark; only the LR is tuned.
        """
        return {}

    # ----------------------------- helpers ------------------------------ #

    def to_device(self, batch: Any, device: torch.device) -> Any:
        """Move a batch (tensor / tuple / list / dict) to ``device``."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self.to_device(b, device) for b in batch)
        if isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        return batch

    def num_classes(self) -> Optional[int]:
        """Number of classes for classification arenas (else ``None``)."""
        return None
