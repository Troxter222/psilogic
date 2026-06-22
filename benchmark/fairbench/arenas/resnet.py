"""Arena 3 -- CNN classification on Tiny ImageNet with ResNet-18/34.

Data: Tiny ImageNet (200 classes, 64x64). The standard archive is downloaded
and extracted automatically, and the flat ``val/`` split is reorganized into
per-class folders so ``ImageFolder`` can read it.

Model: torchvision ResNet-18 (default) or ResNet-34, adapted for 64x64 inputs
(3x3 stem, no initial max-pool) which is the conventional Tiny ImageNet recipe.

Metrics: train/val cross-entropy loss and top-1 validation accuracy.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..datasets import (
    TINY_IMAGENET_DIR,
    TINY_IMAGENET_URL,
    reorganize_tiny_imagenet_val,
    require_local,
    tiny_imagenet_ready,
)
from ..logging_utils import LOGGER
from .base import Arena
from .vit import _SyntheticImages


class ResNetArena(Arena):
    name = "resnet"
    primary_metric = "val_acc"
    primary_mode = "max"
    default_weight_decay = 5e-4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arch = str(self.extra.get("arch", "resnet18"))
        self.img_size = int(self.extra.get("img_size", 64))
        self._num_classes = 200
        self._root = os.path.join(self.data_root, TINY_IMAGENET_DIR)

    def num_classes(self) -> int:
        return self._num_classes

    def _transforms(self):
        from torchvision import transforms

        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
        train_t = transforms.Compose(
            [
                transforms.RandomCrop(self.img_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return train_t, eval_t

    def _prepare(self) -> None:
        if self.synthetic:
            return
        require_local(
            "tiny_imagenet", tiny_imagenet_ready(self.data_root), self.data_root, self.offline
        )
        if tiny_imagenet_ready(self.data_root):
            reorganize_tiny_imagenet_val(os.path.join(self._root, "val"))
            return
        from torchvision.datasets.utils import download_and_extract_archive

        LOGGER.info("Downloading Tiny ImageNet (~240 MB)...")
        download_and_extract_archive(TINY_IMAGENET_URL, download_root=self.data_root)
        reorganize_tiny_imagenet_val(os.path.join(self._root, "val"))

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        self.prepare()
        if self.synthetic:
            train_ds: Dataset = _SyntheticImages(512, self.img_size, self._num_classes, seed=2)
            val_ds: Dataset = _SyntheticImages(128, self.img_size, self._num_classes, seed=3)
        else:
            from torchvision.datasets import ImageFolder

            train_t, eval_t = self._transforms()
            train_ds = ImageFolder(os.path.join(self._root, "train"), transform=train_t)
            val_ds = ImageFolder(os.path.join(self._root, "val"), transform=eval_t)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return train_loader, val_loader

    def build_model(self) -> nn.Module:
        from torchvision import models

        factory = {"resnet18": models.resnet18, "resnet34": models.resnet34}
        if self.arch not in factory:
            raise ValueError(f"Unsupported resnet arch '{self.arch}'.")
        model = factory[self.arch](weights=None, num_classes=self._num_classes)
        # Small-image adaptation: 3x3 stem and no max-pool keeps spatial detail.
        if self.img_size <= 64:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model

    def forward_loss(self, model, batch, device) -> tuple[torch.Tensor, int]:
        x, y = self.to_device(batch, device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, x.size(0)

    @torch.no_grad()
    def evaluate(
        self, model, loader, device, amp_ctx, max_batches: Optional[int] = None
    ) -> dict[str, float]:
        model.eval()
        total_loss, correct, n = 0.0, 0, 0
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with amp_ctx():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            total_loss += float(loss) * x.size(0)
            correct += int((logits.argmax(dim=-1) == y).sum())
            n += x.size(0)
        model.train()
        return {"val_loss": total_loss / max(n, 1), "val_acc": correct / max(n, 1)}

    def psilogic_kwargs(self) -> dict[str, Any]:
        # CNN/vision preset analog.
        return dict(
            gamma=0.04,
            chaos_tau=0.40,
            adaptive_tau=True,
            tau_scale=2.5,
            max_cancel=0.04,
            agc_clip=0.02,
        )
