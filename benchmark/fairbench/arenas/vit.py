"""Arena 2 -- Vision Transformer classification on CIFAR-100 @ 224x224.

Data: CIFAR-100 (torchvision, auto-download) upscaled to 224x224 so the
images match the ViT's native patch grid.

Model: ``vit_tiny_patch16_224`` from ``timm`` when installed, otherwise
torchvision's ``vit_b_16`` head-resized fallback, otherwise a compact built-in
ViT, so the arena always runs.

Metrics: train/val cross-entropy loss and top-1 validation accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..datasets import cifar100_ready, require_local
from ..logging_utils import LOGGER
from .base import Arena


class _SyntheticImages(Dataset):
    """Deterministic synthetic image-classification dataset for smoke tests."""

    def __init__(self, n: int, img_size: int, num_classes: int, channels: int = 3, seed: int = 0):
        self.n = n
        self.img_size = img_size
        self.num_classes = num_classes
        self.channels = channels
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 100003 + idx)
        label = int(torch.randint(0, self.num_classes, (1,), generator=g).item())
        # Class-correlated mean so the task is learnable (sanity for the loop).
        img = torch.randn(self.channels, self.img_size, self.img_size, generator=g)
        img = img + label / self.num_classes
        return img, label


def _build_timm_vit(num_classes: int, img_size: int) -> Optional[nn.Module]:
    try:
        import timm

        return timm.create_model(
            "vit_tiny_patch16_224", pretrained=False, num_classes=num_classes, img_size=img_size
        )
    except Exception as exc:
        LOGGER.warning("timm ViT unavailable (%s); trying torchvision fallback.", exc)
        return None


def _build_torchvision_vit(num_classes: int, img_size: int) -> Optional[nn.Module]:
    try:
        from torchvision.models import vit_b_16

        if img_size != 224:
            return None  # torchvision ViT is fixed at 224
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    except Exception:
        return None


class _TinyViT(nn.Module):
    """Minimal ViT fallback (patch embed -> transformer encoder -> cls head)."""

    def __init__(
        self,
        num_classes: int,
        img_size: int,
        patch: int = 16,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
    ):
        super().__init__()
        assert img_size % patch == 0
        n_patches = (img_size // patch) ** 2
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout=0.0, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


class ViTArena(Arena):
    name = "vit"
    primary_metric = "val_acc"
    primary_mode = "max"
    default_weight_decay = 0.05

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_size = int(self.extra.get("img_size", 224))
        self._num_classes = 100
        self.train_subset = int(self.extra.get("train_subset", 0))  # 0 = full

    def num_classes(self) -> int:
        return self._num_classes

    def _transforms(self):
        from torchvision import transforms

        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
        train_t = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.img_size, padding=max(self.img_size // 28, 4)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_t = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train_t, eval_t

    def _prepare(self) -> None:
        if self.synthetic:
            return
        require_local("cifar100", cifar100_ready(self.data_root), self.data_root, self.offline)
        if cifar100_ready(self.data_root):
            LOGGER.info("CIFAR-100 found locally (%s).", self.data_root)
            return
        from torchvision.datasets import CIFAR100

        LOGGER.info("Downloading CIFAR-100 (~169 MB) -> %s", self.data_root)
        CIFAR100(root=self.data_root, train=True, download=True)
        CIFAR100(root=self.data_root, train=False, download=True)

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        self.prepare()
        if self.synthetic:
            train_ds: Dataset = _SyntheticImages(512, self.img_size, self._num_classes, seed=0)
            val_ds: Dataset = _SyntheticImages(128, self.img_size, self._num_classes, seed=1)
        else:
            from torchvision.datasets import CIFAR100

            train_t, eval_t = self._transforms()
            train_ds = CIFAR100(root=self.data_root, train=True, transform=train_t)
            val_ds = CIFAR100(root=self.data_root, train=False, transform=eval_t)
            if self.train_subset > 0:
                from torch.utils.data import Subset

                train_ds = Subset(train_ds, list(range(min(self.train_subset, len(train_ds)))))

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
        model = _build_timm_vit(self._num_classes, self.img_size)
        if model is None:
            model = _build_torchvision_vit(self._num_classes, self.img_size)
        if model is None:
            LOGGER.warning("Using built-in TinyViT fallback.")
            model = _TinyViT(self._num_classes, self.img_size)
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
        return dict(
            gamma=0.04,
            chaos_tau=0.40,
            adaptive_tau=True,
            tau_scale=2.5,
            max_cancel=0.04,
            agc_clip=0.02,
        )
