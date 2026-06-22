"""Arena 4 -- unconditional DDPM image generation on CelebA @ 64x64.

Data: CelebA (torchvision, auto-download) center-cropped to the face and
resized to 64x64, normalized to [-1, 1]. CelebA's Google-Drive hosting is
flaky; on any failure the arena falls back to a synthetic image dataset so the
benchmark still completes.

Model: :class:`fairbench.models.DDPM` wrapping the compact
:class:`fairbench.models.UNet` (epsilon-prediction).

Metrics: train/val denoising MSE loss, and optionally FID (when
``torchmetrics`` is installed and ``extra.compute_fid=True``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..datasets import (
    celeba_ready,
    celeba_uses_torchvision,
    find_celeba_image_dirs,
    require_local,
)
from ..logging_utils import LOGGER
from ..models import DDPM, UNet
from .base import Arena


class _SyntheticFaces(Dataset):
    """Smooth low-frequency synthetic 'faces' so DDPM training is meaningful."""

    def __init__(self, n: int, img_size: int, seed: int = 0):
        self.n = n
        self.img_size = img_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(self.seed * 7919 + idx)
        low = torch.randn(3, self.img_size // 8, self.img_size // 8, generator=g)
        img = torch.nn.functional.interpolate(
            low[None], size=self.img_size, mode="bilinear", align_corners=False
        )[0]
        return img.tanh(), 0  # already in [-1, 1]


class _CelebAFlatImages(Dataset):
    """Read CelebA JPEGs from a flat ``img_align_celeba/`` folder (HF layout)."""

    def __init__(self, image_dir: str, transform=None, indices: Optional[list[int]] = None):
        from PIL import Image

        self._Image = Image
        self.image_dir = image_dir
        self.transform = transform
        files = sorted(
            f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if indices is not None:
            files = [files[i] for i in indices]
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = os.path.join(self.image_dir, self.files[idx])
        img = self._Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class DiffusionArena(Arena):
    name = "diffusion"
    primary_metric = "val_loss"
    primary_mode = "min"
    default_weight_decay = 0.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.img_size = int(self.extra.get("img_size", 64))
        self.base_ch = int(self.extra.get("base_ch", 64))
        self.timesteps = int(self.extra.get("timesteps", 1000))
        self.compute_fid = bool(self.extra.get("compute_fid", False))
        self.fid_samples = int(self.extra.get("fid_samples", 256))
        self.train_subset = int(self.extra.get("train_subset", 0))

    def _transforms(self):
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.CenterCrop(178),  # CelebA face crop
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
            ]
        )

    def _celeba(self, split: str):
        from torchvision.datasets import CelebA

        return CelebA(
            root=self.data_root,
            split=split,
            target_type=[],
            download=not self.offline,
            transform=self._transforms(),
        )

    def _celeba_from_hf_dirs(self, dirs: dict[str, str]) -> tuple[Dataset, Dataset]:
        """Build train/val sets from HuggingFace-style CelebA folders."""
        transform = self._transforms()
        train_dir = dirs["train"]
        if "valid" in dirs:
            return (
                _CelebAFlatImages(train_dir, transform),
                _CelebAFlatImages(dirs["valid"], transform),
            )
        if "test" in dirs:
            return (
                _CelebAFlatImages(train_dir, transform),
                _CelebAFlatImages(dirs["test"], transform),
            )
        # Only train split — hold out 5% for validation (deterministic).
        files = sorted(
            f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        n_val = max(len(files) // 20, 1)
        val_idx = list(range(n_val))
        train_idx = list(range(n_val, len(files)))
        LOGGER.info(
            "CelebA HF layout: %d train / %d val images (from %s).",
            len(train_idx),
            len(val_idx),
            train_dir,
        )
        return (
            _CelebAFlatImages(train_dir, transform, train_idx),
            _CelebAFlatImages(train_dir, transform, val_idx),
        )

    def _prepare(self) -> None:
        if self.synthetic:
            self._use_synthetic = True
            return
        require_local("celeba", celeba_ready(self.data_root), self.data_root, self.offline)
        if celeba_ready(self.data_root):
            layout = find_celeba_image_dirs(self.data_root)
            kind = "torchvision" if celeba_uses_torchvision(self.data_root) else "huggingface"
            LOGGER.info("CelebA found locally (%s layout, %s).", kind, self.data_root)
            self._celeba_layout = layout
            self._use_synthetic = False
            return
        try:
            LOGGER.info("Downloading CelebA (~1.3 GB) -> %s", self.data_root)
            self._celeba("train")
            self._use_synthetic = False
        except Exception as exc:
            if self.offline:
                raise RuntimeError(
                    f"Offline mode: CelebA not found under {self.data_root}/celeba/. "
                    "Run: python -m fairbench.download --data-root <path>"
                ) from exc
            LOGGER.warning("CelebA unavailable (%s); using synthetic faces.", exc)
            self._use_synthetic = True

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        self.prepare()
        if getattr(self, "_use_synthetic", True):
            train_ds: Dataset = _SyntheticFaces(1024, self.img_size, seed=0)
            val_ds: Dataset = _SyntheticFaces(256, self.img_size, seed=1)
        else:
            layout = getattr(self, "_celeba_layout", find_celeba_image_dirs(self.data_root))
            if celeba_uses_torchvision(self.data_root):
                train_ds = self._celeba("train")
                val_ds = self._celeba("valid")
            else:
                train_ds, val_ds = self._celeba_from_hf_dirs(layout)
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
            drop_last=True,
        )
        return train_loader, val_loader

    def build_model(self) -> nn.Module:
        unet = UNet(in_ch=3, base_ch=self.base_ch, ch_mults=(1, 2, 2), attn_at=(1,))
        return DDPM(unet, timesteps=self.timesteps)

    @staticmethod
    def _unpack(batch):
        # CelebA with target_type=[] yields (img, []) ; synthetic yields (img, 0).
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    def forward_loss(self, model, batch, device) -> tuple[torch.Tensor, int]:
        x = self.to_device(self._unpack(batch), device)
        loss = model.loss(x)
        return loss, x.size(0)

    @torch.no_grad()
    def evaluate(
        self, model, loader, device, amp_ctx, max_batches: Optional[int] = None
    ) -> dict[str, float]:
        model.eval()
        # Fix RNG so the (timestep, noise) draw is identical across optimizers,
        # making the held-out denoising MSE directly comparable.
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(0)
        total_loss, n = 0.0, 0
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = self._unpack(batch).to(device)
            with amp_ctx():
                loss = model.loss(x)
            total_loss += float(loss) * x.size(0)
            n += x.size(0)
        torch.random.set_rng_state(gen_state)

        metrics = {"val_loss": total_loss / max(n, 1)}
        if self.compute_fid:
            fid = self._maybe_fid(model, loader, device)
            if fid is not None:
                metrics["fid"] = fid
        model.train()
        return metrics

    def _maybe_fid(self, model, loader, device) -> Optional[float]:
        """Compute FID between generated and real images if torchmetrics exists."""
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except Exception as exc:
            LOGGER.warning("FID skipped (torchmetrics missing: %s).", exc)
            return None
        try:
            fid = FrechetInceptionDistance(normalize=True).to(device)

            def to_uint(t):
                return (t.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

            real_seen = 0
            for batch in loader:
                imgs = self._unpack(batch).to(device)
                fid.update(to_uint(imgs), real=True)
                real_seen += imgs.size(0)
                if real_seen >= self.fid_samples:
                    break
            remaining = self.fid_samples
            ddpm: DDPM = model  # type: ignore
            while remaining > 0:
                n = min(self.batch_size, remaining)
                samples = ddpm.sample(n, 3, self.img_size, device)
                fid.update(to_uint(samples), real=False)
                remaining -= n
            return float(fid.compute())
        except Exception as exc:
            LOGGER.warning("FID computation failed (%s).", exc)
            return None

    def psilogic_kwargs(self) -> dict[str, Any]:
        return dict(
            gamma=0.03,
            chaos_tau=0.40,
            adaptive_tau=True,
            tau_scale=2.0,
            max_cancel=0.04,
            agc_clip=0.02,
        )
