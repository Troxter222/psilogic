"""Arena 1 -- NLP language modeling on TinyStories with a small GPT.

Data: ``roneneldan/TinyStories`` via HuggingFace ``datasets`` when available,
with a byte-level tokenizer (vocab=256) so there is no tokenizer dependency
and any text corpus works. If ``datasets`` or the network is unavailable, a
synthetic stochastic-grammar corpus is generated so the arena still runs.

Model: the dependency-free :class:`fairbench.models.GPT`.

Metrics: train/val cross-entropy loss and perplexity (= exp(loss)).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..datasets import (
    load_tinystories_cache,
    require_local,
    save_tinystories_cache,
    tinystories_ready,
)
from ..logging_utils import LOGGER
from ..models import GPT, GPTConfig
from .base import Arena


class _BlockDataset(Dataset):
    """Samples fixed-length ``(input, target)`` windows from a 1-D token array.

    ``target`` is ``input`` shifted by one position -- the standard
    next-token-prediction setup.
    """

    def __init__(self, data: np.ndarray, block_size: int, length: int):
        self.data = data
        self.block_size = block_size
        # Number of distinct sampling positions; capped to a virtual length so
        # an epoch is a fixed, comparable number of steps across arenas.
        self.length = min(length, max(len(data) - block_size - 1, 1))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - self.block_size - 1
        start = idx % max(max_start, 1)
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def _synthetic_corpus(n_chars: int, seed: int = 0) -> str:
    """A tiny structured corpus (repeating templated sentences + noise)."""
    rng = np.random.default_rng(seed)
    subjects = ["the cat", "a dog", "the girl", "a boy", "the bird", "my friend"]
    verbs = ["ran to", "looked at", "found", "played with", "liked", "saw"]
    objects = ["the park", "a ball", "the tree", "her house", "the lake", "a toy"]
    out = []
    while sum(len(s) for s in out) < n_chars:
        s = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}. "
        out.append(s)
    return "".join(out)[:n_chars]


class NLPArena(Arena):
    name = "nlp"
    primary_metric = "val_loss"
    primary_mode = "min"
    default_weight_decay = 0.1  # GPT-2 convention

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block_size = int(self.extra.get("block_size", 128))
        self.vocab_size = 256  # byte-level
        self.n_layer = int(self.extra.get("n_layer", 4))
        self.n_head = int(self.extra.get("n_head", 4))
        self.n_embd = int(self.extra.get("n_embd", 256))
        self.train_chars = int(self.extra.get("train_chars", 2_000_000))
        self.val_chars = int(self.extra.get("val_chars", 200_000))
        self.steps_per_epoch = int(self.extra.get("steps_per_epoch", 1000))
        self._train_ids: Optional[np.ndarray] = None
        self._val_ids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #

    def _load_text(self) -> tuple[str, str]:
        if self.synthetic:
            return _synthetic_corpus(self.train_chars), _synthetic_corpus(self.val_chars, seed=1)

        # Optional explicit text files (upload your own corpus).
        train_path = self.extra.get("train_text_path")
        val_path = self.extra.get("val_text_path")
        if train_path and val_path and os.path.isfile(train_path) and os.path.isfile(val_path):
            LOGGER.info("Loading NLP text from local files: %s , %s", train_path, val_path)
            with open(train_path, encoding="utf-8", errors="replace") as fh:
                train_text = fh.read()[: self.train_chars]
            with open(val_path, encoding="utf-8", errors="replace") as fh:
                val_text = fh.read()[: self.val_chars]
            return train_text, val_text

        require_local(
            "tinystories", tinystories_ready(self.data_root), self.data_root, self.offline
        )
        if tinystories_ready(self.data_root):
            train_ids, val_ids = load_tinystories_cache(self.data_root)
            return train_ids.tobytes().decode("latin-1"), val_ids.tobytes().decode("latin-1")

        try:
            from ..datasets import fetch_tinystories_text

            LOGGER.info("Loading TinyStories (will cache locally)...")
            train_text = fetch_tinystories_text("train", self.train_chars)
            val_text = fetch_tinystories_text("validation", self.val_chars)
            return train_text, val_text
        except Exception as exc:
            if self.offline:
                raise RuntimeError(
                    f"Offline mode: TinyStories cache missing under {self.data_root}/tinystories/. "
                    "Run: python -m fairbench.download --data-root <path>"
                ) from exc
            LOGGER.warning("TinyStories unavailable (%s); using synthetic corpus.", exc)
            return _synthetic_corpus(self.train_chars), _synthetic_corpus(self.val_chars, seed=1)

    def _prepare(self) -> None:
        if tinystories_ready(self.data_root):
            train_ids, val_ids = load_tinystories_cache(self.data_root)
            self._train_ids = train_ids
            self._val_ids = val_ids
            LOGGER.info(
                "NLP arena ready (local cache): %d train tokens, %d val tokens.",
                len(self._train_ids),
                len(self._val_ids),
            )
            return

        train_text, val_text = self._load_text()
        self._train_ids = np.frombuffer(
            train_text.encode("utf-8", "replace"), dtype=np.uint8
        ).copy()
        self._val_ids = np.frombuffer(val_text.encode("utf-8", "replace"), dtype=np.uint8).copy()
        if not self.synthetic and not self.offline:
            save_tinystories_cache(
                self.data_root, self._train_ids, self._val_ids, self.train_chars, self.val_chars
            )
        LOGGER.info(
            "NLP arena ready: %d train tokens, %d val tokens.",
            len(self._train_ids),
            len(self._val_ids),
        )

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        self.prepare()
        assert self._train_ids is not None and self._val_ids is not None
        train_ds = _BlockDataset(
            self._train_ids, self.block_size, self.steps_per_epoch * self.batch_size
        )
        val_ds = _BlockDataset(
            self._val_ids, self.block_size, max(self.steps_per_epoch // 5, 1) * self.batch_size
        )
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
        cfg = GPTConfig(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
        )
        return GPT(cfg)

    def forward_loss(
        self, model: nn.Module, batch: Any, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        x, y = self.to_device(batch, device)
        _, loss = model(x, y)
        return loss, x.size(0)

    @torch.no_grad()
    def evaluate(
        self, model, loader, device, amp_ctx, max_batches: Optional[int] = None
    ) -> dict[str, float]:
        model.eval()
        total_loss, n = 0.0, 0
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with amp_ctx():
                _, loss = model(x, y)
            total_loss += float(loss) * x.size(0)
            n += x.size(0)
        model.train()
        avg = total_loss / max(n, 1)
        ppl = float(np.exp(min(avg, 20.0)))  # clamp to avoid overflow on diverged runs
        return {"val_loss": avg, "perplexity": ppl}

    def psilogic_kwargs(self) -> dict[str, Any]:
        # Mirrors psilogic's GPT-from-scratch preset (minus lr/wd which the
        # factory injects). Conservative chaos for unstable from-scratch LM.
        return dict(
            gamma=0.02,
            chaos_tau=0.40,
            adaptive_tau=True,
            tau_scale=3.0,
            max_cancel=0.03,
            agc_clip=0.01,
        )
