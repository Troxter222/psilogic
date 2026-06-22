"""Benchmark arenas: one adapter per (task, dataset, model) triple.

Each arena subclasses :class:`fairbench.arenas.base.Arena` and is registered
in :data:`ARENA_REGISTRY` so the runner can build it by name.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Arena
from .diffusion import DiffusionArena
from .nlp import NLPArena
from .resnet import ResNetArena
from .vit import ViTArena

ARENA_REGISTRY: dict[str, type[Arena]] = {
    "nlp": NLPArena,
    "vit": ViTArena,
    "resnet": ResNetArena,
    "diffusion": DiffusionArena,
}


def build_arena(name: str, **kwargs) -> Arena:
    """Instantiate an arena by its registry name."""
    if name not in ARENA_REGISTRY:
        raise ValueError(f"Unknown arena '{name}'. Available: {sorted(ARENA_REGISTRY)}")
    return ARENA_REGISTRY[name](**kwargs)


__all__ = [
    "Arena",
    "ARENA_REGISTRY",
    "build_arena",
    "NLPArena",
    "ViTArena",
    "ResNetArena",
    "DiffusionArena",
]
