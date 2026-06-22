"""Self-contained model definitions used by the arenas."""

from __future__ import annotations

from .gpt import GPT, GPTConfig
from .unet import DDPM, UNet

__all__ = ["GPT", "GPTConfig", "UNet", "DDPM"]
