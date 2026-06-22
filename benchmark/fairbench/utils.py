"""Small cross-cutting helpers: seeding, device/AMP setup and schedulers."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
from typing import Any, Callable, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs so model initialization and data order are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Return a usable device, falling back to CPU when CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        from .logging_utils import LOGGER

        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def describe_device(device: torch.device) -> dict[str, Any]:
    """Collect human-readable hardware metadata for the active training device."""
    info: dict[str, Any] = {
        "device": str(device),
        "device_type": device.type,
        "gpu_name": "CPU",
        "gpu_index": -1,
        "gpu_vram_gb": 0.0,
        "cuda_version": None,
        "torch_version": torch.__version__,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            gpu_name=props.name,
            gpu_index=int(idx),
            gpu_vram_gb=round(props.total_memory / (1024**3), 2),
            cuda_version=torch.version.cuda,
        )
    return info


def format_device_label(device_info: dict[str, Any]) -> str:
    """Short one-line label for logs, e.g. ``GPU: NVIDIA A100 (40.0 GB)``."""
    if device_info.get("device_type") == "cuda" and device_info.get("gpu_name") != "CPU":
        vram = device_info.get("gpu_vram_gb", 0.0)
        idx = device_info.get("gpu_index", 0)
        name = device_info["gpu_name"]
        if vram:
            return f"GPU #{idx}: {name} ({vram:.1f} GB VRAM)"
        return f"GPU #{idx}: {name}"
    return "CPU"


def log_device_banner(device_info: dict[str, Any]) -> None:
    """Print a prominent console banner with the active training device."""
    from .logging_utils import LOGGER

    label = format_device_label(device_info)
    line = "=" * max(60, len(label) + 10)
    LOGGER.info(line)
    LOGGER.info("Training device: %s", label)
    if device_info.get("cuda_version"):
        LOGGER.info(
            "PyTorch %s | CUDA %s",
            device_info.get("torch_version", "?"),
            device_info["cuda_version"],
        )
    else:
        LOGGER.info("PyTorch %s", device_info.get("torch_version", "?"))
    LOGGER.info(line)


def save_config_with_hardware(cfg, output_path: str, device_info: dict[str, Any]) -> None:
    """Write the benchmark config JSON augmented with detected hardware info."""
    payload = cfg.to_dict()
    payload["runtime_hardware"] = device_info
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def amp_dtype_from_str(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(name.lower(), torch.float16)


def make_autocast(
    device: torch.device, enabled: bool, dtype: torch.dtype
) -> Callable[[], contextlib.AbstractContextManager]:
    """Return a zero-arg factory producing an autocast context (or nullcontext).

    bfloat16 on CUDA / any AMP on CPU does not need a GradScaler; float16 on
    CUDA does. The returned callable is what arenas receive as ``amp_ctx``.
    """
    use_amp = enabled and device.type in ("cuda", "cpu")

    def ctx() -> contextlib.AbstractContextManager:
        if not use_amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=dtype)

    return ctx


def cosine_warmup_lambda(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """LR multiplier: linear warmup then cosine decay to (near) zero."""

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return fn


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_oom_error(exc: BaseException) -> bool:
    """True if the exception is a CUDA out-of-memory error."""
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )
