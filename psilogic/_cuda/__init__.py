"""Optional Triton fused CUDA step backend for PsiLogic."""

from __future__ import annotations

_TRITON_AVAILABLE: bool | None = None


def is_triton_available() -> bool:
    """Return True when the Triton package is importable."""
    global _TRITON_AVAILABLE
    if _TRITON_AVAILABLE is None:
        try:
            import triton  # noqa: F401

            _TRITON_AVAILABLE = True
        except ImportError:
            _TRITON_AVAILABLE = False
    return _TRITON_AVAILABLE


def is_fused_available() -> bool:
    """Return True when the fused CUDA step path can run (CUDA + Triton)."""
    if not is_triton_available():
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def fused_group_step(group: dict, state_dict: dict, *, sync_chaos_ddp: bool, maybe_sync) -> None:
    """Run one fused Triton step for a param group (raises if unavailable)."""
    if not is_fused_available():
        raise RuntimeError("Fused CUDA step requires CUDA and Triton")
    from .step import fused_group_step as _fused_group_step

    _fused_group_step(group, state_dict, sync_chaos_ddp=sync_chaos_ddp, maybe_sync=maybe_sync)
