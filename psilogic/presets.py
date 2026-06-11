"""Recommended hyperparameter presets for common training tasks."""

from __future__ import annotations

from typing import Any


def _base_defaults(
    *,
    gamma: float,
    quantum_decay: float,
    tau_scale: float,
    max_cancel: float,
    agc_clip: float,
    total_steps: int,
) -> dict[str, Any]:
    return {
        "betas": (0.9, 0.999),
        "weight_decay": 1e-4,
        "gamma": gamma,
        "p_ext": 1.0,
        "quantum_decay": quantum_decay,
        "eps": 1e-8,
        "grad_centralize": True,
        "chaos_tau": 0.40,
        "chaos_warmup": -1,
        "adaptive_tau": True,
        "tau_scale": tau_scale,
        "max_cancel": max_cancel,
        "agc_clip": agc_clip,
        "gamma_T_max": total_steps,
        "use_foreach": True,
    }


def nlp_defaults(total_steps: int = 0) -> dict[str, Any]:
    """Hyperparameters for transformer fine-tuning (BERT, RoBERTa, etc.)."""
    return _base_defaults(
        gamma=0.03,
        quantum_decay=2e-4,
        tau_scale=2.0,
        max_cancel=0.05,
        agc_clip=0.01,
        total_steps=total_steps,
    )


def vision_defaults(total_steps: int = 0) -> dict[str, Any]:
    """Hyperparameters for ViT / CNN vision training."""
    return _base_defaults(
        gamma=0.04,
        quantum_decay=0.0,
        tau_scale=2.5,
        max_cancel=0.04,
        agc_clip=0.02,
        total_steps=total_steps,
    )


def gpt_scratch_defaults(total_steps: int = 0) -> dict[str, Any]:
    """Hyperparameters for language model training from scratch."""
    defaults = _base_defaults(
        gamma=0.02,
        quantum_decay=0.0,
        tau_scale=3.0,
        max_cancel=0.03,
        agc_clip=0.01,
        total_steps=total_steps,
    )
    defaults["weight_decay"] = 1e-1
    return defaults
