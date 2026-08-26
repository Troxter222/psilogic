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
    weight_decay: float = 1e-4,
    grad_centralize: bool = False,
) -> dict[str, Any]:
    return {
        "betas": (0.9, 0.999),
        "weight_decay": weight_decay,
        "gamma": gamma,
        "p_ext": 1.0,
        "quantum_decay": quantum_decay,
        "eps": 1e-8,
        "grad_centralize": grad_centralize,
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
    """Hyperparameters for transformer fine-tuning (BERT, RoBERTa, etc.).

    Mild AGC + grad centralization — opt-in extras beyond the plain
    ``PsiLogic(...)`` constructor defaults.
    """
    return _base_defaults(
        gamma=0.03,
        quantum_decay=2e-4,
        tau_scale=2.0,
        max_cancel=0.05,
        agc_clip=0.01,
        grad_centralize=True,
        total_steps=total_steps,
    )


def vision_defaults(total_steps: int = 0) -> dict[str, Any]:
    """Hyperparameters for ViT / CNN vision training.

    Pair with ``vit_param_groups`` for transformer-based vision models so
    patch embeddings receive minimal cancellation (γ=0.005) while attention
    and MLP blocks get the full treatment. Enables AGC + centralize.
    """
    return _base_defaults(
        gamma=0.04,
        quantum_decay=0.0,
        tau_scale=2.5,
        max_cancel=0.04,
        agc_clip=0.02,
        grad_centralize=True,
        total_steps=total_steps,
    )


def gpt_scratch_defaults(total_steps: int = 0) -> dict[str, Any]:
    """Hyperparameters for language model training from scratch.

    Matches the plain drop-in (no AGC / no grad centralize); FairBench
    NLP follow-ups showed those extras hurt TinyStories GPT-scratch vs AdamW.
    ``chaos_warmup=-1`` auto-scales the warmup to ``max(500, steps // 20)``.
    Pass the real ``total_steps`` for warmup auto-scale and cosine γ decay.
    """
    return _base_defaults(
        gamma=0.02,
        quantum_decay=0.0,
        tau_scale=3.0,
        max_cancel=0.03,
        agc_clip=0.0,
        grad_centralize=False,
        total_steps=total_steps,
        weight_decay=1e-1,
    )


def whisper_defaults(total_steps: int = 0) -> dict[str, Any]:
    """Hyperparameters for speech model fine-tuning (Whisper, wav2vec, etc.).

    Audio encoders are sensitive to aggressive shrinkage on convolutional
    front-ends, so cancellation is gentler than the NLP preset.
    """
    return _base_defaults(
        gamma=0.02,
        quantum_decay=1e-4,
        tau_scale=2.0,
        max_cancel=0.04,
        agc_clip=0.01,
        grad_centralize=True,
        total_steps=total_steps,
        weight_decay=1e-2,
    )


def glue_defaults(total_steps: int = 0) -> dict[str, Any]:
    """Hyperparameters for GLUE-style encoder fine-tuning (BERT-large etc.)."""
    return _base_defaults(
        gamma=0.03,
        quantum_decay=2e-4,
        tau_scale=2.0,
        max_cancel=0.05,
        agc_clip=0.01,
        grad_centralize=True,
        total_steps=total_steps,
        weight_decay=1e-2,
    )
