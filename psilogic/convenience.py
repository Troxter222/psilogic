"""Task-specific PsiLogic subclasses with sensible defaults."""

from __future__ import annotations

from typing import Any

from .optimizer import PsiLogic


class PsiLogicNLP(PsiLogic):
    """PsiLogic preset for NLP encoder fine-tuning (BERT, RoBERTa, etc.)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma_T_max: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("gamma", 0.03)
        kwargs.setdefault("chaos_tau", 0.40)
        kwargs.setdefault("chaos_warmup", -1)
        kwargs.setdefault("quantum_decay", 2e-4)
        kwargs.setdefault("agc_clip", 0.01)
        kwargs.setdefault("adaptive_tau", True)
        kwargs.setdefault("tau_scale", 2.0)
        kwargs.setdefault("max_cancel", 0.05)
        super().__init__(params, lr=lr, gamma_T_max=gamma_T_max, **kwargs)


class PsiLogicGPT(PsiLogic):
    """PsiLogic preset for language model training from scratch."""

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        gamma_T_max: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("gamma", 0.02)
        kwargs.setdefault("chaos_tau", 0.40)
        kwargs.setdefault("chaos_warmup", -1)
        kwargs.setdefault("quantum_decay", 0.0)
        kwargs.setdefault("weight_decay", 0.1)
        kwargs.setdefault("agc_clip", 0.01)
        kwargs.setdefault("adaptive_tau", True)
        kwargs.setdefault("tau_scale", 3.0)
        kwargs.setdefault("max_cancel", 0.03)
        super().__init__(params, lr=lr, gamma_T_max=gamma_T_max, **kwargs)


class PsiLogicViT(PsiLogic):
    """PsiLogic preset for Vision Transformer and CNN training."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma_T_max: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("gamma", 0.04)
        kwargs.setdefault("chaos_tau", 0.40)
        kwargs.setdefault("chaos_warmup", -1)
        kwargs.setdefault("quantum_decay", 0.0)
        kwargs.setdefault("agc_clip", 0.02)
        kwargs.setdefault("adaptive_tau", True)
        kwargs.setdefault("tau_scale", 2.5)
        kwargs.setdefault("max_cancel", 0.04)
        super().__init__(params, lr=lr, gamma_T_max=gamma_T_max, **kwargs)
