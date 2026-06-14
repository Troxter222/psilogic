"""Framework integrations for PsiLogic (HuggingFace Transformers, Lightning)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "create_psilogic_optimizer",
    "psilogic_trainer_class",
    "configure_psilogic",
    "ChaosMonitorCallback",
]


def __getattr__(name: str) -> Any:
    if name in ("create_psilogic_optimizer", "psilogic_trainer_class"):
        from .hf import create_psilogic_optimizer, psilogic_trainer_class

        return {
            "create_psilogic_optimizer": create_psilogic_optimizer,
            "psilogic_trainer_class": psilogic_trainer_class,
        }[name]
    if name in ("configure_psilogic", "ChaosMonitorCallback"):
        from .lightning import ChaosMonitorCallback, configure_psilogic

        return {
            "configure_psilogic": configure_psilogic,
            "ChaosMonitorCallback": ChaosMonitorCallback,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
