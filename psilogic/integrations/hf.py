"""HuggingFace Transformers integration.

The core entry point, :func:`create_psilogic_optimizer`, has **no**
transformers dependency — it accepts any object exposing ``learning_rate`` /
``weight_decay`` / ``max_steps`` attributes (e.g. ``TrainingArguments``).

Usage with the HF ``Trainer``::

    from psilogic.integrations.hf import psilogic_trainer_class

    PsiLogicTrainer = psilogic_trainer_class()
    trainer = PsiLogicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        psilogic_preset="nlp",          # "nlp" | "gpt" | "vit" | "auto"
        psilogic_kwargs={"gamma": 0.03},
    )
    trainer.train()
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from ..convenience import infer_architecture
from ..optimizer import PsiLogic
from ..param_groups import gpt_param_groups, nlp_param_groups, vit_param_groups
from ..presets import gpt_scratch_defaults, nlp_defaults, vision_defaults

_PRESETS = {
    "nlp": (nlp_param_groups, nlp_defaults, 2e-5),
    "gpt": (gpt_param_groups, gpt_scratch_defaults, 3e-4),
    "vit": (vit_param_groups, vision_defaults, 1e-3),
}

_ARCH_TO_PRESET = {
    "vit": "vit",
    "vision": "vit",
    "gpt": "gpt",
    "nlp": "nlp",
    "generic": "nlp",
}


def create_psilogic_optimizer(
    model: nn.Module,
    args: Optional[Any] = None,
    **kwargs: Any,
) -> PsiLogic:
    """Build a PsiLogic optimizer for a HuggingFace model.

    Args:
        model: The model to optimize.
        args: Optional ``TrainingArguments``-like object. ``learning_rate``,
            ``weight_decay`` and ``max_steps`` are read when present and not
            explicitly overridden via ``kwargs``.
        **kwargs: ``preset`` ("nlp" | "gpt" | "vit" | "auto", default "auto"),
            ``lr``, ``total_steps``, plus any PsiLogic constructor argument
            as an override.

    Returns:
        Configured :class:`PsiLogic` instance with task-specific param groups.
    """
    preset = kwargs.pop("preset", "auto")
    lr = kwargs.pop("lr", None)
    total_steps = int(kwargs.pop("total_steps", 0) or 0)

    if args is not None:
        if lr is None:
            lr = getattr(args, "learning_rate", None)
        if total_steps <= 0:
            total_steps = int(getattr(args, "max_steps", 0) or 0)
        args_wd = getattr(args, "weight_decay", None)
        if args_wd is not None and args_wd > 0:
            kwargs.setdefault("weight_decay", float(args_wd))

    if preset == "auto":
        preset = _ARCH_TO_PRESET[infer_architecture(model)]
    if preset not in _PRESETS:
        raise ValueError(
            f"Unknown psilogic preset {preset!r}; expected one of {sorted(_PRESETS)} or 'auto'."
        )

    builder, preset_fn, default_lr = _PRESETS[preset]
    lr = float(lr) if lr is not None else default_lr

    defaults = preset_fn(total_steps)
    weight_decay = kwargs.pop("weight_decay", defaults["weight_decay"])
    defaults["weight_decay"] = weight_decay
    defaults.update(kwargs)

    groups = builder(model, lr=lr, weight_decay=weight_decay)
    return PsiLogic(groups, lr=lr, **defaults)


def psilogic_trainer_class() -> type[Any]:
    """Return a ``transformers.Trainer`` subclass wired to PsiLogic.

    Imported lazily so the psilogic package itself never requires
    transformers. Raises ``ImportError`` with install instructions when
    transformers is unavailable.
    """
    try:
        from transformers import Trainer
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "psilogic_trainer_class() requires the 'transformers' package. "
            "Install it with: pip install transformers"
        ) from exc

    class PsiLogicTrainer(Trainer):
        """HF Trainer that creates a PsiLogic optimizer instead of AdamW."""

        def __init__(
            self,
            *args: Any,
            psilogic_preset: str = "auto",
            psilogic_kwargs: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            self._psilogic_preset = psilogic_preset
            self._psilogic_kwargs = dict(psilogic_kwargs or {})
            super().__init__(*args, **kwargs)

        def create_optimizer(self) -> PsiLogic:
            optimizer: PsiLogic | None = getattr(self, "optimizer", None)
            if optimizer is None:
                optimizer = create_psilogic_optimizer(
                    self.model,
                    self.args,
                    preset=self._psilogic_preset,
                    **self._psilogic_kwargs,
                )
                self.optimizer = optimizer
            return optimizer

    return PsiLogicTrainer
