"""PsiLogic — Active Cancellation optimizer for deep neural networks."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from .convenience import PsiLogicGPT, PsiLogicNLP, PsiLogicViT
from .optimizer import PsiLogic
from .param_groups import nlp_param_groups
from .presets import gpt_scratch_defaults, nlp_defaults, vision_defaults

try:
    __version__ = _pkg_version("psilogic")
except Exception:
    __version__ = "0.3.2"

__all__ = [
    "PsiLogic",
    "PsiLogicNLP",
    "PsiLogicGPT",
    "PsiLogicViT",
    "nlp_param_groups",
    "nlp_defaults",
    "vision_defaults",
    "gpt_scratch_defaults",
    "__version__",
]
