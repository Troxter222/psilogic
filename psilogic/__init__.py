"""PsiLogic — Active Cancellation optimizer for deep neural networks."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from . import debug
from ._chaos import get_chaos_metrics
from ._version import __version__ as _fallback_version
from .convenience import (
    PsiLogicGPT,
    PsiLogicNLP,
    PsiLogicViT,
    PsiLogicWhisper,
    build_auto_optimizer,
    infer_architecture,
)
from .optimizer import PsiLogic
from .param_groups import gpt_param_groups, nlp_param_groups, vit_param_groups
from .presets import (
    glue_defaults,
    gpt_scratch_defaults,
    nlp_defaults,
    vision_defaults,
    whisper_defaults,
)

try:
    __version__ = _pkg_version("psilogic")
except Exception:
    __version__ = _fallback_version

__all__ = [
    "PsiLogic",  # includes PsiLogic.auto() classmethod
    "PsiLogicNLP",
    "PsiLogicGPT",
    "PsiLogicViT",
    "PsiLogicWhisper",
    "nlp_param_groups",
    "vit_param_groups",
    "gpt_param_groups",
    "nlp_defaults",
    "vision_defaults",
    "gpt_scratch_defaults",
    "whisper_defaults",
    "glue_defaults",
    "get_chaos_metrics",
    "infer_architecture",
    "build_auto_optimizer",
    "debug",
    "__version__",
]
