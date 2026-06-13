"""Task-specific PsiLogic subclasses, architecture inference and zero-config."""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from .optimizer import PsiLogic
from .param_groups import gpt_param_groups, nlp_param_groups, vit_param_groups
from .presets import gpt_scratch_defaults, nlp_defaults, vision_defaults, whisper_defaults

_AUTO_LR: dict[str, float] = {
    "vit": 1e-3,
    "vision": 1e-3,
    "gpt": 3e-4,
    "nlp": 1e-3,
    "generic": 1e-3,
}

_VIT_NAME_MARKERS: tuple[str, ...] = (
    "patch_embed",
    "patch_embeddings",
    "cls_token",
    "class_token",
    "pos_embed",
    "conv_proj",
)
_GPT_NAME_MARKERS: tuple[str, ...] = (
    "wte",
    "wpe",
    "lm_head",
    "tok_emb",
    "transformer.h.",
)
_ATTN_NAME_MARKERS: tuple[str, ...] = (
    "attn",
    "attention",
    "q_proj",
    "qkv",
    ".query",
    ".key",
    ".value",
)


def infer_architecture(model: nn.Module) -> str:
    """Classify a model as ``vit`` / ``gpt`` / ``nlp`` / ``vision`` / ``generic``.

    Detection inspects module types (Conv*, Embedding, MultiheadAttention) and
    well-known parameter name markers (``patch_embed``, ``wte``, ``q_proj``...).
    """
    names = [name.lower() for name, _ in model.named_parameters()]
    modules = list(model.modules())

    has_conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in modules)
    has_embed = any(isinstance(m, nn.Embedding) for m in modules)
    has_mha = any(isinstance(m, nn.MultiheadAttention) for m in modules)
    has_attn = has_mha or any(marker in name for name in names for marker in _ATTN_NAME_MARKERS)
    is_vit = any(marker in name for name in names for marker in _VIT_NAME_MARKERS)
    is_gpt = any(marker in name for name in names for marker in _GPT_NAME_MARKERS)

    if is_vit and has_attn:
        return "vit"
    if has_embed and has_attn:
        return "gpt" if is_gpt else "nlp"
    if has_conv:
        return "vision"
    if has_embed:
        return "nlp"
    return "generic"


def build_auto_optimizer(
    cls: type[PsiLogic],
    model: nn.Module,
    lr: Optional[float] = None,
    total_steps: int = 0,
    **overrides: Any,
) -> PsiLogic:
    """Build a fully configured PsiLogic optimizer from a bare model.

    Used by ``PsiLogic.auto``. Architecture is inferred, the matching preset
    and parameter-group builder applied, and ``overrides`` are passed through
    to the optimizer constructor. Note that per-group keys set by the builder
    (e.g. embedding gamma) take precedence over constructor-level overrides
    for the parameters in those groups.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"PsiLogic.auto expects an nn.Module, got {type(model).__name__}. "
            "Pass parameters or param groups to the regular constructor instead."
        )

    arch = infer_architecture(model)
    lr = lr if lr is not None else _AUTO_LR[arch]

    if arch == "vit":
        params: Any = vit_param_groups(model, lr=lr)
        defaults = vision_defaults(total_steps)
    elif arch == "gpt":
        params = gpt_param_groups(model, lr=lr)
        defaults = gpt_scratch_defaults(total_steps)
    elif arch == "nlp":
        params = nlp_param_groups(model, lr=lr)
        defaults = nlp_defaults(total_steps)
    elif arch == "vision":
        params = model.parameters()
        defaults = vision_defaults(total_steps)
    else:
        params = model.parameters()
        defaults = {"gamma_T_max": total_steps}

    defaults.update(overrides)
    return cls(params, lr=lr, **defaults)


class PsiLogicNLP(PsiLogic):
    """PsiLogic preset for NLP encoder fine-tuning (BERT, RoBERTa, etc.).

    Accepts either a parameter iterable or a full ``nn.Module`` — in the
    latter case ``nlp_param_groups`` is applied automatically.
    """

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
        if isinstance(params, nn.Module):
            params = nlp_param_groups(params, lr=lr, weight_decay=kwargs.get("weight_decay", 1e-4))
        super().__init__(params, lr=lr, gamma_T_max=gamma_T_max, **kwargs)


class PsiLogicGPT(PsiLogic):
    """PsiLogic preset for language model training from scratch.

    Accepts either a parameter iterable or a full ``nn.Module`` — in the
    latter case ``gpt_param_groups`` is applied automatically (embeddings
    γ=0.005 with no quantum decay, blocks γ=0.02, LM head γ=0.01).
    """

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
        if isinstance(params, nn.Module):
            params = gpt_param_groups(params, lr=lr, weight_decay=kwargs["weight_decay"])
        super().__init__(params, lr=lr, gamma_T_max=gamma_T_max, **kwargs)


class PsiLogicViT(PsiLogic):
    """PsiLogic preset for Vision Transformer and CNN training.

    Accepts either a parameter iterable or a full ``nn.Module`` — in the
    latter case ``vit_param_groups`` is applied automatically (patch embed
    γ=0.005, attention γ=0.02, MLP γ=0.03, norm/bias without weight decay).
    ``lion_blocks=True`` runs Lion updates on transformer blocks while patch
    embeddings keep Adam.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        gamma_T_max: int = 0,
        lion_blocks: bool = False,
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
        if isinstance(params, nn.Module):
            params = vit_param_groups(
                params,
                lr=lr,
                weight_decay=kwargs.get("weight_decay", 1e-4),
                lion_blocks=lion_blocks,
            )
        super().__init__(params, lr=lr, gamma_T_max=gamma_T_max, **kwargs)


class PsiLogicWhisper(PsiLogic):
    """PsiLogic preset for speech model fine-tuning (Whisper, wav2vec, etc.).

    Accepts either a parameter iterable or a full ``nn.Module`` — in the
    latter case ``nlp_param_groups`` is applied (encoder-decoder speech models
    share the transformer naming conventions it targets).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-5,
        gamma_T_max: int = 0,
        **kwargs: Any,
    ) -> None:
        preset = whisper_defaults(gamma_T_max)
        preset.pop("gamma_T_max", None)
        preset.pop("use_foreach", None)
        for key, value in preset.items():
            kwargs.setdefault(key, value)
        if isinstance(params, nn.Module):
            params = nlp_param_groups(params, lr=lr, weight_decay=kwargs.get("weight_decay", 1e-2))
        super().__init__(params, lr=lr, gamma_T_max=gamma_T_max, **kwargs)
