"""Task-specific parameter group builders."""

from __future__ import annotations

from typing import Any, Sequence

import torch.nn as nn

_NO_DECAY_DEFAULTS: tuple[str, ...] = ("bias", "LayerNorm.weight", "layer_norm.weight")
_ATTN_PATTERNS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "c_attn",
    "c_proj",
    "attn.proj",
)


def nlp_param_groups(
    model: nn.Module,
    lr: float = 1e-3,
    *,
    embedding_gamma: float = 0.01,
    attention_gamma: float = 0.03,
    default_gamma: float = 0.03,
    no_decay_names: Sequence[str] = _NO_DECAY_DEFAULTS,
    weight_decay: float = 1e-4,
    **shared_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Split model parameters into groups with per-group gamma values.

    Embeddings receive minimal cancellation. Attention projections get moderate
    cancellation. Bias and LayerNorm parameters are excluded from weight decay.

    Example::

        groups = nlp_param_groups(model, lr=3e-4)
        optimizer = PsiLogic(groups, **nlp_defaults(total_steps))
    """
    for key in ("lr", "weight_decay", "gamma"):
        shared_kwargs.pop(key, None)

    no_decay_set = set(no_decay_names)
    embed_params: list[nn.Parameter] = []
    attn_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []
    decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(nd in name for nd in no_decay_set)
        if "embed" in name.lower() and "weight" in name:
            embed_params.append(param)
        elif any(pattern in name for pattern in _ATTN_PATTERNS):
            (nodecay_params if is_no_decay else attn_params).append(param)
        elif is_no_decay:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        dict(
            params=embed_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=embedding_gamma,
            **shared_kwargs,
        ),
        dict(
            params=attn_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=attention_gamma,
            **shared_kwargs,
        ),
        dict(
            params=nodecay_params,
            lr=lr,
            weight_decay=0.0,
            gamma=default_gamma,
            **shared_kwargs,
        ),
        dict(
            params=decay_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=default_gamma,
            **shared_kwargs,
        ),
    ]
    return [group for group in groups if group["params"]]
