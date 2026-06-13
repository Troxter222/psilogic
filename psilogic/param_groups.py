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

_VIT_PATCH_PATTERNS: tuple[str, ...] = (
    "patch_embed",
    "patch_embeddings",
    "conv_proj",
    "to_patch_embedding",
    "embeddings.patch",
)
_VIT_TOKEN_PATTERNS: tuple[str, ...] = (
    "cls_token",
    "class_token",
    "pos_embed",
    "position_embeddings",
    "dist_token",
    "mask_token",
    "register_tokens",
)
_VIT_ATTN_PATTERNS: tuple[str, ...] = (
    "attn",
    "attention",
    "qkv",
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    ".query",
    ".key",
    ".value",
)
_VIT_MLP_PATTERNS: tuple[str, ...] = (
    "mlp",
    "fc1",
    "fc2",
    "intermediate",
    "output.dense",
    "feed_forward",
    "ffn",
)

_GPT_EMBED_PATTERNS: tuple[str, ...] = (
    "wte",
    "wpe",
    "embed",
    "tok_emb",
    "pos_emb",
    "embedding",
)
_GPT_HEAD_PATTERNS: tuple[str, ...] = ("lm_head", "score", "classifier")


def _match(name: str, patterns: Sequence[str]) -> bool:
    return any(pattern in name for pattern in patterns)


def _check_coverage(model: nn.Module, groups: list[dict[str, Any]]) -> None:
    grouped = sum(len(group["params"]) for group in groups)
    total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    if grouped != total:
        raise ValueError(
            f"Parameter group split lost parameters: grouped {grouped} of {total}. "
            "This is a bug — please report it with your model architecture."
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
        elif _match(name, _ATTN_PATTERNS):
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
    groups = [group for group in groups if group["params"]]
    _check_coverage(model, groups)
    return groups


def vit_param_groups(
    model: nn.Module,
    lr: float = 1e-3,
    *,
    embed_gamma: float = 0.005,
    attention_gamma: float = 0.02,
    mlp_gamma: float = 0.03,
    default_gamma: float = 0.03,
    weight_decay: float = 1e-4,
    lion_blocks: bool = False,
    **shared_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Vision Transformer parameter split with per-group cancellation strength.

    Groups (in matching priority order):

    - **tokens** — ``cls_token`` / ``pos_embed`` / distillation tokens:
      γ=``embed_gamma``, no weight decay.
    - **norm/bias** — every 1-D parameter (LayerNorm, biases): no weight decay.
    - **patch embed** — convolutional/linear patch projection: γ=``embed_gamma``.
    - **attention** — qkv and output projections: γ=``attention_gamma``.
    - **MLP** — feed-forward blocks: γ=``mlp_gamma``.
    - **rest** — classifier head and anything unmatched: γ=``default_gamma``.

    ``lion_blocks=True`` runs Lion sign-momentum on attention/MLP/head groups
    while patch embeddings and tokens keep Adam updates — chaos damping stays
    active everywhere.

    Example::

        groups = vit_param_groups(model, lr=1e-3, lion_blocks=True)
        optimizer = PsiLogic(groups, **vision_defaults(total_steps))
    """
    for key in ("lr", "weight_decay", "gamma", "lion_mode"):
        shared_kwargs.pop(key, None)

    token_params: list[nn.Parameter] = []
    nodecay_params: list[nn.Parameter] = []
    patch_params: list[nn.Parameter] = []
    attn_params: list[nn.Parameter] = []
    mlp_params: list[nn.Parameter] = []
    rest_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if _match(lname, _VIT_TOKEN_PATTERNS):
            token_params.append(param)
        elif param.dim() <= 1:
            nodecay_params.append(param)
        elif _match(lname, _VIT_PATCH_PATTERNS):
            patch_params.append(param)
        elif _match(lname, _VIT_ATTN_PATTERNS):
            attn_params.append(param)
        elif _match(lname, _VIT_MLP_PATTERNS):
            mlp_params.append(param)
        else:
            rest_params.append(param)

    adam_kwargs = dict(shared_kwargs)
    block_kwargs = dict(shared_kwargs)
    if lion_blocks:
        adam_kwargs["lion_mode"] = False
        block_kwargs["lion_mode"] = True

    groups = [
        dict(params=token_params, lr=lr, weight_decay=0.0, gamma=embed_gamma, **adam_kwargs),
        dict(
            params=nodecay_params,
            lr=lr,
            weight_decay=0.0,
            gamma=default_gamma,
            **adam_kwargs,
        ),
        dict(
            params=patch_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=embed_gamma,
            **adam_kwargs,
        ),
        dict(
            params=attn_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=attention_gamma,
            **block_kwargs,
        ),
        dict(
            params=mlp_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=mlp_gamma,
            **block_kwargs,
        ),
        dict(
            params=rest_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=default_gamma,
            **block_kwargs,
        ),
    ]
    groups = [group for group in groups if group["params"]]
    _check_coverage(model, groups)
    return groups


def gpt_param_groups(
    model: nn.Module,
    lr: float = 3e-4,
    *,
    embedding_gamma: float = 0.005,
    block_gamma: float = 0.02,
    head_gamma: float = 0.01,
    default_gamma: float = 0.02,
    weight_decay: float = 0.1,
    **shared_kwargs: Any,
) -> list[dict[str, Any]]:
    """
    From-scratch LM parameter split tuned for chaos-sensitive embeddings.

    Groups (in matching priority order):

    - **norm/bias** — every 1-D parameter: no weight decay.
    - **embeddings** — ``wte`` / ``wpe`` / anything named ``embed``:
      γ=``embedding_gamma``, quantum decay forcibly disabled.
    - **LM head** — output projection: γ=``head_gamma``.
    - **blocks** — transformer body: γ=``block_gamma``.

    Weight-tied heads (``lm_head.weight is wte.weight``) are handled
    naturally: ``named_parameters()`` deduplicates tied tensors, so the tied
    weight lands in the embedding group exactly once.

    Example::

        groups = gpt_param_groups(model, lr=3e-4)
        optimizer = PsiLogic(groups, **gpt_scratch_defaults(total_steps))
    """
    for key in ("lr", "weight_decay", "gamma", "quantum_decay"):
        shared_kwargs.pop(key, None)

    nodecay_params: list[nn.Parameter] = []
    embed_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    block_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if param.dim() <= 1:
            nodecay_params.append(param)
        elif _match(lname, _GPT_EMBED_PATTERNS):
            embed_params.append(param)
        elif _match(lname, _GPT_HEAD_PATTERNS):
            head_params.append(param)
        else:
            block_params.append(param)

    groups = [
        dict(
            params=nodecay_params,
            lr=lr,
            weight_decay=0.0,
            gamma=default_gamma,
            **shared_kwargs,
        ),
        dict(
            params=embed_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=embedding_gamma,
            quantum_decay=0.0,
            **shared_kwargs,
        ),
        dict(
            params=head_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=head_gamma,
            **shared_kwargs,
        ),
        dict(
            params=block_params,
            lr=lr,
            weight_decay=weight_decay,
            gamma=block_gamma,
            **shared_kwargs,
        ),
    ]
    groups = [group for group in groups if group["params"]]
    _check_coverage(model, groups)
    return groups
