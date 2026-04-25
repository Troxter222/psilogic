"""
Monolithic MLOps benchmark: AdamW vs Lion vs PsiLogic v6
Arenas: BERT/SST-2 | ViT-Tiny/CIFAR-100 | GPT-2-scratch/Wikitext-2
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. SUPPRESS WARNINGS (before any other import)
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL CONFIG & API KEYS
# ─────────────────────────────────────────────────────────────────────────────
TG_TOKEN        = ""
TG_CHAT_ID      = ""
RUNPOD_API_KEY  = ""
RUNPOD_POD_ID   = ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. STDLIB & THIRD-PARTY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import csv
import io
import json
import math
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 3. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 4. LION OPTIMIZER (per Google DeepMind paper, arXiv 2302.06675)
# ─────────────────────────────────────────────────────────────────────────────
class Lion(Optimizer):
    """
    Lion: Evo-Learned Optimizer.
    Chen et al. (2023) — https://arxiv.org/abs/2302.06675

    update = sign(β1·m + (1−β1)·g)
    m      ← β2·m + (1−β2)·g
    p      ← p − lr · (update + wd · p)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]

                # Lion update rule
                update = (beta1 * m + (1.0 - beta1) * g).sign()
                p.add_(update, alpha=-lr)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Moment update
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 5. PSILOGIC v6.0 (full implementation)
# ─────────────────────────────────────────────────────────────────────────────
def nlp_param_groups(
    model,
    lr: float = 1e-3,
    *,
    embedding_gamma: float = 0.01,
    attention_gamma: float = 0.03,
    default_gamma: float = 0.03,
    no_decay_names: tuple = ("bias", "LayerNorm.weight", "layer_norm.weight"),
    weight_decay: float = 1e-4,
    **shared_kwargs,
):
    for k in ("lr", "weight_decay", "gamma"):
        shared_kwargs.pop(k, None)

    no_decay_set = set(no_decay_names)
    embed_params, attn_params, nodecay_params, decay_params = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(nd in name for nd in no_decay_set)
        if "embed" in name.lower() and "weight" in name:
            embed_params.append(param)
        elif any(t in name for t in ("q_proj", "k_proj", "v_proj", "out_proj",
                                     "c_attn", "c_proj", "attn.proj")):
            (nodecay_params if is_no_decay else attn_params).append(param)
        elif is_no_decay:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        dict(params=embed_params,   lr=lr, weight_decay=weight_decay, gamma=embedding_gamma, **shared_kwargs),
        dict(params=attn_params,    lr=lr, weight_decay=weight_decay, gamma=attention_gamma, **shared_kwargs),
        dict(params=nodecay_params, lr=lr, weight_decay=0.0,          gamma=default_gamma,   **shared_kwargs),
        dict(params=decay_params,   lr=lr, weight_decay=weight_decay, gamma=default_gamma,   **shared_kwargs),
    ]
    return [g for g in groups if g["params"]]


def nlp_defaults(total_steps: int = 0) -> dict:
    return dict(
        betas=(0.9, 0.999), weight_decay=1e-4, gamma=0.03, p_ext=1.0,
        quantum_decay=2e-4, eps=1e-8, grad_centralize=True,
        chaos_tau=0.40, chaos_warmup=-1, adaptive_tau=True,
        tau_scale=2.0, max_cancel=0.05, agc_clip=0.01,
        gamma_T_max=total_steps, use_foreach=True,
    )


def vision_defaults(total_steps: int = 0) -> dict:
    return dict(
        betas=(0.9, 0.999), weight_decay=1e-4, gamma=0.04, p_ext=1.0,
        quantum_decay=0.0, eps=1e-8, grad_centralize=True,
        chaos_tau=0.40, chaos_warmup=-1, adaptive_tau=True,
        tau_scale=2.5, max_cancel=0.04, agc_clip=0.02,
        gamma_T_max=total_steps, use_foreach=True,
    )


def gpt_scratch_defaults(total_steps: int = 0) -> dict:
    return dict(
        betas=(0.9, 0.999), weight_decay=1e-1, gamma=0.02, p_ext=1.0,
        quantum_decay=0.0, eps=1e-8, grad_centralize=True,
        chaos_tau=0.40, chaos_warmup=-1, adaptive_tau=True,
        tau_scale=3.0, max_cancel=0.03, agc_clip=0.01,
        gamma_T_max=total_steps, use_foreach=True,
    )


class PsiLogic(Optimizer):
    r"""ΨLogic v6.0 — see module docstring for full spec."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        weight_decay: float = 1e-4,
        gamma: float = 0.05,
        p_ext: float = 1.0,
        quantum_decay: float = 0.0,
        eps: float = 1e-8,
        grad_centralize: bool = True,
        chaos_tau: float = 0.5,
        chaos_warmup: int = -1,
        adaptive_tau: bool = True,
        tau_scale: float = 2.0,
        max_cancel: float = 0.05,
        agc_clip: float = 0.02,
        gamma_T_max: int = 0,
        use_foreach: bool = True,
        lion_mode: bool = False,
    ):
        assert lr >= 0
        assert weight_decay >= 0
        assert gamma >= 0
        assert quantum_decay >= 0
        assert 0 <= betas[0] < 1
        assert 0 <= betas[1] < 1
        assert agc_clip >= 0
        assert 0 < max_cancel <= 1

        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay,
            gamma=gamma, p_ext=p_ext, quantum_decay=quantum_decay,
            eps=eps, grad_centralize=grad_centralize,
            chaos_tau=chaos_tau, chaos_warmup=chaos_warmup,
            adaptive_tau=adaptive_tau, tau_scale=tau_scale,
            max_cancel=max_cancel, agc_clip=agc_clip,
            gamma_T_max=gamma_T_max, use_foreach=use_foreach,
            lion_mode=lion_mode,
        )
        super().__init__(params, defaults)

    def _step_scalar(self, group: dict) -> None:
        lr           = group["lr"]
        beta1, beta2 = group["betas"]
        wd           = group["weight_decay"]
        gamma        = group["gamma"]
        p_ext        = group["p_ext"]
        qd           = group["quantum_decay"]
        eps          = group["eps"]
        gc           = group["grad_centralize"]
        chaos_tau    = group["chaos_tau"]
        warmup_cfg   = group["chaos_warmup"]
        adapt_tau    = group["adaptive_tau"]
        tau_scale    = group["tau_scale"]
        max_cancel   = group["max_cancel"]
        agc          = group["agc_clip"]
        T_max        = group["gamma_T_max"]
        lion         = group["lion_mode"]

        auto_warmup = max(50, T_max // 10) if T_max > 0 else 200
        warmup = warmup_cfg if warmup_cfg >= 0 else auto_warmup

        for p in group["params"]:
            if p.grad is None:
                continue

            raw_g = p.grad
            dev   = p.device

            g = raw_g
            if agc > 0.0:
                p_norm   = p.norm()
                g_norm   = g.norm()
                max_norm = agc * p_norm.clamp(min=1e-3)
                clip_cf  = (max_norm / g_norm.clamp(min=1e-6)).clamp(max=1.0)
                g        = g * clip_cf
                raw_g    = g

            if gc and g.dim() > 1:
                g = g - g.mean(dim=tuple(range(1, g.dim())), keepdim=True)

            st = self.state[p]
            if not st:
                st["t"]    = 0
                st["m"]    = torch.zeros_like(p)
                st["v"]    = torch.zeros_like(p)
                st["fast"] = torch.zeros(1, device=dev, dtype=p.dtype)
                st["slow"] = torch.zeros(1, device=dev, dtype=p.dtype)

            st["t"] += 1
            t = st["t"]

            st["m"].mul_(beta1).add_(g, alpha=1.0 - beta1)
            if not lion:
                st["v"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

            gn = g.norm() / math.sqrt(max(g.numel(), 1))
            if t == 1:
                st["fast"].fill_(gn.item())
                st["slow"].fill_(gn.item())
            else:
                st["fast"].mul_(0.9).add_(gn, alpha=0.1)
                st["slow"].mul_(0.99).add_(gn, alpha=0.01)

            slow_v = st["slow"].item()
            fast_v = st["fast"].item()

            if T_max > 0:
                cos_w  = 0.5 * (1.0 + math.cos(math.pi * min(t / T_max, 1.0)))
                g_eff  = gamma * cos_w
                qd_eff = qd * cos_w
            else:
                g_eff  = gamma
                qd_eff = qd

            chaos_active  = (t > warmup)
            chaos_contrib = 0.0
            qd_contrib    = 0.0

            if chaos_active and g_eff > 0:
                if adapt_tau:
                    spike = (fast_v > tau_scale * slow_v + eps)
                else:
                    spike = (slow_v >= chaos_tau)

                if spike:
                    ratio         = fast_v / (slow_v + eps)
                    chaos         = math.tanh(slow_v) * (
                        1.0 + 0.5 * math.tanh(max(ratio - 1.0, 0.0)))
                    raw_cc        = chaos * lr * g_eff * p_ext
                    chaos_contrib = min(raw_cc, max_cancel)
                else:
                    if qd_eff > 0:
                        qd_contrib = None  # handle below

            total_scalar_decay = lr * wd + chaos_contrib
            if total_scalar_decay > 0:
                p.mul_(1.0 - total_scalar_decay)

            if qd_contrib is None and qd_eff > 0:
                p.mul_(1.0 - lr * qd_eff * torch.tanh(raw_g.abs()))

            if lion:
                update = (beta1 * st["m"] + (1.0 - beta1) * g).sign()
                p.add_(update, alpha=-lr)
            else:
                bc1       = 1.0 - beta1 ** t
                bc2       = math.sqrt(1.0 - beta2 ** t)
                step_size = lr * bc2 / bc1
                denom     = st["v"].sqrt().add_(eps)
                p.addcdiv_(st["m"], denom, value=-step_size)

    def _step_foreach(self, group: dict) -> None:
        lr           = group["lr"]
        beta1, beta2 = group["betas"]
        wd           = group["weight_decay"]
        gamma        = group["gamma"]
        p_ext        = group["p_ext"]
        qd           = group["quantum_decay"]
        eps          = group["eps"]
        gc           = group["grad_centralize"]
        warmup_cfg   = group["chaos_warmup"]
        adapt_tau    = group["adaptive_tau"]
        tau_scale    = group["tau_scale"]
        max_cancel   = group["max_cancel"]
        agc          = group["agc_clip"]
        T_max        = group["gamma_T_max"]
        lion         = group["lion_mode"]

        auto_warmup = max(50, T_max // 10) if T_max > 0 else 200
        warmup = warmup_cfg if warmup_cfg >= 0 else auto_warmup

        params_with_grad = [p for p in group["params"] if p.grad is not None]
        if not params_with_grad:
            return

        grads = [p.grad for p in params_with_grad]

        if agc > 0.0:
            p_norms = torch._foreach_norm(params_with_grad)
            g_norms = torch._foreach_norm(grads)
            clipped_grads = []
            for g, pn, gn in zip(grads, p_norms, g_norms):
                max_n = agc * pn.clamp(min=1e-3)
                cf    = (max_n / gn.clamp(min=1e-6)).clamp(max=1.0)
                clipped_grads.append(g * cf)
            grads = clipped_grads

        raw_grads = [g.clone() for g in grads]

        if gc:
            for i, g in enumerate(grads):
                if g.dim() > 1:
                    grads[i] = g - g.mean(dim=tuple(range(1, g.dim())), keepdim=True)

        ms, vs, fasts, slows, ts = [], [], [], [], []
        for p, g in zip(params_with_grad, grads):
            st = self.state[p]
            if not st:
                st["t"]    = 0
                st["m"]    = torch.zeros_like(p)
                st["v"]    = torch.zeros_like(p)
                st["fast"] = torch.zeros(1, device=p.device, dtype=p.dtype)
                st["slow"] = torch.zeros(1, device=p.device, dtype=p.dtype)
            st["t"] += 1
            ms.append(st["m"])
            vs.append(st["v"])
            fasts.append(st["fast"])
            slows.append(st["slow"])
            ts.append(st["t"])

        t = ts[0]

        torch._foreach_mul_(ms, beta1)
        torch._foreach_add_(ms, grads, alpha=1.0 - beta1)
        if not lion:
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, grads, grads, value=1.0 - beta2)

        g_norms = torch._foreach_norm(grads)
        for i, (gn, fast, slow) in enumerate(zip(g_norms, fasts, slows)):
            numel = grads[i].numel()
            gn_s  = gn / math.sqrt(max(numel, 1))
            if t == 1:
                fast.fill_(gn_s.item())
                slow.fill_(gn_s.item())
            else:
                fast.mul_(0.9).add_(gn_s, alpha=0.1)
                slow.mul_(0.99).add_(gn_s, alpha=0.01)

        if T_max > 0:
            cos_w  = 0.5 * (1.0 + math.cos(math.pi * min(t / T_max, 1.0)))
            g_eff  = gamma * cos_w
            qd_eff = qd * cos_w
        else:
            g_eff  = gamma
            qd_eff = qd

        chaos_active = (t > warmup)

        if chaos_active and g_eff > 0:
            for i, (p, raw_g) in enumerate(zip(params_with_grad, raw_grads)):
                slow_v = slows[i].item()
                fast_v = fasts[i].item()

                if adapt_tau:
                    spike = fast_v > tau_scale * slow_v + eps
                else:
                    spike = slow_v >= group["chaos_tau"]

                if spike:
                    ratio         = fast_v / (slow_v + eps)
                    chaos         = math.tanh(slow_v) * (
                        1.0 + 0.5 * math.tanh(max(ratio - 1.0, 0.0)))
                    chaos_contrib = min(chaos * lr * g_eff * p_ext, max_cancel)
                    total_decay   = lr * wd + chaos_contrib
                    p.mul_(1.0 - total_decay)
                else:
                    if wd > 0:
                        p.mul_(1.0 - lr * wd)
                    if qd_eff > 0:
                        p.mul_(1.0 - lr * qd_eff * torch.tanh(raw_g.abs()))
        else:
            if wd > 0:
                torch._foreach_mul_(params_with_grad, 1.0 - lr * wd)

        if lion:
            for p, m, g in zip(params_with_grad, ms, grads):
                update = (beta1 * m + (1.0 - beta1) * g).sign()
                p.add_(update, alpha=-lr)
        else:
            bc1       = 1.0 - beta1 ** t
            bc2       = math.sqrt(1.0 - beta2 ** t)
            step_size = lr * bc2 / bc1
            denoms    = torch._foreach_sqrt(vs)
            torch._foreach_add_(denoms, eps)
            torch._foreach_addcdiv_(params_with_grad, ms, denoms, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_fe = (
                group["use_foreach"]
                and any(p.is_cuda for p in group["params"] if p.grad is not None)
            )
            if use_fe:
                self._step_foreach(group)
            else:
                self._step_scalar(group)

        return loss


class PsiLogicNLP(PsiLogic):
    def __init__(self, params, lr=1e-3, gamma_T_max=0, **kwargs):
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
    def __init__(self, params, lr=3e-4, gamma_T_max=0, **kwargs):
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
    def __init__(self, params, lr=1e-3, gamma_T_max=0, **kwargs):
        kwargs.setdefault("gamma", 0.04)
        kwargs.setdefault("chaos_tau", 0.40)
        kwargs.setdefault("chaos_warmup", -1)
        kwargs.setdefault("quantum_decay", 0.0)
        kwargs.setdefault("agc_clip", 0.02)
        kwargs.setdefault("adaptive_tau", True)
        kwargs.setdefault("tau_scale", 2.5)
        kwargs.setdefault("max_cancel", 0.04)
        super().__init__(params, lr=lr, gamma_T_max=gamma_T_max, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TELEGRAM UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
_TG_BASE = f"https://api.telegram.org/bot{TG_TOKEN}"


def tg_send(text: str, parse_mode: str = "HTML") -> None:
    """Send a text message to TG. Silently ignore network failures."""
    try:
        requests.post(
            f"{_TG_BASE}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": parse_mode},
            timeout=15,
        )
    except Exception:
        pass


def tg_send_photo(fig: plt.Figure, caption: str = "") -> None:
    """Serialize a matplotlib figure and send it as a photo."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        requests.post(
            f"{_TG_BASE}/sendPhoto",
            data={"chat_id": TG_CHAT_ID, "caption": caption},
            files={"photo": ("chart.png", buf, "image/png")},
            timeout=30,
        )
    except Exception:
        pass


def tg_send_file(path: str, caption: str = "") -> None:
    """Send a file (csv / png) to TG."""
    try:
        with open(path, "rb") as f:
            requests.post(
                f"{_TG_BASE}/sendDocument",
                data={"chat_id": TG_CHAT_ID, "caption": caption},
                files={"document": f},
                timeout=30,
            )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 7. GPU INFO
# ─────────────────────────────────────────────────────────────────────────────
def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "⚠️ CUDA not available — running on CPU"
    name   = torch.cuda.get_device_name(0)
    total  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"🖥 GPU: <b>{name}</b>  |  VRAM: <b>{total:.1f} GB</b>"


# ─────────────────────────────────────────────────────────────────────────────
# 8. PARAM-GROUP HELPERS FOR AdamW / Lion (честный weight-decay split)
# ─────────────────────────────────────────────────────────────────────────────
_NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight",
             "BatchNorm.weight", "BatchNorm.bias",
             "norm.weight", "norm.bias")


def make_param_groups(model, lr: float, wd: float):
    """Two-group split: decay / no_decay for AdamW and Lion."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in _NO_DECAY) or param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "lr": lr, "weight_decay": wd},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 9. RUNPOD SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────
def runpod_shutdown() -> None:
    """Terminate the RunPod instance via GraphQL API."""
    url     = "https://api.runpod.io/graphql"
    query   = """
    mutation stopPod($podId: String!) {
        podStop(input: { podId: $podId }) { id }
    }
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }
    payload = {"query": query, "variables": {"podId": RUNPOD_POD_ID}}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=20)
        print(f"[RunPod] Shutdown response: {r.status_code}")
    except Exception as e:
        print(f"[RunPod] Shutdown failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. RESULTS CONTAINER
# ─────────────────────────────────────────────────────────────────────────────
# results[arena][optimizer] = list of (metric_value, train_losses, val_losses)
results: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))


# ─────────────────────────────────────────────────────────────────────────────
# 11. ARENA 1 — BERT fine-tuning on SST-2
# ─────────────────────────────────────────────────────────────────────────────
def run_arena1() -> None:
    print("\n" + "═" * 60)
    print("  ARENA 1 — BERT/SST-2  (NLP Fine-tuning)")
    print("═" * 60)

    from datasets import load_dataset
    from transformers import BertForSequenceClassification, BertTokenizerFast

    ARENA      = "Arena1_BERT_SST2"
    EPOCHS     = 3
    SEEDS      = [42, 123, 777]
    LR         = 2e-5
    WD         = 1e-4
    BATCH      = 32
    MAX_LEN    = 128
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data once
    raw = load_dataset("glue", "sst2")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True,
                         padding="max_length", max_length=MAX_LEN)

    encoded = raw.map(tokenize, batched=True, batch_size=1000)
    encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_ds  = encoded["train"]
    val_ds    = encoded["validation"]
    train_ldr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

    total_steps = EPOCHS * len(train_ldr)

    OPT_CONFIGS = {
        "AdamW":     lambda m: AdamW(make_param_groups(m, LR, WD), lr=LR, weight_decay=WD),
        "Lion":      lambda m: Lion(make_param_groups(m, LR * 0.1, WD), lr=LR * 0.1, weight_decay=WD),
        "PsiLogic":  lambda m: PsiLogicNLP(
            nlp_param_groups(m, lr=LR, weight_decay=WD, **{k: v for k, v in nlp_defaults(total_steps).items()
                                                            if k not in ("weight_decay",)}),
            lr=LR, gamma_T_max=total_steps,
        ),
    }

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for opt_name, opt_factory in OPT_CONFIGS.items():
        seed_train_losses = []
        seed_val_losses   = []
        seed_accs         = []

        for seed in SEEDS:
            seed_everything(seed)
            model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            ).to(DEVICE)
            optimizer = opt_factory(model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=1e-7
            )

            ep_train_losses, ep_val_losses = [], []

            for epoch in range(1, EPOCHS + 1):
                # ── TRAIN ──
                model.train()
                t0     = time.perf_counter()
                t_loss = 0.0
                n_steps = 0

                pbar = tqdm(train_ldr, desc=f"[A1|{opt_name}|s{seed}|e{epoch}]",
                            leave=False, ncols=90)
                for batch in pbar:
                    input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                    labels         = batch["label"].to(DEVICE, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    if scaler:
                        with torch.amp.autocast("cuda"):
                            out  = model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
                            loss = out.loss
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        out  = model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=labels)
                        loss = out.loss
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    scheduler.step()
                    t_loss  += loss.item()
                    n_steps += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                elapsed   = time.perf_counter() - t0
                its       = n_steps / max(elapsed, 1e-6)
                mean_tl   = t_loss / n_steps
                ep_train_losses.append(mean_tl)

                # ── VALIDATE ──
                model.eval()
                correct, total, v_loss_sum, v_steps = 0, 0, 0.0, 0
                with torch.no_grad():
                    for batch in val_ldr:
                        input_ids      = batch["input_ids"].to(DEVICE, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
                        labels         = batch["label"].to(DEVICE, non_blocking=True)
                        with torch.amp.autocast("cuda") if scaler else torch.no_grad():
                            out = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                        v_loss_sum += out.loss.item()
                        v_steps    += 1
                        preds       = out.logits.argmax(-1)
                        correct    += (preds == labels).sum().item()
                        total      += labels.size(0)

                val_acc  = correct / total
                mean_vl  = v_loss_sum / v_steps
                ep_val_losses.append(mean_vl)

                msg = (
                    f"📊 <b>[{ARENA}]</b> <code>{opt_name}</code> "
                    f"seed={seed} epoch={epoch}/{EPOCHS}\n"
                    f"  Train Loss: {mean_tl:.4f}  |  Val Loss: {mean_vl:.4f}\n"
                    f"  Val Acc: <b>{val_acc:.4f}</b>  |  Speed: {its:.1f} it/s"
                )
                tg_send(msg)
                print(f"  ↳ {opt_name} | s{seed} | e{epoch} | "
                      f"TLoss={mean_tl:.4f} VLoss={mean_vl:.4f} "
                      f"Acc={val_acc:.4f} {its:.1f}it/s")

            seed_train_losses.append(ep_train_losses)
            seed_val_losses.append(ep_val_losses)
            seed_accs.append(val_acc)

        results[ARENA][opt_name] = {
            "metric":       seed_accs,
            "train_losses": seed_train_losses,
            "val_losses":   seed_val_losses,
        }

    # ── Post-arena chart ──────────────────────────────────────────────────────
    fig = _plot_learning_curves(
        results[ARENA], ARENA,
        x_label="Epoch", metric_name="Val Accuracy"
    )
    tg_send_photo(fig, caption=f"📈 Learning curves — {ARENA}")
    plt.close(fig)
    print(f"[{ARENA}] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# 12. ARENA 2 — ViT-Tiny on CIFAR-100
# ─────────────────────────────────────────────────────────────────────────────
def run_arena2() -> None:
    print("\n" + "═" * 60)
    print("  ARENA 2 — ViT-Tiny/CIFAR-100  (Vision Transformer)")
    print("═" * 60)

    import timm
    import torchvision
    import torchvision.transforms as T

    ARENA  = "Arena2_ViT_CIFAR100"
    EPOCHS = 15
    SEEDS  = [42, 123, 777]
    LR     = 1e-3
    WD     = 1e-4
    BATCH  = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds  = torchvision.datasets.CIFAR100("./data", train=True,  download=True, transform=train_tf)
    val_ds    = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=val_tf)
    train_ldr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    total_steps = EPOCHS * len(train_ldr)

    OPT_CONFIGS = {
        "AdamW":    lambda m: AdamW(make_param_groups(m, LR, WD), lr=LR, weight_decay=WD),
        "Lion":     lambda m: Lion(make_param_groups(m, LR * 0.1, WD), lr=LR * 0.1, weight_decay=WD),
        "PsiLogic": lambda m: PsiLogicViT(
            make_param_groups(m, LR, WD), lr=LR,
            gamma_T_max=total_steps,
            **{k: v for k, v in vision_defaults(total_steps).items()
               if k not in ("lr", "weight_decay", "gamma_T_max")},
        ),
    }

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for opt_name, opt_factory in OPT_CONFIGS.items():
        seed_train_losses = []
        seed_val_losses   = []
        seed_accs         = []

        for seed in SEEDS:
            seed_everything(seed)
            model = timm.create_model(
                "vit_tiny_patch16_224", pretrained=False, num_classes=100
            ).to(DEVICE)
            optimizer = opt_factory(model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=1e-7
            )

            ep_train_losses, ep_val_losses = [], []

            for epoch in range(1, EPOCHS + 1):
                model.train()
                t0 = time.perf_counter()
                t_loss, n_steps = 0.0, 0

                pbar = tqdm(train_ldr,
                            desc=f"[A2|{opt_name}|s{seed}|e{epoch}/{EPOCHS}]",
                            leave=False, ncols=90)
                for imgs, labels in pbar:
                    imgs   = imgs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    if scaler:
                        with torch.amp.autocast("cuda"):
                            loss = F.cross_entropy(model(imgs), labels)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss = F.cross_entropy(model(imgs), labels)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    scheduler.step()
                    t_loss  += loss.item()
                    n_steps += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                elapsed = time.perf_counter() - t0
                its     = n_steps / max(elapsed, 1e-6)
                mean_tl = t_loss / n_steps
                ep_train_losses.append(mean_tl)

                model.eval()
                correct, total_n, v_loss_sum, v_steps = 0, 0, 0.0, 0
                with torch.no_grad():
                    for imgs, labels in val_ldr:
                        imgs   = imgs.to(DEVICE, non_blocking=True)
                        labels = labels.to(DEVICE, non_blocking=True)
                        with torch.amp.autocast("cuda") if scaler else torch.no_grad():
                            logits = model(imgs)
                        v_loss_sum += F.cross_entropy(logits, labels).item()
                        v_steps    += 1
                        correct    += (logits.argmax(-1) == labels).sum().item()
                        total_n    += labels.size(0)

                val_acc = correct / total_n
                mean_vl = v_loss_sum / v_steps
                ep_val_losses.append(mean_vl)

                msg = (
                    f"📊 <b>[{ARENA}]</b> <code>{opt_name}</code> "
                    f"seed={seed} epoch={epoch}/{EPOCHS}\n"
                    f"  Train Loss: {mean_tl:.4f}  |  Val Loss: {mean_vl:.4f}\n"
                    f"  Top-1 Acc: <b>{val_acc:.4f}</b>  |  Speed: {its:.1f} it/s"
                )
                tg_send(msg)
                print(f"  ↳ {opt_name} | s{seed} | e{epoch:02d} | "
                      f"TLoss={mean_tl:.4f} VLoss={mean_vl:.4f} "
                      f"Acc={val_acc:.4f} {its:.1f}it/s")

            seed_train_losses.append(ep_train_losses)
            seed_val_losses.append(ep_val_losses)
            seed_accs.append(val_acc)

        results[ARENA][opt_name] = {
            "metric":       seed_accs,
            "train_losses": seed_train_losses,
            "val_losses":   seed_val_losses,
        }

    fig = _plot_learning_curves(
        results[ARENA], ARENA, x_label="Epoch", metric_name="Top-1 Acc"
    )
    tg_send_photo(fig, caption=f"📈 Learning curves — {ARENA}")
    plt.close(fig)
    print(f"[{ARENA}] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# 13. ARENA 3 — GPT-2 from scratch on Wikitext-2
# ─────────────────────────────────────────────────────────────────────────────
def run_arena3() -> None:
    print("\n" + "═" * 60)
    print("  ARENA 3 — GPT-2 scratch / Wikitext-2  (LM Pre-training)")
    print("═" * 60)

    from datasets import load_dataset
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    ARENA       = "Arena3_GPT2_Wikitext2"
    TOTAL_STEPS = 3000
    SEEDS       = [42, 123]
    LR          = 3e-4
    WD          = 0.1
    BATCH       = 8
    SEQ_LEN     = 256
    LOG_EVERY   = 200
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        concat = {k: sum(examples[k], []) for k in examples}
        total  = len(concat["input_ids"])
        total  = (total // SEQ_LEN) * SEQ_LEN
        result = {k: [concat[k][i: i + SEQ_LEN]
                      for i in range(0, total, SEQ_LEN)]
                  for k in concat}
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized   = raw.map(tokenize_fn,  batched=True, remove_columns=["text"])
    lm_dataset  = tokenized.map(group_texts, batched=True)
    lm_dataset.set_format("torch")

    train_ds  = lm_dataset["train"]
    val_ds    = lm_dataset["validation"]
    train_ldr = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    val_ldr   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)

    # Small GPT-2 config (from scratch, not pretrained)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=SEQ_LEN,
        n_embd=256,
        n_layer=6,
        n_head=8,
    )

    def _eval_ppl(model) -> Tuple[float, float]:
        model.eval()
        v_loss_sum, v_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_ldr:
                ids    = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                with torch.amp.autocast("cuda") if DEVICE.type == "cuda" else torch.no_grad():
                    out = model(input_ids=ids, labels=labels)
                v_loss_sum += out.loss.item()
                v_steps    += 1
        mean_vl = v_loss_sum / max(v_steps, 1)
        return mean_vl, math.exp(min(mean_vl, 20))

    OPT_CONFIGS = {
        "AdamW":    lambda m: AdamW(make_param_groups(m, LR, WD), lr=LR, weight_decay=WD),
        "Lion":     lambda m: Lion(make_param_groups(m, LR * 0.1, WD), lr=LR * 0.1, weight_decay=WD),
        "PsiLogic": lambda m: PsiLogicGPT(
            make_param_groups(m, LR, WD), lr=LR,
            gamma_T_max=TOTAL_STEPS,
            **{k: v for k, v in gpt_scratch_defaults(TOTAL_STEPS).items()
               if k not in ("lr", "weight_decay", "gamma_T_max")},
        ),
    }

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for opt_name, opt_factory in OPT_CONFIGS.items():
        seed_ppls         = []
        seed_train_losses = []
        seed_val_losses   = []

        for seed in SEEDS:
            seed_everything(seed)
            model = GPT2LMHeadModel(config).to(DEVICE)
            optimizer = opt_factory(model)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=TOTAL_STEPS, eta_min=1e-7
            )

            step          = 0
            ep_t_losses   = []
            ep_v_losses   = []
            train_iter    = iter(train_ldr)
            checkpoint_t  = []
            checkpoint_v  = []
            t0            = time.perf_counter()

            pbar = tqdm(range(1, TOTAL_STEPS + 1),
                        desc=f"[A3|{opt_name}|s{seed}]",
                        ncols=90, leave=False)

            for step in pbar:
                model.train()
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_ldr)
                    batch      = next(train_iter)

                ids    = batch["input_ids"].to(DEVICE, non_blocking=True)
                labels = batch["labels"].to(DEVICE,    non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        out  = model(input_ids=ids, labels=labels)
                        loss = out.loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out  = model(input_ids=ids, labels=labels)
                    loss = out.loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                ep_t_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if step % LOG_EVERY == 0 or step == TOTAL_STEPS:
                    elapsed = time.perf_counter() - t0
                    its     = LOG_EVERY / max(elapsed, 1e-6)
                    t0      = time.perf_counter()
                    mean_tl = np.mean(ep_t_losses[-LOG_EVERY:])
                    vl, ppl = _eval_ppl(model)
                    ep_v_losses.append(vl)
                    checkpoint_t.append(mean_tl)
                    checkpoint_v.append(vl)

                    msg = (
                        f"🔄 <b>[{ARENA}]</b> <code>{opt_name}</code> "
                        f"seed={seed} step={step}/{TOTAL_STEPS}\n"
                        f"  Train Loss: {mean_tl:.4f}  |  Val Loss: {vl:.4f}\n"
                        f"  PPL: <b>{ppl:.2f}</b>  |  Speed: {its:.1f} it/s"
                    )
                    tg_send(msg)
                    print(f"  ↳ {opt_name} | s{seed} | step={step:4d} | "
                          f"TLoss={mean_tl:.4f} VLoss={vl:.4f} "
                          f"PPL={ppl:.2f} {its:.1f}it/s")

            _, final_ppl = _eval_ppl(model)
            seed_ppls.append(final_ppl)
            seed_train_losses.append(checkpoint_t)
            seed_val_losses.append(checkpoint_v)

        results[ARENA][opt_name] = {
            "metric":       seed_ppls,         # lower = better
            "train_losses": seed_train_losses,
            "val_losses":   seed_val_losses,
        }

    fig = _plot_learning_curves(
        results[ARENA], ARENA,
        x_label=f"Checkpoint (×{LOG_EVERY} steps)",
        metric_name="Val PPL"
    )
    tg_send_photo(fig, caption=f"📈 Learning curves — {ARENA}")
    plt.close(fig)
    print(f"[{ARENA}] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# 14. PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_COLORS = {"AdamW": "#4C72B0", "Lion": "#DD8452", "PsiLogic": "#55A868"}
_STYLES = {"AdamW": "-",       "Lion": "--",       "PsiLogic": "-."}


def _plot_learning_curves(
    arena_results: dict,
    arena_name: str,
    x_label: str = "Epoch",
    metric_name: str = "Metric",
) -> plt.Figure:
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(arena_name, fontsize=14, fontweight="bold")

    for opt, data in arena_results.items():
        color = _COLORS.get(opt, "gray")
        ls    = _STYLES.get(opt, "-")
        t_mat = np.array(data["train_losses"])   # (n_seeds, n_epochs/ckpts)
        v_mat = np.array(data["val_losses"])

        t_mean = t_mat.mean(0);  t_std = t_mat.std(0)
        v_mean = v_mat.mean(0);  v_std = v_mat.std(0)
        xs     = np.arange(1, len(t_mean) + 1)

        ax_t.plot(xs, t_mean, label=opt, color=color, ls=ls, lw=2)
        ax_t.fill_between(xs, t_mean - t_std, t_mean + t_std, alpha=0.15, color=color)

        ax_v.plot(xs, v_mean, label=opt, color=color, ls=ls, lw=2)
        ax_v.fill_between(xs, v_mean - v_std, v_mean + v_std, alpha=0.15, color=color)

    for ax, title in [(ax_t, "Train Loss"), (ax_v, "Val Loss")]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Loss")
        ax.legend()

    fig.tight_layout()
    return fig


def _plot_summary_bar(all_results: dict) -> plt.Figure:
    plt.style.use("seaborn-v0_8-darkgrid")

    arenas    = list(all_results.keys())
    opt_names = ["AdamW", "Lion", "PsiLogic"]
    n_arenas  = len(arenas)

    fig, axes = plt.subplots(1, n_arenas, figsize=(5 * n_arenas, 5))
    if n_arenas == 1:
        axes = [axes]

    METRIC_LABELS = {
        "Arena1_BERT_SST2":       "Val Accuracy ↑",
        "Arena2_ViT_CIFAR100":    "Top-1 Acc ↑",
        "Arena3_GPT2_Wikitext2":  "Val PPL ↓",
    }

    for ax, arena in zip(axes, arenas):
        x      = np.arange(len(opt_names))
        means  = []
        stds   = []
        for opt in opt_names:
            vals = all_results[arena].get(opt, {}).get("metric", [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        bars = ax.bar(
            x, means, yerr=stds, capsize=5,
            color=[_COLORS.get(o, "gray") for o in opt_names],
            edgecolor="black", linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(opt_names, rotation=15)
        ax.set_title(arena.replace("_", "\n"), fontsize=10, fontweight="bold")
        ax.set_ylabel(METRIC_LABELS.get(arena, "Metric"))

        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.002,
                f"{mean:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle("Benchmark Summary — Mean ± Std across seeds",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 15. CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(all_results: dict, path: str = "results.csv") -> None:
    rows = []
    for arena, opt_data in all_results.items():
        for opt, data in opt_data.items():
            for i, val in enumerate(data.get("metric", [])):
                rows.append({
                    "arena":     arena,
                    "optimizer": opt,
                    "seed_idx":  i,
                    "metric":    val,
                })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arena", "optimizer", "seed_idx", "metric"])
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 16. FINAL SUMMARY MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def build_summary_message(all_results: dict) -> str:
    lines = ["🏆 <b>Benchmark Complete — Final Summary</b>\n"]
    HIGHER_IS_BETTER = {"Arena1_BERT_SST2", "Arena2_ViT_CIFAR100"}

    for arena, opt_data in all_results.items():
        lines.append(f"<b>📌 {arena}</b>")
        rows = []
        for opt in ["AdamW", "Lion", "PsiLogic"]:
            vals = opt_data.get(opt, {}).get("metric", [float("nan")])
            m    = np.mean(vals)
            s    = np.std(vals)
            rows.append((opt, m, s))

        # sort: higher better for acc, lower better for PPL
        rev = arena in HIGHER_IS_BETTER
        rows.sort(key=lambda r: r[1], reverse=rev)
        medal = ["🥇", "🥈", "🥉"]
        for rank, (opt, m, s) in enumerate(rows):
            lines.append(f"  {medal[rank]} <code>{opt:<10}</code> {m:.4f} ± {s:.4f}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 17. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    tg_send(
        f"🟢 <b>Benchmark started!</b>\n"
        f"{gpu_info()}\n"
        f"Arenas: BERT/SST-2 | ViT-Tiny/CIFAR-100 | GPT-2-scratch/Wikitext-2\n"
        f"Optimizers: AdamW vs Lion vs PsiLogic v6"
    )
    print("=" * 60)
    print("  ULTIMATE OPTIMIZER BENCHMARK  v1.0")
    print(f"  Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    run_arena1()
    run_arena2()
    run_arena3()

    # ── CSV ──────────────────────────────────────────────────────────────────
    save_csv(results, "results.csv")
    tg_send_file("results.csv", caption="📁 Raw results (all seeds)")

    # ── Summary bar chart ────────────────────────────────────────────────────
    fig_bar = _plot_summary_bar(results)
    fig_bar.savefig("summary_bar_chart.png", dpi=150, bbox_inches="tight")
    tg_send_file("summary_bar_chart.png", caption="📊 Summary bar chart")
    plt.close(fig_bar)

    # ── Combined learning curves figure ─────────────────────────────────────
    plt.style.use("seaborn-v0_8-darkgrid")
    arenas = list(results.keys())
    fig_lc, axes = plt.subplots(len(arenas), 2,
                                figsize=(13, 5 * len(arenas)))
    if len(arenas) == 1:
        axes = [axes]

    X_LABELS = {
        "Arena1_BERT_SST2":      "Epoch",
        "Arena2_ViT_CIFAR100":   "Epoch",
        "Arena3_GPT2_Wikitext2": "Checkpoint (×200 steps)",
    }
    for row, arena in enumerate(arenas):
        ax_t, ax_v = axes[row]
        for opt, data in results[arena].items():
            color = _COLORS.get(opt, "gray")
            ls    = _STYLES.get(opt, "-")
            t_mat = np.array(data["train_losses"])
            v_mat = np.array(data["val_losses"])
            t_mean, t_std = t_mat.mean(0), t_mat.std(0)
            v_mean, v_std = v_mat.mean(0), v_mat.std(0)
            xs = np.arange(1, len(t_mean) + 1)
            ax_t.plot(xs, t_mean, label=opt, color=color, ls=ls, lw=2)
            ax_t.fill_between(xs, t_mean - t_std, t_mean + t_std, alpha=0.15, color=color)
            ax_v.plot(xs, v_mean, label=opt, color=color, ls=ls, lw=2)
            ax_v.fill_between(xs, v_mean - v_std, v_mean + v_std, alpha=0.15, color=color)
        ax_t.set_title(f"{arena} — Train Loss", fontsize=10, fontweight="bold")
        ax_v.set_title(f"{arena} — Val Loss",   fontsize=10, fontweight="bold")
        ax_t.set_xlabel(X_LABELS.get(arena, "Step"))
        ax_v.set_xlabel(X_LABELS.get(arena, "Step"))
        ax_t.set_ylabel("Loss"); ax_v.set_ylabel("Loss")
        ax_t.legend(fontsize=8); ax_v.legend(fontsize=8)

    fig_lc.suptitle("Learning Curves — All Arenas", fontsize=14, fontweight="bold")
    fig_lc.tight_layout()
    fig_lc.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
    tg_send_file("learning_curves.png", caption="📈 All learning curves")
    plt.close(fig_lc)

    # ── Final summary text ───────────────────────────────────────────────────
    summary = build_summary_message(results)
    tg_send(summary)
    print("\n" + summary.replace("<b>", "").replace("</b>", "")
                         .replace("<code>", "").replace("</code>", ""))


# ─────────────────────────────────────────────────────────────────────────────
# 18. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        snippet = tb[-1000:] if len(tb) > 1000 else tb
        tg_send(
            f"💥 <b>Benchmark CRASHED!</b>\n"
            f"<pre>{snippet}</pre>"
        )
        print(tb, file=sys.stderr)
        sys.exit(1)
    finally:
        tg_send("🔴 <b>RunPod shutdown initiated...</b>")
        runpod_shutdown()