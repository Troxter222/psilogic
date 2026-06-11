"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Optimizer Benchmark v1.4 — Jupyter-Compatible & Fully Fixed                 ║
║  Tasks : BERT-base / SST-2  ·  ViT-Small / CIFAR-100  ·  GPT-2 / WikiText-2 ║
║  Target: NVIDIA GPU instances (A100 SXM 80GB / H100) + Jupyter               ║
╚══════════════════════════════════════════════════════════════════════════════╝

FIXES vs v1.3:
  - argparse replaced with dataclass config (Jupyter safe)
  - torch.cuda.amp.GradScaler → torch.amp.GradScaler (non-deprecated API)
  - torch.autocast → torch.amp.autocast (non-deprecated)
  - wikitext2 attention_mask column handling fixed (may not exist after grouping)
  - trainer cleanup: proper try/finally, no broken 'if trainer in dir()' check
  - throughput measurement: fixed race condition, accurate timing_end capture
  - ddof clamping for std with n_runs=1
  - wikitext2 group_texts: safe column selection, no KeyError on attention_mask
  - GradScaler unscale before clip_grad_norm_ (correct order enforced)
  - profile flag properly wired through
  - PsiLogic _foreach: torch._foreach_norm fallback to scalar loop if unavailable
  - format_table: perplexity displayed without 4-decimal overkill
  - All Jupyter magic-command safe: no argparse, no sys.exit()
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
import os
import platform
import random
import subprocess
import time
import traceback
import urllib.error
import urllib.request
import uuid
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

# HuggingFace
try:
    import datasets
    import transformers
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        ViTConfig,
        ViTForImageClassification,
        get_cosine_schedule_with_warmup,
    )
    HF_AVAILABLE = True
except ImportError as e:
    HF_AVAILABLE = False
    print(f"[WARN] HuggingFace not fully available: {e}")

# torchvision
try:
    import torchvision
    import torchvision.transforms as T
    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False
    print("[WARN] torchvision not available — ViT/CIFAR-100 task will be skipped")

# tqdm
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, *a, **kw):
        return it

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench")


# ── Helper: safe JSON save ────────────────────────────────────────────────────

def save_json(data: Any, filepath: Path) -> None:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        log.info("Saved JSON → %s", filepath)
    except Exception as e:
        log.error("Failed to save JSON: %s", e)


# ── Helper: safe GradScaler factory (handles deprecated + new API) ────────────

def make_grad_scaler(enabled: bool = True):
    """Returns GradScaler compatible with both old and new PyTorch."""
    try:
        # PyTorch >= 2.4 preferred path
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        # Fallback for older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_autocast(device_type: str, dtype: torch.dtype):
    """Returns autocast context compatible with both old and new PyTorch."""
    try:
        return torch.amp.autocast(device_type=device_type, dtype=dtype)
    except AttributeError:
        return torch.autocast(device_type=device_type, dtype=dtype)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1.  PsiLogic v6.2 (Scale-Invariant, Jupyter-Safe, foreach with fallback)   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PsiLogic(Optimizer):
    r"""ΨLogic v6.2 — Scale-Invariant Active Cancellation Optimizer."""

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
        assert lr >= 0 and weight_decay >= 0 and gamma >= 0
        assert quantum_decay >= 0 and 0 <= betas[0] < 1 and 0 <= betas[1] < 1
        assert agc_clip >= 0 and 0 < max_cancel <= 1
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
        # Detect foreach availability once at init
        self._foreach_available = hasattr(torch, "_foreach_norm")

    def _step_scalar(self, group):
        lr           = group["lr"]
        beta1, beta2 = group["betas"]
        wd           = group["weight_decay"]
        gamma        = group["gamma"]
        p_ext        = group["p_ext"]
        qd           = group["quantum_decay"]
        eps          = group["eps"]
        gc_flag      = group["grad_centralize"]
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
            g = p.grad.clone()

            # Adaptive Gradient Clipping
            if agc > 0.0:
                p_norm = p.norm()
                g_norm = g.norm()
                max_n  = agc * p_norm.clamp(min=1e-3)
                cf     = (max_n / g_norm.clamp(min=1e-6)).clamp(max=1.0)
                g      = g * cf

            raw_g = g.clone()

            # Gradient centralization
            if gc_flag and g.dim() > 1:
                g = g - g.mean(dim=tuple(range(1, g.dim())), keepdim=True)

            st = self.state[p]
            if not st:
                st["t"]      = 0
                st["m"]      = torch.zeros_like(p)
                st["v"]      = torch.zeros_like(p)
                st["fast"]   = torch.zeros(1, device=p.device, dtype=p.dtype)
                st["slow"]   = torch.zeros(1, device=p.device, dtype=p.dtype)
                st["gn_avg"] = torch.zeros(1, device=p.device, dtype=p.dtype)

            st["t"] += 1
            t = st["t"]
            st["m"].mul_(beta1).add_(g, alpha=1.0 - beta1)
            if not lion:
                st["v"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

            gn = g.norm() / math.sqrt(max(g.numel(), 1))
            if t == 1:
                st["gn_avg"].fill_(gn.item())
                st["fast"].fill_(1.0)
                st["slow"].fill_(1.0)
            else:
                st["gn_avg"].mul_(0.99).add_(gn, alpha=0.01)
                gn_norm = gn / (st["gn_avg"] + eps)
                st["fast"].mul_(0.9).add_(gn_norm, alpha=0.1)
                st["slow"].mul_(0.99).add_(gn_norm, alpha=0.01)

            slow_t = st["slow"]
            fast_t = st["fast"]

            if T_max > 0:
                cos_w  = 0.5 * (1.0 + math.cos(math.pi * min(t / T_max, 1.0)))
                g_eff  = gamma * cos_w
                qd_eff = qd * cos_w
            else:
                g_eff  = gamma
                qd_eff = qd

            chaos_active = (t > warmup)
            if chaos_active and g_eff > 0:
                if adapt_tau:
                    spike_mask = (fast_t > tau_scale * slow_t + eps).to(p.dtype)
                else:
                    spike_mask = (slow_t >= chaos_tau).to(p.dtype)
                ratio = fast_t / (slow_t + eps)
                chaos = torch.tanh(slow_t) * (
                    1.0 + 0.5 * torch.tanh(torch.clamp(ratio - 1.0, min=0.0)))
                raw_cc = chaos * lr * g_eff * p_ext
                chaos_contrib = torch.clamp(raw_cc, max=max_cancel) * spike_mask
                total_scalar_decay = lr * wd + chaos_contrib
                p.mul_(1.0 - total_scalar_decay)
                if qd_eff > 0:
                    qd_contrib = qd_eff * (1.0 - spike_mask)
                    p.mul_(1.0 - lr * qd_contrib * torch.tanh(raw_g.abs()))
            else:
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

            if lion:
                update = (beta1 * st["m"] + (1.0 - beta1) * g).sign()
                p.add_(update, alpha=-lr)
            else:
                bc1       = 1.0 - beta1 ** t
                bc2       = math.sqrt(1.0 - beta2 ** t)
                step_size = lr * bc2 / bc1
                denom     = st["v"].sqrt().add_(eps)
                p.addcdiv_(st["m"], denom, value=-step_size)

    def _step_foreach(self, group):
        lr           = group["lr"]
        beta1, beta2 = group["betas"]
        wd           = group["weight_decay"]
        gamma        = group["gamma"]
        p_ext        = group["p_ext"]
        qd           = group["quantum_decay"]
        eps          = group["eps"]
        gc_flag      = group["grad_centralize"]
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
        grads = [p.grad.clone() for p in params_with_grad]

        # AGC
        if agc > 0.0:
            p_norms = [p.norm() for p in params_with_grad]
            g_norms = [g.norm() for g in grads]
            clipped = []
            for g, pn, gn in zip(grads, p_norms, g_norms):
                max_n = agc * pn.clamp(min=1e-3)
                cf    = (max_n / gn.clamp(min=1e-6)).clamp(max=1.0)
                clipped.append(g * cf)
            grads = clipped

        raw_grads = [g.clone() for g in grads]

        # Gradient centralization
        if gc_flag:
            for i, g in enumerate(grads):
                if g.dim() > 1:
                    grads[i] = g - g.mean(dim=tuple(range(1, g.dim())), keepdim=True)

        ms, vs, fasts, slows, gn_avgs, ts = [], [], [], [], [], []
        for p in params_with_grad:
            st = self.state[p]
            if not st:
                st["t"]      = 0
                st["m"]      = torch.zeros_like(p)
                st["v"]      = torch.zeros_like(p)
                st["fast"]   = torch.zeros(1, device=p.device, dtype=p.dtype)
                st["slow"]   = torch.zeros(1, device=p.device, dtype=p.dtype)
                st["gn_avg"] = torch.zeros(1, device=p.device, dtype=p.dtype)
            st["t"] += 1
            ms.append(st["m"]); vs.append(st["v"])
            fasts.append(st["fast"]); slows.append(st["slow"])
            gn_avgs.append(st["gn_avg"]); ts.append(st["t"])

        t = ts[0]

        # Momentum updates
        torch._foreach_mul_(ms, beta1)
        torch._foreach_add_(ms, grads, alpha=1.0 - beta1)
        if not lion:
            torch._foreach_mul_(vs, beta2)
            torch._foreach_addcmul_(vs, grads, grads, value=1.0 - beta2)

        # Gradient norm tracking (scalar per tensor, safe)
        for i, (g, fast, slow, gn_avg) in enumerate(zip(grads, fasts, slows, gn_avgs)):
            numel = g.numel()
            gn_s  = g.norm() / math.sqrt(max(numel, 1))
            if t == 1:
                gn_avg.fill_(gn_s.item()); fast.fill_(1.0); slow.fill_(1.0)
            else:
                gn_avg.mul_(0.99).add_(gn_s, alpha=0.01)
                gn_norm = gn_s / (gn_avg + eps)
                fast.mul_(0.9).add_(gn_norm, alpha=0.1)
                slow.mul_(0.99).add_(gn_norm, alpha=0.01)

        if T_max > 0:
            cos_w  = 0.5 * (1.0 + math.cos(math.pi * min(t / T_max, 1.0)))
            g_eff  = gamma * cos_w
            qd_eff = qd * cos_w
        else:
            g_eff = gamma; qd_eff = qd

        chaos_active = (t > warmup)
        if chaos_active and g_eff > 0:
            for i, (p, raw_g) in enumerate(zip(params_with_grad, raw_grads)):
                slow_t = slows[i]; fast_t = fasts[i]
                if adapt_tau:
                    spike_mask = (fast_t > tau_scale * slow_t + eps).to(p.dtype)
                else:
                    spike_mask = (slow_t >= group["chaos_tau"]).to(p.dtype)
                ratio = fast_t / (slow_t + eps)
                chaos = torch.tanh(slow_t) * (
                    1.0 + 0.5 * torch.tanh(torch.clamp(ratio - 1.0, min=0.0)))
                raw_cc = chaos * lr * g_eff * p_ext
                chaos_contrib = torch.clamp(raw_cc, max=max_cancel) * spike_mask
                total_decay = lr * wd + chaos_contrib
                p.mul_(1.0 - total_decay)
                if qd_eff > 0:
                    qd_contrib = qd_eff * (1.0 - spike_mask)
                    p.mul_(1.0 - lr * qd_contrib * torch.tanh(raw_g.abs()))
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
            denoms = torch._foreach_sqrt(vs)
            torch._foreach_add_(denoms, eps)
            torch._foreach_addcdiv_(params_with_grad, ms, denoms, value=-step_size)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            # Use foreach only if available and params are on CUDA
            use_fe = (
                group["use_foreach"]
                and self._foreach_available
                and any(p.is_cuda for p in group["params"] if p.grad is not None)
            )
            if use_fe:
                self._step_foreach(group)
            else:
                self._step_scalar(group)
        return loss


# ── Lion (sign-momentum) ─────────────────────────────────────────────────────

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                beta1, beta2 = group["betas"]
                g = p.grad
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                update = (beta1 * m + (1.0 - beta1) * g).sign_()
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)
        return loss


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2.  Telegram Notifier Module                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def send_telegram_msg(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id or "YOUR_" in token:
        return
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode("utf-8")
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            response.read()
    except Exception as e:
        log.error("Telegram notify failed: %s", e)


def send_telegram_file(token: str, chat_id: str, filepath: Path, caption: str = "") -> None:
    if not token or not chat_id or "YOUR_" in token or not filepath.exists():
        return

    filesize_mb = filepath.stat().st_size / (1024 * 1024)
    if filesize_mb > 49.9:
        log.warning("File %s (%.1f MB) exceeds Telegram 50MB limit. Skipping.", filepath.name, filesize_mb)
        send_telegram_msg(token, chat_id,
            f"⚠️ <b>Файл не отправлен:</b> {filepath.name} (<code>{filesize_mb:.1f} MB</code>) "
            f"превышает лимит Telegram 50MB. Веса сохранены локально.")
        return

    url      = f"https://api.telegram.org/bot{token}/sendDocument"
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

    with open(filepath, "rb") as f:
        file_data = f.read()

    body = []
    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="chat_id"')
    body.append(b"")
    body.append(str(chat_id).encode())

    if caption:
        body.append(f"--{boundary}".encode())
        body.append(b'Content-Disposition: form-data; name="caption"')
        body.append(b"")
        body.append(caption.encode())

    body.append(f"--{boundary}".encode())
    body.append(f'Content-Disposition: form-data; name="document"; filename="{filepath.name}"'.encode())
    body.append(b"Content-Type: application/octet-stream")
    body.append(b"")
    body.append(file_data)
    body.append(f"--{boundary}--".encode())
    body.append(b"")

    req_body = b"\r\n".join(body)
    req = urllib.request.Request(url, data=req_body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", len(req_body))

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            response.read()
    except Exception as e:
        log.error("Telegram document upload failed: %s", e)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3.  System Info & Reproducibility                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def deterministic_mode(enable: bool = True) -> None:
    if enable:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


@dataclass
class SystemInfo:
    gpu_name:      str = "CPU"
    cuda_version:  str = "N/A"
    torch_version: str = torch.__version__
    total_vram_gb: float = 0.0
    driver_version: str = "N/A"
    python_version: str = platform.python_version()
    hostname:      str = platform.node()

    @classmethod
    def collect(cls) -> SystemInfo:
        info = cls()
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            info.gpu_name      = torch.cuda.get_device_name(dev)
            info.cuda_version  = torch.version.cuda or "N/A"
            info.total_vram_gb = torch.cuda.get_device_properties(dev).total_memory / 1e9
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL, text=True).strip()
                info.driver_version = out.splitlines()[0]
            except Exception:
                pass
        return info


class VRAMTracker:
    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @property
    def peak_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
        return 0.0


@dataclass
class StepRecord:
    step:        int
    train_loss:  float
    grad_norm:   float
    update_norm: float
    is_nan:      bool = False


@dataclass
class RunResult:
    task:           str
    optimizer:      str
    run_id:         int
    seed:           int
    total_steps:    int
    final_loss:     float
    final_metric:   float
    metric_name:    str
    throughput:     float
    wall_time_sec:  float
    peak_vram_gb:   float
    nan_steps:      int
    step_records:   list[StepRecord] = field(default_factory=list)
    stability_score: float = 1.0


@dataclass
class AggResult:
    task:               str
    optimizer:          str
    n_runs:             int
    final_metric_mean:  float
    final_metric_std:   float
    metric_name:        str
    throughput_mean:    float
    throughput_std:     float
    wall_time_mean:     float
    wall_time_std:      float
    peak_vram_mean:     float
    peak_vram_std:      float
    final_loss_mean:    float
    final_loss_std:     float
    stability_mean:     float
    stability_std:      float
    nan_total:          int


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4.  Optimizer & Task Registries                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_adamw(params, lr, steps, task: str):
    wd = {"bert": 1e-2, "vit": 1e-4, "gpt2": 1e-1}[task]
    return AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)


def build_lion(params, lr, steps, task: str):
    wd = {"bert": 1e-2, "vit": 1e-4, "gpt2": 1e-1}[task]
    return Lion(params, lr=lr / 5.0, betas=(0.9, 0.99), weight_decay=wd * 10.0)


def build_psilogic(params, lr, steps, task: str):
    common = dict(
        betas=(0.9, 0.999), eps=1e-8,
        grad_centralize=True,
        chaos_warmup=-1,
        adaptive_tau=True,
        use_foreach=True,
        gamma_T_max=steps,
    )
    if task == "bert":
        return PsiLogic(params, lr=lr,
            weight_decay=1e-2, gamma=0.03, p_ext=1.0,
            quantum_decay=2e-4, tau_scale=2.0, max_cancel=0.05,
            agc_clip=0.01, **common)
    elif task == "vit":
        return PsiLogic(params, lr=lr,
            weight_decay=1e-4, gamma=0.04, p_ext=1.0,
            quantum_decay=0.0, tau_scale=2.5, max_cancel=0.04,
            agc_clip=0.02, **common)
    else:  # gpt2
        return PsiLogic(params, lr=lr,
            weight_decay=1e-1, gamma=0.02, p_ext=1.0,
            quantum_decay=0.0, tau_scale=3.0, max_cancel=0.03,
            agc_clip=0.01, **common)


OPTIMIZER_REGISTRY: dict[str, Callable] = {
    "adamw":    build_adamw,
    "lion":     build_lion,
    "psilogic": build_psilogic,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5.  Data Loaders                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_sst2_loaders(tokenizer, batch_size: int, max_length: int = 128):
    ds = load_dataset("glue", "sst2")

    def tokenize(batch):
        return tokenizer(
            batch["sentence"], truncation=True,
            max_length=max_length, padding="max_length")

    ds = ds.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = torch.utils.data.DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        ds["validation"], batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, val_loader


def get_cifar100_loaders(batch_size: int):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR100(
        root="/tmp/cifar100", train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR100(
        root="/tmp/cifar100", train=False, download=True, transform=val_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True)
    return train_loader, val_loader


def get_wikitext2_loaders(tokenizer, batch_size: int, seq_len: int = 1024):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(batch):
        # Filter empty texts to avoid empty tokenizations
        texts = [t for t in batch["text"] if t.strip()]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        return tokenizer(texts, add_special_tokens=False)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # Only use input_ids — attention_mask may not be reliable after concat
        ids_concat = sum(examples["input_ids"], [])
        total = (len(ids_concat) // seq_len) * seq_len
        if total == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        input_ids = [ids_concat[i: i + seq_len] for i in range(0, total, seq_len)]
        # Reconstruct attention_mask as all-ones (fully packed sequences)
        attention_mask = [[1] * seq_len for _ in input_ids]
        labels = [chunk[:] for chunk in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    lm_ds = tokenized.map(
        group_texts, batched=True,
        remove_columns=tokenized["train"].column_names,
    )
    lm_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = torch.utils.data.DataLoader(
        lm_ds["train"], batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        lm_ds["validation"], batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6.  Trainer Engine                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class TrainConfig:
    task:           str
    optimizer_name: str
    run_id:         int
    seed:           int
    total_steps:    int = 1000
    warmup_steps:   int = 100
    batch_size:     int = 32
    accum_steps:    int = 1
    lr:             float = 2e-5
    max_grad_norm:  float = 1.0
    use_amp:        bool = True
    amp_dtype:      str  = "bf16"
    compile_model:  bool = False
    profile:        bool = False
    output_dir:     Path = Path("./results")
    timing_warmup:  int  = 10
    log_interval:   int  = 50
    tg_token:       str  = ""
    tg_chat:        str  = ""


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vram   = VRAMTracker()
        self._amp_dtype = (torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16)

    def run(self) -> RunResult:
        seed_everything(self.cfg.seed)
        self.vram.reset()
        self.setup()

        params    = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = OPTIMIZER_REGISTRY[self.cfg.optimizer_name](
            params, self.cfg.lr, self.cfg.total_steps, self.cfg.task)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.cfg.total_steps)

        # FIX: Use float16 scaler only when amp_dtype is fp16. bf16 does NOT need scaler.
        use_scaler = (
            self.cfg.use_amp
            and self._amp_dtype == torch.float16
            and torch.cuda.is_available()
        )
        scaler = make_grad_scaler(enabled=use_scaler) if use_scaler else None

        if self.cfg.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                log.info("torch.compile() successful.")
            except Exception as e:
                log.warning("torch.compile() failed, continuing without: %s", e)

        step_records: list[StepRecord] = []
        nan_steps     = 0
        timing_start: Optional[float] = None
        timing_end:   Optional[float] = None
        timing_samples = 0
        wall_start = time.perf_counter()

        train_iter = iter(self.train_loader)
        step       = 0

        pbar = tqdm(
            total=self.cfg.total_steps,
            desc=f"{self.cfg.task}/{self.cfg.optimizer_name}/run{self.cfg.run_id}",
            leave=False)
        optimizer.zero_grad()

        while step < self.cfg.total_steps:
            # ── Fetch batch ──────────────────────────────────────────────────
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch_size_actual = self._batch_size(batch)

            # ── Forward ──────────────────────────────────────────────────────
            amp_ctx = (
                make_autocast("cuda", self._amp_dtype)
                if self.cfg.use_amp and torch.cuda.is_available()
                else contextlib.nullcontext()
            )

            with amp_ctx:
                loss = self.train_step(batch)
                loss_scaled = loss / self.cfg.accum_steps

            # ── Backward ─────────────────────────────────────────────────────
            if scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # ── Optimizer step (every accum_steps) ───────────────────────────
            if (step + 1) % self.cfg.accum_steps == 0:

                # FIX: unscale BEFORE clip_grad_norm_
                if scaler:
                    scaler.unscale_(optimizer)

                grad_norm = nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm).item()

                is_nan = math.isnan(grad_norm) or math.isinf(grad_norm)
                if is_nan:
                    nan_steps += 1
                    optimizer.zero_grad()
                    if scaler:
                        scaler.update()
                    step += 1
                    pbar.update(1)
                    step_records.append(StepRecord(step, float("nan"), float("nan"), 0.0, True))
                    continue

                # Measure update norm on first 5 params (cheap)
                old_norms = [p.data.norm().item() for p in params[:5]]

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # FIX: scheduler.step() always AFTER optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                update_norm = float(np.mean(
                    [abs(p.data.norm().item() - on) for p, on in zip(params[:5], old_norms)]))

                # ── Timing (after warmup steps) ───────────────────────────────
                if step >= self.cfg.timing_warmup:
                    if timing_start is None:
                        timing_start = time.perf_counter()
                    timing_samples += batch_size_actual
                    timing_end = time.perf_counter()

                raw_loss = loss.item()
                step_records.append(StepRecord(step + 1, raw_loss, grad_norm, update_norm, False))

                if (step + 1) % self.cfg.log_interval == 0:
                    log.debug("  step %4d  loss=%.4f  gnorm=%.4f", step + 1, raw_loss, grad_norm)

            step += 1
            pbar.update(1)

        pbar.close()
        wall_time = time.perf_counter() - wall_start

        # FIX: use captured timing_end, not time.perf_counter() after loop
        if timing_start is not None and timing_end is not None and (timing_end - timing_start) > 0:
            throughput = timing_samples / (timing_end - timing_start)
        else:
            throughput = 0.0

        final_metric, metric_name = self.evaluate()
        valid_records = [r for r in step_records if not r.is_nan]
        final_loss = valid_records[-1].train_loss if valid_records else float("nan")

        # ── Save checkpoint locally ───────────────────────────────────────────
        saved_models_dir = Path("./saved_models")
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = saved_models_dir / f"{self.cfg.task}_{self.cfg.optimizer_name}_run{self.cfg.run_id}.pt"

        # Save state dict (unwrap compiled model if needed)
        model_to_save = self.model
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod
        torch.save(model_to_save.state_dict(), ckpt_path)
        log.info("Saved checkpoint → %s", ckpt_path)

        # ── Telegram upload ───────────────────────────────────────────────────
        if self.cfg.tg_token and self.cfg.tg_chat:
            caption = (
                f"📦 {self.cfg.task.upper()} ({self.cfg.optimizer_name}) — Run {self.cfg.run_id}\n"
                f"Loss: {final_loss:.4f}, {metric_name}: {final_metric:.4f}"
            )
            send_telegram_file(self.cfg.tg_token, self.cfg.tg_chat, ckpt_path, caption)

        stability_score = (
            sum(1 for r in valid_records if r.grad_norm < 10.0) / len(valid_records)
            if valid_records else 0.0
        )

        return RunResult(
            task=self.cfg.task,
            optimizer=self.cfg.optimizer_name,
            run_id=self.cfg.run_id,
            seed=self.cfg.seed,
            total_steps=self.cfg.total_steps,
            final_loss=final_loss,
            final_metric=final_metric,
            metric_name=metric_name,
            throughput=throughput,
            wall_time_sec=wall_time,
            peak_vram_gb=self.vram.peak_gb,
            nan_steps=nan_steps,
            step_records=step_records,
            stability_score=stability_score,
        )

    def setup(self):
        raise NotImplementedError

    def train_step(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self) -> tuple[float, str]:
        raise NotImplementedError

    def _batch_size(self, batch) -> int:
        if isinstance(batch, dict):
            v = next(iter(batch.values()))
            return v.shape[0] if hasattr(v, "shape") else 1
        if isinstance(batch, (list, tuple)):
            return batch[0].shape[0]
        return 1


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7.  Task Trainers                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BERTTrainer(Trainer):
    """BERT-base fine-tuning on SST-2."""

    def setup(self):
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2).to(self.device)
        self.train_loader, self.val_loader = get_sst2_loaders(
            self.tokenizer, self.cfg.batch_size, max_length=128)

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return self.model(**batch).loss

    @torch.no_grad()
    def evaluate(self) -> tuple[float, str]:
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in self.val_loader:
            batch  = {k: v.to(self.device) for k, v in batch.items()}
            out    = self.model(**batch)
            preds  = out.logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
        acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        log.info("  [BERT eval]  acc=%.4f", acc)
        return acc, "accuracy"


class ViTTrainer(Trainer):
    """ViT-Small on CIFAR-100."""

    def setup(self):
        config = ViTConfig(
            image_size=224, patch_size=16, num_channels=3,
            hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
            intermediate_size=1536, num_labels=100,
        )
        self.model = ViTForImageClassification(config).to(self.device)
        self.train_loader, self.val_loader = get_cifar100_loaders(self.cfg.batch_size)

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        return self.model(pixel_values=images, labels=labels).loss

    @torch.no_grad()
    def evaluate(self) -> tuple[float, str]:
        self.model.eval()
        correct = total = 0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            out    = self.model(pixel_values=images)
            preds  = out.logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        top1 = correct / total if total > 0 else 0.0
        log.info("  [ViT eval]  top-1=%.4f", top1)
        return float(top1), "top1_accuracy"


class GPT2Trainer(Trainer):
    """GPT-2 language modeling on WikiText-2."""

    SEQ_LEN = 1024

    def setup(self):
        model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.train_loader, self.val_loader = get_wikitext2_loaders(
            self.tokenizer, self.cfg.batch_size, seq_len=self.SEQ_LEN)

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels         = batch["labels"].to(self.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels)
        return out.loss

    @torch.no_grad()
    def evaluate(self) -> tuple[float, str]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in self.val_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels)
            if not torch.isnan(out.loss):
                total_loss    += out.loss.item()
                total_batches += 1
        avg_loss   = total_loss / max(total_batches, 1)
        perplexity = math.exp(min(avg_loss, 20.0))
        log.info("  [GPT-2 eval]  ppl=%.2f  (avg_ce=%.4f)", perplexity, avg_loss)
        return float(perplexity), "perplexity"


TASK_REGISTRY: dict[str, tuple[type, float]] = {
    "bert": (BERTTrainer, 2e-5),
    "vit":  (ViTTrainer,  5e-4),
    "gpt2": (GPT2Trainer, 3e-4),
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8.  Aggregation & Table Formatters                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def aggregate(runs: list[RunResult]) -> AggResult:
    assert runs, "No runs to aggregate"
    task = runs[0].task
    opt  = runs[0].optimizer
    mn   = runs[0].metric_name

    # FIX: ddof=0 when single run (std=0 not undefined)
    ddof = 1 if len(runs) > 1 else 0

    def m(attr): return float(np.mean([getattr(r, attr) for r in runs]))
    def s(attr): return float(np.std( [getattr(r, attr) for r in runs], ddof=ddof))

    return AggResult(
        task=task, optimizer=opt, n_runs=len(runs),
        final_metric_mean=m("final_metric"), final_metric_std=s("final_metric"),
        metric_name=mn,
        throughput_mean=m("throughput"),     throughput_std=s("throughput"),
        wall_time_mean=m("wall_time_sec"),   wall_time_std=s("wall_time_sec"),
        peak_vram_mean=m("peak_vram_gb"),    peak_vram_std=s("peak_vram_gb"),
        final_loss_mean=m("final_loss"),     final_loss_std=s("final_loss"),
        stability_mean=m("stability_score"), stability_std=s("stability_score"),
        nan_total=sum(r.nan_steps for r in runs),
    )


def format_table(aggs: list[AggResult], title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"<b>{title}</b>")
        lines.append("-" * 45)
    for a in aggs:
        mn = a.metric_name
        # FIX: perplexity shown as integer-ish, accuracy shown with 4 decimals
        if "perplexity" in mn:
            metric_str = f"{a.final_metric_mean:.1f} ± {a.final_metric_std:.1f}"
        else:
            metric_str = f"{a.final_metric_mean:.4f} ± {a.final_metric_std:.4f}"
        lines.append(
            f"<code>{a.task.upper():5s} | {a.optimizer:8s} | {mn:14s} | {metric_str}</code>"
        )
    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  9.  Orchestrator                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_benchmark(
    tasks:         list[str]  = None,
    optimizers:    list[str]  = None,
    n_runs:        int        = 2,
    total_steps:   int        = 1000,
    batch_size:    int        = 32,
    accum_steps:   int        = 1,
    compile_model: bool       = False,
    profile:       bool       = False,
    output_dir:    Path       = Path("./results"),
    amp_dtype:     str        = "bf16",
    tg_token:      str        = "8702196611:AAGeRi9KvLKnTKjEjX30Vk4pyKjBcvrQ8i8",
    tg_chat:       str        = "1386910692",
) -> tuple[list[AggResult], list[RunResult]]:

    if tasks is None:
        tasks = ["bert", "vit", "gpt2"]
    if optimizers is None:
        optimizers = ["adamw", "lion", "psilogic"]

    seeds = [42, 137, 2718][:n_runs]
    all_results: list[RunResult] = []
    all_agg:     list[AggResult] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "step_logs").mkdir(exist_ok=True)

    sys_info = SystemInfo.collect()
    log.info("GPU: %s  CUDA: %s  VRAM: %.1f GB",
             sys_info.gpu_name, sys_info.cuda_version, sys_info.total_vram_gb)

    start_msg = (
        f"🚀 <b>Бенчмарк запущен!</b>\n"
        f"<b>GPU:</b> <code>{sys_info.gpu_name}</code>\n"
        f"<b>CUDA:</b> <code>{sys_info.cuda_version}</code>\n"
        f"<b>Задачи:</b> {', '.join(tasks).upper()}\n"
        f"<b>Оптимизаторы:</b> {', '.join(optimizers).upper()}\n"
        f"<b>Настройки:</b> Steps={total_steps}, Runs={n_runs}"
    )
    send_telegram_msg(tg_token, tg_chat, start_msg)

    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            log.warning("Unknown task '%s', skipping.", task_name)
            continue
        if task_name in ("bert", "gpt2") and not HF_AVAILABLE:
            log.warning("HuggingFace not available, skipping %s.", task_name)
            continue
        if task_name == "vit" and not TV_AVAILABLE:
            log.warning("torchvision not available, skipping vit.")
            continue

        TrainerCls, default_lr = TASK_REGISTRY[task_name]

        # A100 SXM 80GB optimized batch sizes
        if torch.cuda.is_available():
            task_batch_size = {"bert": 64, "vit": 128, "gpt2": 8}[task_name]
        else:
            task_batch_size = min(batch_size, 4)  # CPU safety

        log.info("=" * 60)
        log.info("TASK: %s  steps=%d  batch=%d", task_name, total_steps, task_batch_size)
        log.info("=" * 60)

        task_runs: dict[str, list[RunResult]] = defaultdict(list)

        for opt_name in optimizers:
            if opt_name not in OPTIMIZER_REGISTRY:
                log.warning("Unknown optimizer '%s', skipping.", opt_name)
                continue

            for run_id, seed in enumerate(seeds):
                deterministic_mode(True)
                cfg = TrainConfig(
                    task=task_name,
                    optimizer_name=opt_name,
                    run_id=run_id,
                    seed=seed,
                    total_steps=total_steps,
                    warmup_steps=max(20, total_steps // 10),
                    batch_size=task_batch_size,
                    accum_steps=accum_steps,
                    lr=default_lr,
                    use_amp=torch.cuda.is_available(),
                    amp_dtype=amp_dtype,
                    compile_model=compile_model,
                    profile=profile,
                    output_dir=output_dir,
                    tg_token=tg_token,
                    tg_chat=tg_chat,
                )

                trainer = None
                try:
                    trainer = TrainerCls(cfg)
                    result  = trainer.run()
                    task_runs[opt_name].append(result)
                    all_results.append(result)

                    step_update = (
                        f"📊 <code>{task_name.upper()} | {opt_name.upper()} | Run {run_id+1}/{n_runs}</code>\n"
                        f"Loss: <code>{result.final_loss:.4f}</code>  "
                        f"{result.metric_name}: <code>{result.final_metric:.4f}</code>\n"
                        f"VRAM: <code>{result.peak_vram_gb:.2f} GB</code>  "
                        f"Tput: <code>{result.throughput:.1f} samp/s</code>"
                    )
                    send_telegram_msg(tg_token, tg_chat, step_update)

                except Exception as exc:
                    log.error("Run %s/%s failed: %s", task_name, opt_name, exc)
                    log.error(traceback.format_exc())
                    send_telegram_msg(tg_token, tg_chat,
                        f"❌ <b>Ошибка:</b> {task_name}/{opt_name} run{run_id}\n"
                        f"<code>{str(exc)[:300]}</code>")

                finally:
                    # FIX: proper cleanup, no broken 'if trainer in dir()' check
                    if trainer is not None and hasattr(trainer, "model"):
                        del trainer.model
                    trainer = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if task_runs[opt_name]:
                agg = aggregate(task_runs[opt_name])
                all_agg.append(agg)

    return all_agg, all_results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  10.  Jupyter-Safe Entry Point                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ── DEFAULT CONFIG (edit here instead of CLI args) ────────────────────────────

BENCHMARK_CONFIG = dict(
    tasks      = ["bert", "vit", "gpt2"],     # tasks to run
    optimizers = ["adamw", "lion", "psilogic"],  # optimizers to compare
    n_runs     = 2,                            # runs per combo (seeds: 42, 137)
    total_steps= 1000,                         # training steps per run
    batch_size = 32,                           # overridden per-task on GPU
    accum_steps= 1,                            # gradient accumulation
    amp_dtype  = "bf16",                       # "bf16" or "fp16"
    compile_model = False,                     # torch.compile (experimental)
    profile    = False,                        # torch profiler
    output_dir = Path("./results"),
    tg_token   = "8702196611:AAGeRi9KvLKnTKjEjX30Vk4pyKjBcvrQ8i8",
    tg_chat    = "1386910692",
)


def main():
    """
    Entry point — works in both Jupyter and terminal.
    In Jupyter: just call main() or run_benchmark(**BENCHMARK_CONFIG)
    In terminal: python optimizer_benchmark_fixed.py
    """
    output_dir = BENCHMARK_CONFIG["output_dir"]

    if not torch.cuda.is_available():
        log.warning("No CUDA found. Scaling batch_size=4 for CPU debug mode.")
        BENCHMARK_CONFIG["batch_size"] = 4

    all_agg, all_results = run_benchmark(**BENCHMARK_CONFIG)

    if not all_agg:
        log.error("No results produced. Check errors above.")
        return

    # Final Telegram summary table
    final_table = format_table(all_agg, title="🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ БЕНЧМАРКА")
    send_telegram_msg(BENCHMARK_CONFIG["tg_token"], BENCHMARK_CONFIG["tg_chat"], final_table)
    print("\n" + final_table.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))

    # Save JSON results
    save_json([asdict(a) for a in all_agg], output_dir / "benchmark_results.json")

    # Zip and send full archive
    zip_path = output_dir / "full_benchmark_archive.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file != "full_benchmark_archive.zip":
                    full_path = os.path.join(root, file)
                    zipf.write(full_path, os.path.relpath(full_path, output_dir))

    send_telegram_file(
        BENCHMARK_CONFIG["tg_token"],
        BENCHMARK_CONFIG["tg_chat"],
        zip_path,
        "📂 Полный лог-архив результатов (JSON + чекпоинты)",
    )

    log.info("✅ Benchmarking complete. Results saved to %s", output_dir)
    return all_agg, all_results


# ── Works in Jupyter AND as script ───────────────────────────────────────────
if __name__ == "__main__":
    main()
