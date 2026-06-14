"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Optimizer Benchmark v2.0 — package-backed, Jupyter & CLI compatible          ║
║  Arenas : BERT/SST-2 · ViT-S/CIFAR-100 · GPT-2/WikiText-2                     ║
║           ResNet-18/CIFAR-10 · nanoGPT/Tiny Shakespeare                       ║
║  Target : NVIDIA GPU instances (A100 / H100) + Jupyter + CPU debug            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Changes vs v1.4:
  - PsiLogic imported from the `psilogic` package (inline copy deleted)
  - Telegram credentials read from PSILOGIC_TG_TOKEN / PSILOGIC_TG_CHAT env vars
  - `--preset vit` shortcut: PsiLogicViT with per-group gamma param groups
  - Local cosine-with-warmup scheduler (no transformers dependency for it)
  - New arenas: cifar10 (ResNet-18) and nanogpt (char-GPT, Tiny Shakespeare)
  - argparse only under __main__ — main()/run_benchmark() stay Jupyter-safe
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
import urllib.request
import uuid
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

# ── psilogic package (single source of truth for the optimizer) ──────────────
from psilogic import (
    PsiLogic,
    PsiLogicViT,
    gpt_scratch_defaults,
    nlp_defaults,
    vision_defaults,
)

# HuggingFace
try:
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        ViTConfig,
        ViTForImageClassification,
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
    print("[WARN] torchvision not available — vision tasks will be skipped")

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

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
DATA_ROOT = Path(os.environ.get("PSILOGIC_DATA_DIR", "/tmp/psilogic_data"))


# ── Helpers ───────────────────────────────────────────────────────────────────


def save_json(data: Any, filepath: Path) -> None:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        log.info("Saved JSON → %s", filepath)
    except Exception as e:
        log.error("Failed to save JSON: %s", e)


def make_grad_scaler(enabled: bool = True):
    """GradScaler compatible with both old and new PyTorch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_autocast(device_type: str, dtype: torch.dtype):
    """autocast context compatible with both old and new PyTorch."""
    try:
        return torch.amp.autocast(device_type=device_type, dtype=dtype)
    except AttributeError:
        return torch.autocast(device_type=device_type, dtype=dtype)


def cosine_with_warmup(
    optimizer: Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to zero (transformers-free)."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Lion baseline (sign-momentum) ─────────────────────────────────────────────


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


# ── Telegram notifier (credentials via env only) ──────────────────────────────


def tg_credentials() -> tuple[str, str]:
    return (
        os.environ.get("PSILOGIC_TG_TOKEN", ""),
        os.environ.get("PSILOGIC_TG_CHAT", ""),
    )


def send_telegram_msg(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            response.read()
    except Exception as e:
        log.error("Telegram notify failed: %s", e)


def send_telegram_file(token: str, chat_id: str, filepath: Path, caption: str = "") -> None:
    if not token or not chat_id or not filepath.exists():
        return

    filesize_mb = filepath.stat().st_size / (1024 * 1024)
    if filesize_mb > 49.9:
        log.warning(
            "File %s (%.1f MB) exceeds Telegram 50MB limit. Skipping.",
            filepath.name,
            filesize_mb,
        )
        send_telegram_msg(
            token,
            chat_id,
            f"⚠️ <b>File not sent:</b> {filepath.name} (<code>{filesize_mb:.1f} MB</code>) "
            f"exceeds the Telegram 50MB limit. Saved locally instead.",
        )
        return

    url = f"https://api.telegram.org/bot{token}/sendDocument"
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
    body.append(
        f'Content-Disposition: form-data; name="document"; filename="{filepath.name}"'.encode()
    )
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


# ── System info & reproducibility ─────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deterministic_mode(enable: bool = True) -> None:
    if enable:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


@dataclass
class SystemInfo:
    gpu_name: str = "CPU"
    cuda_version: str = "N/A"
    torch_version: str = torch.__version__
    total_vram_gb: float = 0.0
    driver_version: str = "N/A"
    python_version: str = platform.python_version()
    hostname: str = platform.node()

    @classmethod
    def collect(cls) -> SystemInfo:
        info = cls()
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            info.gpu_name = torch.cuda.get_device_name(dev)
            info.cuda_version = torch.version.cuda or "N/A"
            info.total_vram_gb = torch.cuda.get_device_properties(dev).total_memory / 1e9
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
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
    step: int
    train_loss: float
    grad_norm: float
    update_norm: float
    is_nan: bool = False


@dataclass
class RunResult:
    task: str
    optimizer: str
    run_id: int
    seed: int
    total_steps: int
    final_loss: float
    final_metric: float
    metric_name: str
    throughput: float
    wall_time_sec: float
    peak_vram_gb: float
    nan_steps: int
    step_records: list[StepRecord] = field(default_factory=list)
    stability_score: float = 1.0


@dataclass
class AggResult:
    task: str
    optimizer: str
    n_runs: int
    final_metric_mean: float
    final_metric_std: float
    metric_name: str
    throughput_mean: float
    throughput_std: float
    wall_time_mean: float
    wall_time_std: float
    peak_vram_mean: float
    peak_vram_std: float
    final_loss_mean: float
    final_loss_std: float
    stability_mean: float
    stability_std: float
    nan_total: int


# ── Optimizer registry (builders receive the full model) ─────────────────────

_TASK_WD = {"bert": 1e-2, "vit": 1e-4, "gpt2": 1e-1, "cifar10": 5e-4, "nanogpt": 1e-1}


def build_adamw(model: nn.Module, lr: float, steps: int, task: str, preset: str) -> Optimizer:
    wd = _TASK_WD[task]
    return AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)


def build_lion(model: nn.Module, lr: float, steps: int, task: str, preset: str) -> Optimizer:
    wd = _TASK_WD[task]
    return Lion(model.parameters(), lr=lr / 5.0, betas=(0.9, 0.99), weight_decay=wd * 10.0)


def build_psilogic(model: nn.Module, lr: float, steps: int, task: str, preset: str) -> Optimizer:
    if preset == "vit":
        # Roadmap V1/V2 path: per-group gamma split via vit_param_groups.
        return PsiLogicViT(model, lr=lr, gamma_T_max=steps)
    if preset == "auto":
        return PsiLogic.auto(model, lr=lr, total_steps=steps)

    preset_fns: dict[str, Callable[[int], dict]] = {
        "bert": nlp_defaults,
        "vit": vision_defaults,
        "gpt2": gpt_scratch_defaults,
        "cifar10": vision_defaults,
        "nanogpt": gpt_scratch_defaults,
    }
    cfg = preset_fns[task](steps)
    cfg["weight_decay"] = _TASK_WD[task]
    return PsiLogic(model.parameters(), lr=lr, **cfg)


OPTIMIZER_REGISTRY: dict[str, Callable] = {
    "adamw": build_adamw,
    "lion": build_lion,
    "psilogic": build_psilogic,
}


# ── Data loaders ──────────────────────────────────────────────────────────────


def get_sst2_loaders(tokenizer, batch_size: int, max_length: int = 128):
    ds = load_dataset("glue", "sst2")

    def tokenize(batch):
        return tokenizer(
            batch["sentence"], truncation=True, max_length=max_length, padding="max_length"
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = torch.utils.data.DataLoader(
        ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        ds["validation"], batch_size=batch_size * 2, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader


def get_cifar100_loaders(batch_size: int):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    root = str(DATA_ROOT / "cifar100")
    train_ds = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=train_tf
    )
    val_ds = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=val_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def get_cifar10_loaders(batch_size: int):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    root = str(DATA_ROOT / "cifar10")
    train_ds = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf
    )
    val_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=val_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader


def get_wikitext2_loaders(tokenizer, batch_size: int, seq_len: int = 1024):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(batch):
        texts = [t for t in batch["text"] if t.strip()]
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        return tokenizer(texts, add_special_tokens=False)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        ids_concat = sum(examples["input_ids"], [])
        total = (len(ids_concat) // seq_len) * seq_len
        if total == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        input_ids = [ids_concat[i : i + seq_len] for i in range(0, total, seq_len)]
        attention_mask = [[1] * seq_len for _ in input_ids]
        labels = [chunk[:] for chunk in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    lm_ds = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized["train"].column_names,
    )
    lm_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = torch.utils.data.DataLoader(
        lm_ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        lm_ds["validation"], batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader


def get_tinyshakespeare_loaders(batch_size: int, block_size: int = 128):
    """Char-level Tiny Shakespeare windows for the nanoGPT arena."""
    data_dir = DATA_ROOT / "tinyshakespeare"
    data_dir.mkdir(parents=True, exist_ok=True)
    txt_path = data_dir / "input.txt"
    if not txt_path.exists():
        log.info("Downloading Tiny Shakespeare → %s", txt_path)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, txt_path)

    text = txt_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    def windows(seq: torch.Tensor) -> torch.utils.data.TensorDataset:
        n = (len(seq) - 1) // block_size
        x = seq[: n * block_size].reshape(n, block_size)
        y = seq[1 : n * block_size + 1].reshape(n, block_size)
        return torch.utils.data.TensorDataset(x, y)

    train_loader = torch.utils.data.DataLoader(
        windows(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        windows(val_data), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, val_loader, len(chars)


# ── nanoGPT-style char model ──────────────────────────────────────────────────


class CharGPTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp_c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.mlp_c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.ln_1(x)
        q, k, v = self.c_attn(h).split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.dropout(self.c_proj(attn))
        h = self.ln_2(x)
        return x + self.dropout(self.mlp_c_proj(F.gelu(self.mlp_c_fc(h))))


class CharGPT(nn.Module):
    """Minimal nanoGPT-style character language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([CharGPTBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(positions))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


# ── Trainer engine ────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    task: str
    optimizer_name: str
    run_id: int
    seed: int
    total_steps: int = 1000
    warmup_steps: int = 100
    batch_size: int = 32
    accum_steps: int = 1
    lr: float = 2e-5
    max_grad_norm: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "bf16"
    compile_model: bool = False
    profile: bool = False
    output_dir: Path = Path("./results")
    timing_warmup: int = 10
    log_interval: int = 50
    psilogic_preset: str = "task"  # "task" | "vit" | "auto"
    tg_token: str = ""
    tg_chat: str = ""


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vram = VRAMTracker()
        self._amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    def run(self) -> RunResult:
        seed_everything(self.cfg.seed)
        self.vram.reset()
        self.setup()

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = OPTIMIZER_REGISTRY[self.cfg.optimizer_name](
            self.model, self.cfg.lr, self.cfg.total_steps, self.cfg.task, self.cfg.psilogic_preset
        )

        scheduler = cosine_with_warmup(optimizer, self.cfg.warmup_steps, self.cfg.total_steps)

        use_scaler = (
            self.cfg.use_amp and self._amp_dtype == torch.float16 and torch.cuda.is_available()
        )
        scaler = make_grad_scaler(enabled=use_scaler) if use_scaler else None

        if self.cfg.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                log.info("torch.compile() successful.")
            except Exception as e:
                log.warning("torch.compile() failed, continuing without: %s", e)

        step_records: list[StepRecord] = []
        nan_steps = 0
        timing_start: Optional[float] = None
        timing_end: Optional[float] = None
        timing_samples = 0
        wall_start = time.perf_counter()

        train_iter = iter(self.train_loader)
        step = 0

        pbar = tqdm(
            total=self.cfg.total_steps,
            desc=f"{self.cfg.task}/{self.cfg.optimizer_name}/run{self.cfg.run_id}",
            leave=False,
        )
        optimizer.zero_grad()

        while step < self.cfg.total_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch_size_actual = self._batch_size(batch)

            amp_ctx = (
                make_autocast("cuda", self._amp_dtype)
                if self.cfg.use_amp and torch.cuda.is_available()
                else contextlib.nullcontext()
            )

            with amp_ctx:
                loss = self.train_step(batch)
                loss_scaled = loss / self.cfg.accum_steps

            if scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step + 1) % self.cfg.accum_steps == 0:
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

                old_norms = [p.data.norm().item() for p in params[:5]]

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                update_norm = float(
                    np.mean(
                        [abs(p.data.norm().item() - on) for p, on in zip(params[:5], old_norms)]
                    )
                )

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

        if timing_start is not None and timing_end is not None and (timing_end - timing_start) > 0:
            throughput = timing_samples / (timing_end - timing_start)
        else:
            throughput = 0.0

        final_metric, metric_name = self.evaluate()
        valid_records = [r for r in step_records if not r.is_nan]
        final_loss = valid_records[-1].train_loss if valid_records else float("nan")

        saved_models_dir = Path("./saved_models")
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = saved_models_dir / (
            f"{self.cfg.task}_{self.cfg.optimizer_name}_run{self.cfg.run_id}.pt"
        )

        model_to_save = self.model
        if hasattr(model_to_save, "_orig_mod"):
            model_to_save = model_to_save._orig_mod
        torch.save(model_to_save.state_dict(), ckpt_path)
        log.info("Saved checkpoint → %s", ckpt_path)

        if self.cfg.tg_token and self.cfg.tg_chat:
            caption = (
                f"📦 {self.cfg.task.upper()} ({self.cfg.optimizer_name}) — "
                f"Run {self.cfg.run_id}\n"
                f"Loss: {final_loss:.4f}, {metric_name}: {final_metric:.4f}"
            )
            send_telegram_file(self.cfg.tg_token, self.cfg.tg_chat, ckpt_path, caption)

        stability_score = (
            sum(1 for r in valid_records if r.grad_norm < 10.0) / len(valid_records)
            if valid_records
            else 0.0
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


# ── Task trainers ─────────────────────────────────────────────────────────────


class BERTTrainer(Trainer):
    """BERT-base fine-tuning on SST-2."""

    def setup(self):
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(self.device)
        self.train_loader, self.val_loader = get_sst2_loaders(
            self.tokenizer, self.cfg.batch_size, max_length=128
        )

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return self.model(**batch).loss

    @torch.no_grad()
    def evaluate(self) -> tuple[float, str]:
        self.model.eval()
        all_preds, all_labels = [], []
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            preds = out.logits.argmax(dim=-1).cpu().numpy()
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
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            num_labels=100,
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
            out = self.model(pixel_values=images)
            preds = out.logits.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
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
            self.tokenizer, self.cfg.batch_size, seq_len=self.SEQ_LEN
        )

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss

    @torch.no_grad()
    def evaluate(self) -> tuple[float, str]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if not torch.isnan(out.loss):
                total_loss += out.loss.item()
                total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        perplexity = math.exp(min(avg_loss, 20.0))
        log.info("  [GPT-2 eval]  ppl=%.2f  (avg_ce=%.4f)", perplexity, avg_loss)
        return float(perplexity), "perplexity"


class Cifar10ResNetTrainer(Trainer):
    """ResNet-18 (CIFAR-adapted stem) on CIFAR-10 — reference Arena 1."""

    def setup(self):
        model = torchvision.models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.val_loader = get_cifar10_loaders(self.cfg.batch_size)

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        return self.criterion(self.model(images), labels)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, str]:
        self.model.eval()
        correct = total = 0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            preds = self.model(images).argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        top1 = correct / total if total > 0 else 0.0
        log.info("  [CIFAR-10 eval]  top-1=%.4f", top1)
        return float(top1), "top1_accuracy"


class NanoGPTTrainer(Trainer):
    """Char-level nanoGPT on Tiny Shakespeare — reference Arena 7."""

    BLOCK_SIZE = 128

    def setup(self):
        self.train_loader, self.val_loader, vocab_size = get_tinyshakespeare_loaders(
            self.cfg.batch_size, block_size=self.BLOCK_SIZE
        )
        self.model = CharGPT(
            vocab_size=vocab_size,
            block_size=self.BLOCK_SIZE,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.1,
        ).to(self.device)

    def train_step(self, batch) -> torch.Tensor:
        self.model.train()
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        _, loss = self.model(x, y)
        return loss

    @torch.no_grad()
    def evaluate(self) -> tuple[float, str]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            _, loss = self.model(x, y)
            if not torch.isnan(loss):
                total_loss += loss.item()
                total_batches += 1
        val_loss = total_loss / max(total_batches, 1)
        log.info("  [nanoGPT eval]  val_loss=%.4f", val_loss)
        return float(val_loss), "val_loss"


TASK_REGISTRY: dict[str, tuple[type, float]] = {
    "bert": (BERTTrainer, 2e-5),
    "vit": (ViTTrainer, 5e-4),
    "gpt2": (GPT2Trainer, 3e-4),
    "cifar10": (Cifar10ResNetTrainer, 1e-3),
    "nanogpt": (NanoGPTTrainer, 3e-4),
}

_GPU_BATCH_SIZES = {"bert": 64, "vit": 128, "gpt2": 8, "cifar10": 256, "nanogpt": 64}
_HF_TASKS = {"bert", "gpt2", "vit"}
_TV_TASKS = {"vit", "cifar10"}


# ── Aggregation & table formatting ────────────────────────────────────────────


def aggregate(runs: list[RunResult]) -> AggResult:
    if not runs:
        raise ValueError("No runs to aggregate")
    task = runs[0].task
    opt = runs[0].optimizer
    mn = runs[0].metric_name

    ddof = 1 if len(runs) > 1 else 0

    def m(attr):
        return float(np.mean([getattr(r, attr) for r in runs]))

    def s(attr):
        return float(np.std([getattr(r, attr) for r in runs], ddof=ddof))

    return AggResult(
        task=task,
        optimizer=opt,
        n_runs=len(runs),
        final_metric_mean=m("final_metric"),
        final_metric_std=s("final_metric"),
        metric_name=mn,
        throughput_mean=m("throughput"),
        throughput_std=s("throughput"),
        wall_time_mean=m("wall_time_sec"),
        wall_time_std=s("wall_time_sec"),
        peak_vram_mean=m("peak_vram_gb"),
        peak_vram_std=s("peak_vram_gb"),
        final_loss_mean=m("final_loss"),
        final_loss_std=s("final_loss"),
        stability_mean=m("stability_score"),
        stability_std=s("stability_score"),
        nan_total=sum(r.nan_steps for r in runs),
    )


def format_table(aggs: list[AggResult], title: str = "") -> str:
    lines = []
    if title:
        lines.append(f"<b>{title}</b>")
        lines.append("-" * 45)
    for a in aggs:
        mn = a.metric_name
        if "perplexity" in mn:
            metric_str = f"{a.final_metric_mean:.1f} ± {a.final_metric_std:.1f}"
        else:
            metric_str = f"{a.final_metric_mean:.4f} ± {a.final_metric_std:.4f}"
        lines.append(
            f"<code>{a.task.upper():8s} | {a.optimizer:8s} | {mn:14s} | {metric_str}</code>"
        )
    return "\n".join(lines)


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_benchmark(
    tasks: Optional[list[str]] = None,
    optimizers: Optional[list[str]] = None,
    n_runs: int = 2,
    total_steps: int = 1000,
    batch_size: int = 32,
    accum_steps: int = 1,
    compile_model: bool = False,
    profile: bool = False,
    output_dir: Path = Path("./results"),
    amp_dtype: str = "bf16",
    psilogic_preset: str = "task",
    tg_token: Optional[str] = None,
    tg_chat: Optional[str] = None,
) -> tuple[list[AggResult], list[RunResult]]:

    if tasks is None:
        tasks = ["bert", "vit", "gpt2"]
    if optimizers is None:
        optimizers = ["adamw", "lion", "psilogic"]

    env_token, env_chat = tg_credentials()
    tg_token = tg_token if tg_token is not None else env_token
    tg_chat = tg_chat if tg_chat is not None else env_chat

    seeds = [42, 137, 2718, 31415, 27182][:n_runs]
    all_results: list[RunResult] = []
    all_agg: list[AggResult] = []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "step_logs").mkdir(exist_ok=True)

    sys_info = SystemInfo.collect()
    log.info(
        "GPU: %s  CUDA: %s  VRAM: %.1f GB",
        sys_info.gpu_name,
        sys_info.cuda_version,
        sys_info.total_vram_gb,
    )

    start_msg = (
        f"🚀 <b>Benchmark started</b>\n"
        f"<b>GPU:</b> <code>{sys_info.gpu_name}</code>\n"
        f"<b>CUDA:</b> <code>{sys_info.cuda_version}</code>\n"
        f"<b>Tasks:</b> {', '.join(tasks).upper()}\n"
        f"<b>Optimizers:</b> {', '.join(optimizers).upper()}\n"
        f"<b>Settings:</b> Steps={total_steps}, Runs={n_runs}"
    )
    send_telegram_msg(tg_token, tg_chat, start_msg)

    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            log.warning("Unknown task '%s', skipping.", task_name)
            continue
        if task_name in _HF_TASKS and not HF_AVAILABLE:
            log.warning("HuggingFace not available, skipping %s.", task_name)
            continue
        if task_name in _TV_TASKS and not TV_AVAILABLE:
            log.warning("torchvision not available, skipping %s.", task_name)
            continue

        TrainerCls, default_lr = TASK_REGISTRY[task_name]

        if torch.cuda.is_available():
            task_batch_size = _GPU_BATCH_SIZES[task_name]
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
                    psilogic_preset=psilogic_preset,
                    tg_token=tg_token,
                    tg_chat=tg_chat,
                )

                trainer = None
                try:
                    trainer = TrainerCls(cfg)
                    result = trainer.run()
                    task_runs[opt_name].append(result)
                    all_results.append(result)

                    step_update = (
                        f"📊 <code>{task_name.upper()} | {opt_name.upper()} | "
                        f"Run {run_id + 1}/{n_runs}</code>\n"
                        f"Loss: <code>{result.final_loss:.4f}</code>  "
                        f"{result.metric_name}: <code>{result.final_metric:.4f}</code>\n"
                        f"VRAM: <code>{result.peak_vram_gb:.2f} GB</code>  "
                        f"Tput: <code>{result.throughput:.1f} samp/s</code>"
                    )
                    send_telegram_msg(tg_token, tg_chat, step_update)

                except Exception as exc:
                    log.error("Run %s/%s failed: %s", task_name, opt_name, exc)
                    log.error(traceback.format_exc())
                    send_telegram_msg(
                        tg_token,
                        tg_chat,
                        f"❌ <b>Error:</b> {task_name}/{opt_name} run{run_id}\n"
                        f"<code>{str(exc)[:300]}</code>",
                    )

                finally:
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


# ── Entry point (Jupyter-safe; CLI parsing only under __main__) ───────────────

BENCHMARK_CONFIG: dict[str, Any] = dict(
    tasks=["bert", "vit", "gpt2"],
    optimizers=["adamw", "lion", "psilogic"],
    n_runs=2,
    total_steps=1000,
    batch_size=32,
    accum_steps=1,
    amp_dtype="bf16",
    compile_model=False,
    profile=False,
    output_dir=Path("./results"),
    psilogic_preset="task",
)


def main(**overrides: Any) -> Optional[tuple[list[AggResult], list[RunResult]]]:
    """
    Entry point — works in both Jupyter and terminal.
    In Jupyter: main() or main(tasks=["vit"], psilogic_preset="vit")
    In terminal: python benchmark/run_benchmark.py --preset vit
    """
    cfg = dict(BENCHMARK_CONFIG)
    cfg.update(overrides)

    # `--preset vit` shortcut: restrict to the ViT arena unless tasks were
    # explicitly chosen, and route psilogic through PsiLogicViT param groups.
    if cfg.get("psilogic_preset") == "vit" and "tasks" not in overrides:
        cfg["tasks"] = ["vit"]

    output_dir = Path(cfg["output_dir"])

    if not torch.cuda.is_available():
        log.warning("No CUDA found. Scaling batch_size=4 for CPU debug mode.")
        cfg["batch_size"] = 4

    all_agg, all_results = run_benchmark(**cfg)

    if not all_agg:
        log.error("No results produced. Check errors above.")
        return None

    tg_token, tg_chat = tg_credentials()
    final_table = format_table(all_agg, title="🏆 FINAL BENCHMARK RESULTS")
    send_telegram_msg(tg_token, tg_chat, final_table)
    print(
        "\n"
        + final_table.replace("<b>", "")
        .replace("</b>", "")
        .replace("<code>", "")
        .replace("</code>", "")
    )

    save_json([asdict(a) for a in all_agg], output_dir / "benchmark_results.json")

    zip_path = output_dir / "full_benchmark_archive.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(output_dir):
            for file in files:
                if file != "full_benchmark_archive.zip":
                    full_path = os.path.join(root, file)
                    zipf.write(full_path, os.path.relpath(full_path, output_dir))

    send_telegram_file(
        tg_token,
        tg_chat,
        zip_path,
        "📂 Full benchmark archive (JSON + checkpoints)",
    )

    log.info("✅ Benchmarking complete. Results saved to %s", output_dir)
    return all_agg, all_results


def _parse_cli() -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="PsiLogic optimizer benchmark suite")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_REGISTRY),
        help="Arenas to run (default: bert vit gpt2)",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=sorted(OPTIMIZER_REGISTRY),
        help="Optimizers to compare (default: adamw lion psilogic)",
    )
    parser.add_argument("--runs", type=int, help="Seeded runs per combination")
    parser.add_argument("--steps", type=int, help="Training steps per run")
    parser.add_argument("--batch-size", type=int, help="CPU-mode batch size cap")
    parser.add_argument("--accum-steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16"], help="AMP dtype")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--output-dir", type=Path, help="Results directory")
    parser.add_argument(
        "--preset",
        choices=["task", "vit", "auto"],
        dest="psilogic_preset",
        help="PsiLogic config source: per-task presets, PsiLogicViT param groups, or PsiLogic.auto",
    )
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    mapping = {
        "tasks": args.tasks,
        "optimizers": args.optimizers,
        "n_runs": args.runs,
        "total_steps": args.steps,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "amp_dtype": args.amp_dtype,
        "output_dir": args.output_dir,
        "psilogic_preset": args.psilogic_preset,
    }
    for key, value in mapping.items():
        if value is not None:
            overrides[key] = value
    if args.compile:
        overrides["compile_model"] = True
    return overrides


if __name__ == "__main__":
    main(**_parse_cli())
