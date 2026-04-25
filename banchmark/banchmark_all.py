"""
benchmark_all.py
================
Fair, deterministic, and reproducible benchmark for three optimizers:
    Adam  |  AdamW  |  PsiLogic

Task 1: CIFAR-10  — custom ResNet-18 (3x3 conv1, no maxpool), 15 epochs, 10 seeds
Task 2: nanoGPT   — Tiny Shakespeare (char-level), 2000 steps, 5 seeds

Requirements: torch >= 2.0, torchvision, tqdm
Recommended GPU: RTX 3090 / A5000 / A100 (see notes at the bottom of this file)
"""

import os, sys, math, random, time, urllib.request, urllib.error
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torch.optim import Adam, AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.optimizer import Optimizer


# ════════════════════════════════════════════════════════════════════════════
#  PsiLogicV4Fast — self-contained copy (no external file dependency)
# ════════════════════════════════════════════════════════════════════════════

class PsiLogicV4Fast(Optimizer):
    """ΨLogic v4 — minimal CPU/GPU synchronization, clean CUDA path."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4,
                 gamma=0.05, p_ext=1.2, quantum_decay=1e-3, eps=1e-8,
                 grad_centralize=True, chaos_tau=0.3, gamma_T_max=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        gamma=gamma, p_ext=p_ext, quantum_decay=quantum_decay,
                        eps=eps, grad_centralize=grad_centralize,
                        chaos_tau=chaos_tau, gamma_T_max=gamma_T_max)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            wd           = group["weight_decay"]
            gamma        = group["gamma"]
            p_ext        = group["p_ext"]
            qd           = group["quantum_decay"]
            eps          = group["eps"]
            gc           = group["grad_centralize"]
            tau          = group["chaos_tau"]
            T_max        = group["gamma_T_max"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                dev = p.device

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

                if wd:
                    p.mul_(1.0 - lr * wd)

                st["m"].mul_(beta1).add_(g, alpha=1.0 - beta1)
                st["v"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                gn = g.norm() / math.sqrt(g.numel())
                if t == 1:
                    st["fast"].copy_(gn); st["slow"].copy_(gn)
                else:
                    st["fast"].mul_(0.9).add_(gn, alpha=0.1)
                    st["slow"].mul_(0.99).add_(gn, alpha=0.01)

                slow, fast = st["slow"], st["fast"]

                if T_max > 0:
                    cos_w  = 0.5 * (1.0 + math.cos(math.pi * min(t / T_max, 1.0)))
                    g_base = gamma * cos_w
                else:
                    g_base = gamma

                ratio   = fast / (slow + eps)
                chaos   = torch.tanh(slow) * (1.0 + 0.5 * torch.tanh(torch.relu(ratio - 1.0)))
                c_coeff = torch.where(slow < tau,
                                      torch.zeros(1, device=dev, dtype=p.dtype),
                                      chaos * (lr * g_base * p_ext))
                p.mul_(1.0 - c_coeff)

                if qd:
                    qd_w = torch.where(slow >= tau,
                                       torch.full((1,), lr * qd, device=dev, dtype=p.dtype),
                                       torch.zeros(1, device=dev, dtype=p.dtype))
                    p.mul_(1.0 - qd_w * torch.tanh(g.abs()))

                step_size  = lr / (1.0 - beta1 ** t)
                bias_corr2 = math.sqrt(1.0 - beta2 ** t)
                denom = st["v"].sqrt().div_(bias_corr2).add_(eps)
                p.addcdiv_(st["m"], denom, value=-step_size)

        return loss


# ════════════════════════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════════════════════════

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


def seed_everything(seed: int):
    """Set all random seeds for fully deterministic, reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def make_optimizer(name: str, params, lr: float, steps_total: int):
    """Optimizer factory — identical hyperparameters where applicable."""
    wd = 1e-4
    if name == "Adam":
        return Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif name == "AdamW":
        return AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    elif name == "PsiLogic":
        return PsiLogicV4Fast(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd,
                              gamma=0.05, p_ext=1.2, quantum_decay=1e-3,
                              eps=1e-8, grad_centralize=True, chaos_tau=0.3,
                              gamma_T_max=steps_total)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ════════════════════════════════════════════════════════════════════════════
#  Task 1 — CIFAR-10 + custom ResNet-18
# ════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for 32x32 images.

    Key differences from the ImageNet architecture:
    - 3x3 initial convolution with stride=1 instead of 7x7 / stride-2
    - No MaxPool after the stem
    - Identity skip on the first residual block
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_cifar10_loaders(batch_size=128):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader


def run_cifar10(opt_name: str, seed: int, epochs: int = 15):
    seed_everything(seed)
    train_loader, val_loader = get_cifar10_loaders()

    model = ResNet18CIFAR().to(DEVICE)
    lr    = 1e-3
    steps = epochs * len(train_loader)
    opt   = make_optimizer(opt_name, model.parameters(), lr, steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler    = GradScaler(enabled=(DEVICE.type == "cuda"))
    criterion = nn.CrossEntropyLoss()
    history   = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running_loss = 0.0; n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(DEVICE.type == "cuda")):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            running_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        train_loss = running_loss / n

        # Validate
        model.eval()
        val_loss = 0.0; correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                with autocast(enabled=(DEVICE.type == "cuda")):
                    logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                correct  += (logits.argmax(1) == yb).sum().item()
                total    += xb.size(0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss / total)
        history["val_acc"].append(correct / total * 100.0)
        print(f"  [CIFAR-10][{opt_name}] seed={seed} "
              f"ep={epoch:02d}/{epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={history['val_loss'][-1]:.4f}  "
              f"val_acc={history['val_acc'][-1]:.2f}%", flush=True)

    return {k: v[-1] for k, v in history.items()}


# ════════════════════════════════════════════════════════════════════════════
#  Task 2 — nanoGPT (char-level) + Tiny Shakespeare
# ════════════════════════════════════════════════════════════════════════════

SHAKESPEARE_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = DATA_DIR / "tiny_shakespeare.txt"


def download_shakespeare():
    if SHAKESPEARE_PATH.exists():
        return
    print("Downloading Tiny Shakespeare...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data       = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def get_shakespeare_loaders(block_size: int = 128, batch_size: int = 64):
    download_shakespeare()
    text   = SHAKESPEARE_PATH.read_text(encoding="utf-8")
    chars  = sorted(set(text))
    stoi   = {c: i for i, c in enumerate(chars)}
    vocab  = len(chars)
    data   = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n      = int(0.9 * len(data))
    train_ds = CharDataset(data[:n], block_size)
    val_ds   = CharDataset(data[n:], block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=True)
    return train_loader, val_loader, vocab


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln2(x))
        x = x + self.ff(self.ln1(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x    = self.blocks(x)
        x    = self.ln_f(x)
        logits = self.head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def run_nanogpt(opt_name: str, seed: int, total_steps: int = 2000,
                block_size: int = 128, batch_size: int = 64):
    seed_everything(seed)
    train_loader, val_loader, vocab = get_shakespeare_loaders(block_size, batch_size)

    model = NanoGPT(vocab_size=vocab, block_size=block_size).to(DEVICE)
    lr    = 3e-4
    opt   = make_optimizer(opt_name, model.parameters(), lr, total_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    train_iter      = iter(train_loader)
    step            = 0
    best_val_loss   = float("inf")
    last_train_loss = float("nan")

    while step < total_steps:
        model.train()
        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=(DEVICE.type == "cuda")):
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        sched.step()
        last_train_loss = loss.item()
        step += 1

        # Validate every 200 steps
        if step % 200 == 0 or step == total_steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i, (xv, yv) in enumerate(val_loader):
                    if i >= 50: break
                    xv, yv = xv.to(DEVICE, non_blocking=True), yv.to(DEVICE, non_blocking=True)
                    with autocast(enabled=(DEVICE.type == "cuda")):
                        _, vl = model(xv, yv)
                    val_losses.append(vl.item())
            val_loss = float(np.mean(val_losses))
            best_val_loss = min(best_val_loss, val_loss)
            print(f"  [nanoGPT][{opt_name}] seed={seed}  "
                  f"step={step:04d}/{total_steps}  "
                  f"train_loss={last_train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}", flush=True)

    # Final validation over more batches
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i, (xv, yv) in enumerate(val_loader):
            if i >= 100: break
            xv, yv = xv.to(DEVICE, non_blocking=True), yv.to(DEVICE, non_blocking=True)
            with autocast(enabled=(DEVICE.type == "cuda")):
                _, vl = model(xv, yv)
            val_losses.append(vl.item())
    final_val = float(np.mean(val_losses))

    return {"train_loss": last_train_loss, "val_loss": final_val}


# ════════════════════════════════════════════════════════════════════════════
#  Results table
# ════════════════════════════════════════════════════════════════════════════

def fmt(vals):
    a = np.mean(vals); s = np.std(vals)
    return f"{a:.4f} +/- {s:.4f}"


def print_table(cifar_results, gpt_results, opt_names):
    """Print a formatted ASCII summary table to stdout."""
    sep  = "-" * 90
    sep2 = "=" * 90

    print("\n")
    print(sep2)
    print("  BENCHMARK RESULTS  --  Mean +/- Std across seeds")
    print(sep2)

    print(f"\n  {'TASK 1 -- CIFAR-10 (ResNet-18)':^86}")
    print(f"  {'15 epochs -- 10 seeds':^86}")
    print(f"  {sep}")
    print(f"  {'Optimizer':<14} {'Train Loss':>18} {'Val Loss':>20} {'Val Acc (%)':>22}")
    print(f"  {sep}")
    for opt in opt_names:
        r  = cifar_results[opt]
        tl = fmt([x["train_loss"] for x in r])
        vl = fmt([x["val_loss"]   for x in r])
        va = fmt([x["val_acc"]    for x in r])
        print(f"  {opt:<14} {tl:>18} {vl:>20} {va:>22}")
    print(f"  {sep}")

    print(f"\n  {'TASK 2 -- nanoGPT / Tiny Shakespeare (char-level)':^86}")
    print(f"  {'2000 steps -- 5 seeds':^86}")
    print(f"  {sep}")
    print(f"  {'Optimizer':<14} {'Train Loss':>18} {'Val Loss':>20}")
    print(f"  {sep}")
    for opt in opt_names:
        r  = gpt_results[opt]
        tl = fmt([x["train_loss"] for x in r])
        vl = fmt([x["val_loss"]   for x in r])
        print(f"  {opt:<14} {tl:>18} {vl:>20}")
    print(f"  {sep}")
    print(sep2)
    print()


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  benchmark_all.py")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    OPT_NAMES   = ["Adam", "AdamW", "PsiLogic"]
    CIFAR_SEEDS = list(range(10))   # seeds 0-9
    GPT_SEEDS   = list(range(5))    # seeds 0-4

    cifar_results = defaultdict(list)
    gpt_results   = defaultdict(list)

    # Task 1: CIFAR-10
    print("Starting Task 1: CIFAR-10\n")
    t0 = time.time()
    for seed in CIFAR_SEEDS:
        print(f"\n-- Seed {seed} -----------------------------------------")
        for opt_name in OPT_NAMES:
            res = run_cifar10(opt_name, seed=seed, epochs=15)
            cifar_results[opt_name].append(res)
    print(f"\nTask 1 done in {(time.time()-t0)/60:.1f} min\n")

    # Task 2: nanoGPT / Tiny Shakespeare
    print("Starting Task 2: nanoGPT / Tiny Shakespeare\n")
    t1 = time.time()
    for seed in GPT_SEEDS:
        print(f"\n-- Seed {seed} -----------------------------------------")
        for opt_name in OPT_NAMES:
            res = run_nanogpt(opt_name, seed=seed, total_steps=2000)
            gpt_results[opt_name].append(res)
    print(f"\nTask 2 done in {(time.time()-t1)/60:.1f} min\n")

    print_table(cifar_results, gpt_results, OPT_NAMES)

    total_min = (time.time() - t0) / 60
    print(f"  Total wall time: {total_min:.1f} min\n")


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════════════════
#
#  Cloud / RunPod setup notes
#  --------------------------
#  Recommended GPUs (priority order):
#    1. RTX 3090 (24 GB) -- best value, 24 GB VRAM, fast
#    2. RTX A5000 (24 GB) -- professional grade, slightly more reliable
#    3. A100 40/80 GB -- maximum throughput, ~3x faster than RTX 3090
#
#  Estimated wall time (RTX 3090):
#    Task 1 (CIFAR-10):  ~3 min/run x 3 optimizers x 10 seeds = ~90 min
#    Task 2 (nanoGPT):   ~4 min/run x 3 optimizers x 5 seeds  = ~60 min
#    Total:              ~2.5 hours
#    On A100: ~50-60 minutes total.
#
#  How to run on RunPod:
#    1. Select the "PyTorch 2.x" template (includes torch + torchvision)
#    2. Upload this file via rsync, scp, or the Jupyter file browser
#    3. Run: python benchmark_all.py | tee results.log
#
#  Install dependencies (if needed):
#    pip install torch torchvision tqdm --quiet
#
# ════════════════════════════════════════════════════════════════════════════