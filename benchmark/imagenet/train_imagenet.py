"""
ImageNet-1k training harness — DDP-ready, bf16 AMP, cosine LR.

Reference benchmarks CV-1 (ResNet-50) and CV-2 (ViT-Base) from the roadmap.

Single GPU:
    python benchmark/imagenet/train_imagenet.py \
        --data-dir /datasets/imagenet --model resnet50 --optimizer psilogic

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 benchmark/imagenet/train_imagenet.py \
        --data-dir /datasets/imagenet --model resnet50 --optimizer psilogic

Expects the standard ImageFolder layout: <data-dir>/train and <data-dir>/val.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from psilogic import PsiLogic, PsiLogicViT, vision_defaults

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("imagenet")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class ImageNetConfig:
    data_dir: Path = Path("/datasets/imagenet")
    model: str = "resnet50"  # resnet50 | vit_b_16
    optimizer: str = "psilogic"  # adamw | lion | psilogic
    epochs: int = 90
    batch_size: int = 256  # per GPU
    lr: float = 1e-3  # peak LR (scaled by world size below)
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    seed: int = 42
    workers: int = 8
    amp_dtype: str = "bf16"  # bf16 | fp16 | off
    channels_last: bool = True
    output_dir: Path = Path("./results/imagenet")
    resume: Optional[Path] = None
    profile_step_time: bool = False
    log_interval: int = 50
    # populated at runtime
    rank: int = field(default=0, init=False)
    world_size: int = field(default=1, init=False)
    local_rank: int = field(default=0, init=False)


def parse_args() -> ImageNetConfig:
    parser = argparse.ArgumentParser(description="ImageNet-1k benchmark (DDP, bf16)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet50", "vit_b_16"], default="resnet50")
    parser.add_argument("--optimizer", choices=["adamw", "lion", "psilogic"], default="psilogic")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("./results/imagenet"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--profile-step-time", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    cfg = ImageNetConfig(
        data_dir=args.data_dir,
        model=args.model,
        optimizer=args.optimizer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        workers=args.workers,
        amp_dtype=args.amp_dtype,
        channels_last=not args.no_channels_last,
        output_dir=args.output_dir,
        resume=args.resume,
        profile_step_time=args.profile_step_time,
        log_interval=args.log_interval,
    )
    return cfg


# ── Distributed helpers ───────────────────────────────────────────────────────


def setup_distributed(cfg: ImageNetConfig) -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend="nccl")
        log.info("DDP rank %d/%d on cuda:%d", cfg.rank, cfg.world_size, cfg.local_rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(cfg: ImageNetConfig) -> bool:
    return cfg.rank == 0


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


# ── Lion baseline ─────────────────────────────────────────────────────────────


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
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                update = (beta1 * m + (1.0 - beta1) * p.grad).sign_()
                if group["weight_decay"] > 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
                m.mul_(beta2).add_(p.grad, alpha=1.0 - beta2)
        return loss


# ── Model / optimizer / data builders ────────────────────────────────────────


def build_model(cfg: ImageNetConfig) -> nn.Module:
    if cfg.model == "resnet50":
        model = torchvision.models.resnet50(weights=None, num_classes=1000)
    elif cfg.model == "vit_b_16":
        model = torchvision.models.vit_b_16(weights=None, num_classes=1000)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")
    return model


def build_optimizer(cfg: ImageNetConfig, model: nn.Module, total_steps: int) -> Optimizer:
    scaled_lr = cfg.lr * cfg.world_size

    if cfg.optimizer == "adamw":
        return AdamW(
            model.parameters(),
            lr=scaled_lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    if cfg.optimizer == "lion":
        return Lion(
            model.parameters(),
            lr=scaled_lr / 5.0,
            betas=(0.9, 0.99),
            weight_decay=cfg.weight_decay * 10.0,
        )
    if cfg.optimizer == "psilogic":
        if cfg.model == "vit_b_16":
            # Per-group gamma split: patch embed 0.005, attention 0.02, MLP 0.03
            return PsiLogicViT(
                model,
                lr=scaled_lr,
                gamma_T_max=total_steps,
                weight_decay=cfg.weight_decay,
                sync_chaos_ddp=cfg.world_size > 1,
                profile_step_time=cfg.profile_step_time,
            )
        defaults = vision_defaults(total_steps)
        defaults["weight_decay"] = cfg.weight_decay
        return PsiLogic(
            model.parameters(),
            lr=scaled_lr,
            sync_chaos_ddp=cfg.world_size > 1,
            profile_step_time=cfg.profile_step_time,
            **defaults,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_loaders(cfg: ImageNetConfig) -> tuple[DataLoader, DataLoader, DistributedSampler]:
    train_tf = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = torchvision.datasets.ImageFolder(str(cfg.data_dir / "train"), train_tf)
    val_ds = torchvision.datasets.ImageFolder(str(cfg.data_dir / "val"), val_tf)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=True, seed=cfg.seed
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=cfg.world_size, rank=cfg.rank, shuffle=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=val_sampler,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=cfg.workers > 0,
    )
    return train_loader, val_loader, train_sampler


def cosine_with_warmup(
    optimizer: Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Train / eval loops ────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    cfg: ImageNetConfig, model: nn.Module, val_loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    correct1 = torch.zeros(1, device=device)
    correct5 = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        if cfg.channels_last:
            images = images.to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        _, top5 = logits.topk(5, dim=1)
        match = top5.eq(labels.unsqueeze(1))
        correct1 += match[:, 0].sum()
        correct5 += match.any(dim=1).sum()
        total += labels.size(0)

    if dist.is_available() and dist.is_initialized():
        for tensor in (correct1, correct5, total):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    top1 = (correct1 / total.clamp(min=1)).item()
    top5_acc = (correct5 / total.clamp(min=1)).item()
    return top1, top5_acc


def train(cfg: ImageNetConfig) -> dict:
    setup_distributed(cfg)
    seed_everything(cfg.seed, cfg.rank)

    device = torch.device(f"cuda:{cfg.local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, train_sampler = build_loaders(cfg)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs

    model = build_model(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.world_size > 1:
        model = DDP(model, device_ids=[cfg.local_rank])

    optimizer = build_optimizer(cfg, model, total_steps)
    scheduler = cosine_with_warmup(optimizer, cfg.warmup_epochs * steps_per_epoch, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    amp_enabled = cfg.amp_dtype != "off" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch = 0
    best_top1 = 0.0
    if cfg.resume is not None and cfg.resume.exists():
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        (model.module if isinstance(model, DDP) else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_top1 = ckpt.get("best_top1", 0.0)
        log.info("Resumed from %s at epoch %d", cfg.resume, start_epoch)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    global_step = start_epoch * steps_per_epoch
    train_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_start = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            if cfg.channels_last:
                images = images.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            amp_ctx = (
                torch.amp.autocast("cuda", dtype=amp_dtype)
                if amp_enabled
                else contextlib.nullcontext()
            )
            with amp_ctx:
                loss = criterion(model(images), labels)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item() * labels.size(0)
            epoch_samples += labels.size(0)

            if is_main(cfg) and batch_idx % cfg.log_interval == 0:
                msg = (
                    f"epoch {epoch:3d}  step {batch_idx:5d}/{steps_per_epoch}  "
                    f"loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}"
                )
                if cfg.profile_step_time and isinstance(optimizer, PsiLogic):
                    msg += f"  step_ms {optimizer.step_time_ms_ema or 0.0:.2f}"
                log.info(msg)

        top1, top5 = evaluate(cfg, model, val_loader, device)
        epoch_time = time.perf_counter() - epoch_start
        best_top1 = max(best_top1, top1)

        if is_main(cfg):
            record = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(epoch_samples, 1),
                "top1": top1,
                "top5": top5,
                "epoch_time_sec": epoch_time,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(record)
            log.info(
                "epoch %3d done  top1=%.4f  top5=%.4f  best=%.4f  (%.1fs)",
                epoch,
                top1,
                top5,
                best_top1,
                epoch_time,
            )

            raw_model = model.module if isinstance(model, DDP) else model
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_top1": best_top1,
                    "config": {k: str(v) for k, v in asdict(cfg).items()},
                },
                cfg.output_dir / f"{cfg.model}_{cfg.optimizer}_last.pt",
            )
            with open(cfg.output_dir / f"{cfg.model}_{cfg.optimizer}_history.json", "w") as f:
                json.dump(history, f, indent=2)

    wall_time = time.perf_counter() - train_start
    result = {
        "model": cfg.model,
        "optimizer": cfg.optimizer,
        "epochs": cfg.epochs,
        "best_top1": best_top1,
        "final_top1": history[-1]["top1"] if history else 0.0,
        "wall_time_sec": wall_time,
        "world_size": cfg.world_size,
    }
    if is_main(cfg):
        log.info("Training complete: %s", result)
        with open(cfg.output_dir / f"{cfg.model}_{cfg.optimizer}_result.json", "w") as f:
            json.dump(result, f, indent=2)

    cleanup_distributed()
    return result


if __name__ == "__main__":
    train(parse_args())
