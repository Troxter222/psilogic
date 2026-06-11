"""DDP integration smoke test (CPU-safe via gloo backend)."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from psilogic import PsiLogic


def _setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _cleanup() -> None:
    dist.destroy_process_group()


def _demo_basic(rank: int, world_size: int) -> None:
    _setup(rank, world_size)
    torch.manual_seed(42 + rank)

    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
    ddp_model = nn.parallel.DistributedDataParallel(model)
    loss_fn = nn.MSELoss()
    optimizer = PsiLogic(ddp_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=False)

    for _ in range(3):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(4, 10))
        labels = torch.randn(4, 2)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    _cleanup()


def test_ddp() -> None:
    world_size = 2
    mp.spawn(_demo_basic, args=(world_size,), nprocs=world_size, join=True)
