"""FSDP integration smoke test (requires 2+ CUDA devices)."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from psilogic import PsiLogic

pytestmark = pytest.mark.multi_gpu


def _setup(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _cleanup() -> None:
    dist.destroy_process_group()


def _demo_fsdp(rank: int, world_size: int) -> None:
    _setup(rank, world_size)
    torch.manual_seed(42 + rank)
    torch.cuda.set_device(rank)

    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)).cuda(rank)
    fsdp_model = FSDP(model, device_id=rank)
    loss_fn = nn.MSELoss()
    optimizer = PsiLogic(fsdp_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True)

    for _ in range(3):
        optimizer.zero_grad()
        outputs = fsdp_model(torch.randn(4, 10, device=f"cuda:{rank}"))
        labels = torch.randn(4, 2, device=f"cuda:{rank}")
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    _cleanup()


def test_fsdp() -> None:
    world_size = 2
    mp.spawn(_demo_fsdp, args=(world_size,), nprocs=world_size, join=True)
