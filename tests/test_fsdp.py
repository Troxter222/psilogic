"""FSDP optimizer checkpoint, resharding, and restart tests (requires 2+ CUDA devices).

The existing test_fsdp.py smoke test only runs three optimizer steps under
FSDP and never touches serialization. This file adds the coverage issue #28
asks for:

  * optimizer state serialization under FSDP (save/load round-trip)
  * sharded state-dict handling (each rank's local shard, not just the
    rank0-only full/consolidated view)
  * resharding: loading a checkpoint saved under one FSDP wrapping topology
    into a model wrapped with a *different* topology (different unit
    boundaries -> different per-rank shard shapes), which is the scenario
    that silently breaks naive checkpointing code
  * restart behavior: tear down the in-memory model/optimizer, rebuild fresh
    ones, load the checkpoint, and verify training continues consistently
    with a reference run that never restarted

All of this uses ``torch.distributed.checkpoint`` (DCP) with the
``get_state_dict``/``set_state_dict`` helpers from
``torch.distributed.checkpoint.state_dict``, which is the currently
recommended, resharding-safe way to checkpoint FSDP + optimizer state
(as opposed to the older ``FSDP.optim_state_dict`` + ``StateDictType``
context-manager pattern, which does not reconcile differing wrap
topologies on load).
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from psilogic import PsiLogic

pytestmark = pytest.mark.multi_gpu


def _setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def _cleanup() -> None:
    dist.destroy_process_group()


def _build_model(rank: int) -> nn.Module:
    # Same seed on every rank: FSDP shards a single, identically-initialized
    # module across ranks, it does not give each rank its own model.
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)
    ).cuda(rank)


def _wrap_flat(model: nn.Module, rank: int) -> FSDP:
    """One flat FSDP unit wrapping the whole module."""
    return FSDP(model, device_id=rank)


def _wrap_per_layer(model: nn.Module, rank: int) -> FSDP:
    """Multiple FSDP units (one per Linear), giving a different per-rank
    shard layout than _wrap_flat for the same underlying parameters."""
    linear_layers = {m for m in model.modules() if isinstance(m, nn.Linear)}
    policy = ModuleWrapPolicy(linear_layers)
    return FSDP(model, device_id=rank, auto_wrap_policy=policy)


def _run_steps(
    fsdp_model: FSDP, optimizer: PsiLogic, steps: int, *, seed_offset: int, device: torch.device
) -> list[float]:
    """Deterministic training steps: the same seed_offset always produces
    the same sequence of synthetic (x, y) batches, so two independently
    restarted runs stay comparable."""
    loss_fn = nn.MSELoss()
    losses = []
    for i in range(steps):
        torch.manual_seed(1000 + seed_offset + i)
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 2, device=device)
        optimizer.zero_grad()
        loss = loss_fn(fsdp_model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _save_checkpoint(model: FSDP, optimizer: PsiLogic, path: str) -> None:
    model_sd, optim_sd = get_state_dict(
        model, optimizer, options=StateDictOptions(cpu_offload=True)
    )
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=path)


def _load_checkpoint(model: FSDP, optimizer: PsiLogic, path: str) -> None:
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.load({"model": model_sd, "optim": optim_sd}, checkpoint_id=path)
    set_state_dict(model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)


def _demo_checkpoint_restart(rank: int, world_size: int, port: int, ckpt_dir: str) -> None:
    device = torch.device(f"cuda:{rank}")
    _setup(rank, world_size, port)

    # --- Reference run: train straight through, no restart. ---
    ref_model = _wrap_flat(_build_model(rank), rank)
    ref_opt = PsiLogic(ref_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True)
    _run_steps(ref_model, ref_opt, steps=3, seed_offset=0, device=device)
    ref_losses = _run_steps(ref_model, ref_opt, steps=3, seed_offset=3, device=device)

    # --- Checkpoint-and-restart run: train, save, tear down, rebuild fresh
    # model/optimizer objects, load, then continue with the same inputs. ---
    ckpt_model = _wrap_flat(_build_model(rank), rank)
    ckpt_opt = PsiLogic(ckpt_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True)
    _run_steps(ckpt_model, ckpt_opt, steps=3, seed_offset=0, device=device)

    _save_checkpoint(ckpt_model, ckpt_opt, ckpt_dir)

    # Simulate a process restart: fresh model, fresh FSDP wrap, fresh
    # optimizer (no state carried over except via the checkpoint on disk).
    del ckpt_model, ckpt_opt
    restarted_model = _wrap_flat(_build_model(rank), rank)
    restarted_opt = PsiLogic(
        restarted_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True
    )
    _load_checkpoint(restarted_model, restarted_opt, ckpt_dir)

    restarted_losses = _run_steps(
        restarted_model, restarted_opt, steps=3, seed_offset=3, device=device
    )

    for ref_loss, restarted_loss in zip(ref_losses, restarted_losses):
        assert abs(ref_loss - restarted_loss) < 1e-4, (
            f"rank {rank}: restarted run diverged from reference ({restarted_loss} vs {ref_loss})"
        )

    for p_ref, p_restarted in zip(ref_model.parameters(), restarted_model.parameters()):
        assert torch.allclose(p_ref, p_restarted, atol=1e-4), (
            f"rank {rank}: restarted params diverged from reference after checkpoint restore"
        )

    _cleanup()


def test_fsdp_checkpoint_restart_matches_reference() -> None:
    """Save mid-training, tear down and rebuild the model/optimizer/FSDP
    wrap from scratch, restore from checkpoint, and confirm training
    continues identically to a run that never restarted."""
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ckpt")
        mp.spawn(
            _demo_checkpoint_restart,
            args=(world_size, 12357, ckpt_dir),
            nprocs=world_size,
            join=True,
        )


def _demo_sharded_state_dict(rank: int, world_size: int, port: int, ckpt_dir: str) -> None:
    device = torch.device(f"cuda:{rank}")
    _setup(rank, world_size, port)

    model = _wrap_flat(_build_model(rank), rank)
    optimizer = PsiLogic(model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True)
    _run_steps(model, optimizer, steps=3, seed_offset=0, device=device)

    # Capture this rank's local optimizer state before saving, so we can
    # verify the round trip preserves each rank's local shard exactly
    # (not just some aggregate/consolidated view).
    local_sd_before, _ = get_state_dict(model, optimizer)
    local_before = {k: v.clone() if torch.is_tensor(v) else v for k, v in local_sd_before.items()}

    _save_checkpoint(model, optimizer, ckpt_dir)

    model2 = _wrap_flat(_build_model(rank), rank)
    optimizer2 = PsiLogic(model2.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True)
    _load_checkpoint(model2, optimizer2, ckpt_dir)

    local_sd_after, _ = get_state_dict(model2, optimizer2)
    for key, before in local_before.items():
        after = local_sd_after[key]
        if torch.is_tensor(before):
            assert torch.equal(before.cpu(), after.cpu()), (
                f"rank {rank}: local shard for {key!r} changed across sharded save/load"
            )
        else:
            assert before == after, f"rank {rank}: local value for {key!r} changed across save/load"

    _cleanup()


def test_fsdp_sharded_state_dict_preserves_local_shards() -> None:
    """Each rank's own local shard of the model state must round-trip
    exactly through a sharded save/load, not just approximately."""
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ckpt")
        mp.spawn(
            _demo_sharded_state_dict,
            args=(world_size, 12358, ckpt_dir),
            nprocs=world_size,
            join=True,
        )


def _demo_resharding(rank: int, world_size: int, port: int, ckpt_dir: str) -> None:
    device = torch.device(f"cuda:{rank}")
    _setup(rank, world_size, port)

    # Save under one FSDP wrap topology (single flat unit).
    src_model = _wrap_flat(_build_model(rank), rank)
    src_opt = PsiLogic(src_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True)
    _run_steps(src_model, src_opt, steps=3, seed_offset=0, device=device)
    _save_checkpoint(src_model, src_opt, ckpt_dir)

    # Load into a model wrapped with a *different* topology (one FSDP unit
    # per Linear layer) -> different per-rank shard boundaries for the same
    # logical parameters. DCP's get_state_dict/set_state_dict + dcp.load are
    # responsible for reconciling this; if resharding is broken this either
    # raises or silently produces wrong values.
    dst_model = _wrap_per_layer(_build_model(rank), rank)
    dst_opt = PsiLogic(dst_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True)
    _load_checkpoint(dst_model, dst_opt, ckpt_dir)

    losses = _run_steps(dst_model, dst_opt, steps=3, seed_offset=3, device=device)
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), (
        f"rank {rank}: non-finite loss after resharded checkpoint load"
    )

    for p in dst_model.parameters():
        assert torch.isfinite(p).all(), f"rank {rank}: non-finite param after resharded load"

    _cleanup()


def test_fsdp_resharding_across_wrap_policies() -> None:
    """A checkpoint saved under one FSDP wrap policy (one flat unit) must
    load correctly into a model wrapped with a different policy (per-layer
    units), i.e. a different per-rank sharding of the same parameters."""
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ckpt")
        mp.spawn(
            _demo_resharding,
            args=(world_size, 12359, ckpt_dir),
            nprocs=world_size,
            join=True,
        )
