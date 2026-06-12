"""Gradient accumulation correctness tests."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from psilogic import PsiLogic


def test_accumulated_microbatches_match_full_batch():
    """k micro-batches with loss/k must produce the same step as one full batch."""
    torch.manual_seed(0)
    model_full = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4))
    model_accum = copy.deepcopy(model_full)
    x = torch.randn(8, 8)
    y = torch.randint(0, 4, (8,))
    crit = nn.CrossEntropyLoss()

    opt_full = PsiLogic(model_full.parameters(), lr=1e-2, chaos_warmup=0)
    opt_accum = PsiLogic(model_accum.parameters(), lr=1e-2, chaos_warmup=0)

    opt_full.zero_grad()
    crit(model_full(x), y).backward()
    opt_full.step()

    accum_steps = 4
    micro = 8 // accum_steps
    opt_accum.zero_grad()
    for i in range(accum_steps):
        xb = x[i * micro : (i + 1) * micro]
        yb = y[i * micro : (i + 1) * micro]
        (crit(model_accum(xb), yb) / accum_steps).backward()
    opt_accum.step()

    for p_full, p_accum in zip(model_full.parameters(), model_accum.parameters()):
        assert torch.allclose(p_full, p_accum, atol=1e-6), (
            "Gradient accumulation diverged from the full-batch step"
        )


def test_training_with_accumulation_converges():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    opt = PsiLogic(model.parameters(), lr=1e-2, chaos_warmup=0)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(16, 8)
    y = torch.randint(0, 4, (16,))

    with torch.no_grad():
        initial = crit(model(x), y).item()

    accum_steps = 4
    for _ in range(10):
        opt.zero_grad()
        for i in range(accum_steps):
            xb = x[i * 4 : (i + 1) * 4]
            yb = y[i * 4 : (i + 1) * 4]
            (crit(model(xb), yb) / accum_steps).backward()
        opt.step()

    assert crit(model(x), y).item() < initial


def test_chaos_state_advances_once_per_optimizer_step():
    """Accumulation must not advance the chaos clock per micro-batch."""
    torch.manual_seed(0)
    model = nn.Linear(8, 2)
    opt = PsiLogic(model.parameters(), lr=1e-2)
    crit = nn.MSELoss()
    x = torch.randn(8, 8)
    y = torch.randn(8, 2)

    for _ in range(3):
        opt.zero_grad()
        for i in range(4):
            (crit(model(x[i * 2 : (i + 1) * 2]), y[i * 2 : (i + 1) * 2]) / 4).backward()
        opt.step()

    for p in model.parameters():
        assert opt.state[p]["t"] == 3
