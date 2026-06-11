"""Stress tests for gradient explosion and max_cancel clamping."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic

pytestmark = pytest.mark.gpu


def test_gradient_explosion() -> None:
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2)).cuda()
    opt = PsiLogic(model.parameters(), lr=1e-3, chaos_warmup=0, chaos_tau=0.01, max_cancel=0.05)
    crit = nn.CrossEntropyLoss()

    x = torch.randn(4, 10, device="cuda")
    y = torch.randint(0, 2, (4,), device="cuda")

    for _ in range(5):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

    pre_spike_norm = next(model.parameters()).norm().item()

    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(1000.0)
    opt.step()

    post_spike_norm = next(model.parameters()).norm().item()
    weights = next(model.parameters())

    assert not torch.isnan(weights).any(), "Weights became NaN after explosion"
    assert not torch.isinf(weights).any(), "Weights became Inf after explosion"

    shrinkage = pre_spike_norm / post_spike_norm
    assert shrinkage < 1.10, f"Weights collapsed too much (shrinkage={shrinkage:.4f})"
