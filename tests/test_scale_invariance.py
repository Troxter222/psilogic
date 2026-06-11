"""Scale-invariance of the chaos detector EMAs."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic

pytestmark = pytest.mark.gpu


def test_scale_invariance() -> None:
    torch.manual_seed(42)
    model_normal = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)).cuda()
    model_scaled = copy.deepcopy(model_normal)

    opt_normal = PsiLogic(
        model_normal.parameters(), lr=1e-3, chaos_warmup=0, chaos_tau=0.01, agc_clip=0.0
    )
    opt_scaled = PsiLogic(
        model_scaled.parameters(), lr=1e-3, chaos_warmup=0, chaos_tau=0.01, agc_clip=0.0
    )

    criterion = nn.MSELoss()
    x = torch.randn(4, 10, device="cuda")
    y = torch.randn(4, 2, device="cuda")
    scale = 100.0

    for _ in range(5):
        opt_normal.zero_grad()
        loss_normal = criterion(model_normal(x), y)
        loss_normal.backward()
        opt_normal.step()

        opt_scaled.zero_grad()
        loss_scaled = criterion(model_scaled(x), y) * scale
        loss_scaled.backward()
        opt_scaled.step()

    p_normal = next(model_normal.parameters())
    p_scaled = next(model_scaled.parameters())
    state_normal = opt_normal.state[p_normal]
    state_scaled = opt_scaled.state[p_scaled]

    assert torch.allclose(state_normal["slow"], state_scaled["slow"], atol=1e-4), (
        "Slow EMA should be scale-invariant"
    )
    assert torch.allclose(state_normal["fast"], state_scaled["fast"], atol=1e-4), (
        "Fast EMA should be scale-invariant"
    )
