"""DeepSpeed ZeRO-1 smoke test (GPU-only, optional deepspeed extra)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic

pytest.importorskip("deepspeed")
pytestmark = pytest.mark.gpu


def test_deepspeed_zero1_smoke() -> None:
    """PsiLogic must run under DeepSpeed ZeRO-1 without NaNs or exceptions."""
    import deepspeed

    ds_config = {
        "train_batch_size": 4,
        "zero_optimization": {"stage": 1},
        "fp16": {"enabled": False},
    }
    torch.manual_seed(0)
    model = nn.Linear(8, 4).cuda()
    optimizer = PsiLogic(model.parameters(), lr=1e-3)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    x = torch.randn(4, 8, device="cuda")
    y = torch.randn(4, 4, device="cuda")
    crit = nn.MSELoss()

    for _ in range(5):
        engine.zero_grad()
        loss = crit(engine(x), y)
        engine.backward(loss)
        engine.step()

        assert torch.isfinite(loss), "loss became non-finite under DeepSpeed ZeRO-1"
