"""torch.compile(fullgraph=True) compatibility smoke test."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic


def _torch_version_tuple() -> tuple[int, int]:
    major, minor, *_rest = torch.__version__.split("+")[0].split(".")
    return int(major), int(minor)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
@pytest.mark.skipif(
    _torch_version_tuple() < (2, 3),
    reason="Dynamo cannot read Parameter.grad under fullgraph on torch < 2.3",
)
def test_torch_compile_fullgraph() -> None:
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = PsiLogic(model.parameters(), lr=1e-3, use_foreach=True)
    criterion = nn.MSELoss()

    def train_step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = criterion(model(x), y)
        loss.backward()
        return loss

    def opt_step() -> None:
        optimizer.step()

    compiled_opt_step = torch.compile(opt_step, fullgraph=True)

    x = torch.randn(4, 10)
    y = torch.randn(4, 2)
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()

    optimizer.zero_grad()
    try:
        train_step(x, y)
        compiled_opt_step()
    except Exception as exc:
        pytest.fail(f"torch.compile(fullgraph=True) failed: {exc}")

    for _ in range(3):
        optimizer.zero_grad()
        train_step(x, y)
        compiled_opt_step()
