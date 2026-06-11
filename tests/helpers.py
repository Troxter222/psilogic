"""Shared test utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


def run_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    n: int = 10,
) -> tuple[float, float]:
    """Run ``n`` gradient steps and return ``(initial_loss, final_loss)``."""
    with torch.no_grad():
        initial_loss = criterion(model(x), y).item()
    for _ in range(n):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final_loss = criterion(model(x), y).item()
    return initial_loss, final_loss
