"""Shared pytest fixtures and collection hooks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip GPU-marked tests when the required hardware is unavailable."""
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))
        if "multi_gpu" in item.keywords and torch.cuda.device_count() < 2:
            item.add_marker(pytest.mark.skip(reason="2+ CUDA devices required"))


@pytest.fixture
def simple_model() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(16, 4)


@pytest.fixture
def simple_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    return x, y


@pytest.fixture
def criterion() -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss()
