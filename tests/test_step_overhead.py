"""Step-time overhead vs AdamW and the built-in step profiler."""

from __future__ import annotations

import statistics
import time
from typing import Callable

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic._cuda import is_fused_available


def _median_step_seconds(
    optimizer_factory: Callable[..., torch.optim.Optimizer],
    device: str = "cpu",
    n_timed: int = 30,
    n_warmup: int = 5,
) -> float:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)).to(device)
    opt = optimizer_factory(model.parameters())
    crit = nn.MSELoss()
    x = torch.randn(32, 256, device=device)
    y = torch.randn(32, 256, device=device)

    times: list[float] = []
    for i in range(n_warmup + n_timed):
        opt.zero_grad()
        crit(model(x), y).backward()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= n_warmup:
            times.append(elapsed)
    return statistics.median(times)


def test_cpu_overhead_is_bounded():
    """CPU scalar path sanity bound (the <=15% roadmap target is the GPU
    foreach path vs fused AdamW; the pure-python loop is naturally slower)."""
    adamw = _median_step_seconds(lambda p: torch.optim.AdamW(p, lr=1e-3))
    psi = _median_step_seconds(lambda p: PsiLogic(p, lr=1e-3, chaos_warmup=0))
    assert psi < adamw * 20.0, (
        f"PsiLogic CPU step is {psi / adamw:.1f}x AdamW — something regressed badly"
    )


@pytest.mark.gpu
def test_gpu_foreach_overhead():
    """Foreach path must stay in the same ballpark as fused AdamW.

    Documented target is <=15% on A100; the CI assertion is deliberately
    looser (2.5x) to stay robust on shared/noisy runners.
    """
    adamw = _median_step_seconds(
        lambda p: torch.optim.AdamW(p, lr=1e-3, foreach=True), device="cuda"
    )
    psi = _median_step_seconds(
        lambda p: PsiLogic(p, lr=1e-3, chaos_warmup=0, use_foreach=True, use_fused_cuda=False),
        device="cuda",
    )
    assert psi < adamw * 2.5, f"PsiLogic foreach GPU step is {psi / adamw:.2f}x AdamW"


@pytest.mark.gpu
@pytest.mark.skipif(not is_fused_available(), reason="Triton fused CUDA path unavailable")
def test_gpu_fused_overhead():
    """Fused Triton path should be much closer to AdamW than the foreach path."""
    adamw = _median_step_seconds(
        lambda p: torch.optim.AdamW(p, lr=1e-3, foreach=True), device="cuda"
    )
    psi = _median_step_seconds(
        lambda p: PsiLogic(p, lr=1e-3, chaos_warmup=0, use_fused_cuda=True),
        device="cuda",
    )
    assert psi < adamw * 1.25, f"PsiLogic fused GPU step is {psi / adamw:.2f}x AdamW (target <=1.25x)"


def test_profile_step_time_records_metrics():
    torch.manual_seed(0)
    model = nn.Linear(16, 4)
    opt = PsiLogic(model.parameters(), lr=1e-3, profile_step_time=True)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))

    assert opt.step_time_ms_ema is None
    for _ in range(5):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()

    assert opt.last_step_time_ms > 0.0
    assert opt.step_time_ms_ema is not None and opt.step_time_ms_ema > 0.0


def test_profiling_disabled_by_default():
    model = nn.Linear(4, 2)
    opt = PsiLogic(model.parameters(), lr=1e-3)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()
    assert opt.last_step_time_ms == 0.0
    assert opt.step_time_ms_ema is None
