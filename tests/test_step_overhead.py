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
    model_factory: Callable[[], nn.Module] | None = None,
    batch_shape: tuple[int, ...] = (32, 256),
) -> float:
    torch.manual_seed(0)
    if model_factory is None:
        model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)).to(device)
        x = torch.randn(*batch_shape, device=device)
        y = torch.randn(batch_shape[0], 256, device=device)
        crit: nn.Module = nn.MSELoss()
    else:
        model = model_factory().to(device)
        x = torch.randn(32, 64, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        crit = nn.CrossEntropyLoss()
    opt = optimizer_factory(model.parameters())

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


def _cuda_is_ampere_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(0)
    return major >= 8


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
    looser to stay robust on shared/noisy runners and consumer GPUs
    (GTX 16xx / etc.), where kernel launch overhead dominates.
    """
    adamw = _median_step_seconds(
        lambda p: torch.optim.AdamW(p, lr=1e-3, foreach=True), device="cuda"
    )
    psi = _median_step_seconds(
        lambda p: PsiLogic(p, lr=1e-3, chaos_warmup=0, use_foreach=True, use_fused_cuda=False),
        device="cuda",
    )
    # Ampere+ (sm>=8): 2.5x. Pre-Ampere consumer cards: 4x (launch-bound).
    limit = 2.5 if _cuda_is_ampere_or_newer() else 4.0
    assert psi < adamw * limit, f"PsiLogic foreach GPU step is {psi / adamw:.2f}x AdamW"


@pytest.mark.gpu
@pytest.mark.skipif(not is_fused_available(), reason="Triton fused CUDA path unavailable")
@pytest.mark.skipif(
    not _cuda_is_ampere_or_newer(),
    reason="Fused <=1.25x AdamW target is for Ampere+ (A100/H100); skip on older GPUs",
)
def test_gpu_fused_overhead():
    """Fused Triton path should be much closer to AdamW than the foreach path."""
    adamw = _median_step_seconds(
        lambda p: torch.optim.AdamW(p, lr=1e-3, foreach=True), device="cuda"
    )
    psi = _median_step_seconds(
        lambda p: PsiLogic(p, lr=1e-3, chaos_warmup=0, use_fused_cuda=True),
        device="cuda",
    )
    assert psi < adamw * 1.25, (
        f"PsiLogic fused GPU step is {psi / adamw:.2f}x AdamW (target <=1.25x)"
    )


class _TinyViTLike(nn.Module):
    def __init__(self, depth: int = 12, dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU()) for _ in range(depth)]
        )
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return self.head(x)


@pytest.mark.gpu
@pytest.mark.skipif(not is_fused_available(), reason="Triton fused CUDA path unavailable")
@pytest.mark.skipif(
    not _cuda_is_ampere_or_newer(),
    reason="Fused <=1.25x AdamW target is for Ampere+ (A100/H100); skip on older GPUs",
)
def test_gpu_fused_vit_like_overhead():
    """ViT-like many-small-tensor model catches launch-bound fused regressions."""
    torch.manual_seed(0)
    model = _TinyViTLike().cuda()
    crit = nn.CrossEntropyLoss()
    x = torch.randn(32, 128, device="cuda")
    y = torch.randint(0, 10, (32,), device="cuda")

    def _median(factory: Callable[..., torch.optim.Optimizer]) -> float:
        opt = factory(model.parameters())
        times: list[float] = []
        for i in range(35):
            opt.zero_grad(set_to_none=True)
            crit(model(x), y).backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            if i >= 5:
                times.append(time.perf_counter() - t0)
        return statistics.median(times)

    adamw = _median(lambda p: torch.optim.AdamW(p, lr=3e-4, foreach=True))
    psi = _median(
        lambda p: PsiLogic(
            p,
            lr=3e-4,
            gamma=0.04,
            chaos_warmup=0,
            use_fused_cuda=True,
        )
    )
    assert psi < adamw * 1.25, (
        f"PsiLogic fused ViT-like step is {psi / adamw:.2f}x AdamW (target <=1.25x)"
    )


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
