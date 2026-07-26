"""Regression tests for the `.item()`-sync removal in the fused CUDA step.

Two separate concerns are checked, deliberately kept apart:

1. `test_fused_group_step_no_device_sync` — the *performance* claim: the
   fused step must not force a GPU->CPU sync anywhere in the decay/quantum
   path. This test would have failed against the pre-fix code (it called
   `.item()` twice per parameter per step in the branches exercised here).

2. `test_fused_matches_scalar_*` — the *correctness* claim: replacing the
   Python-float kernel arguments with 1-element device tensors must not
   change a single computed value. These compare the fused path against
   `_step_scalar` bit-for-bit-ish (tight tolerance, fp32) across the three
   branches that touch the values which used to go through `.item()`:
   pure chaos decay, weight-decay-only decay, and quantum decay.

Both require CUDA + Triton; they're skipped everywhere else.
"""

from __future__ import annotations

import copy
import contextlib
from typing import Any

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic._cuda import is_fused_available


pytestmark = pytest.mark.gpu


def _tiny_model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))


def _step_n(opt: PsiLogic, model: nn.Module, n_steps: int, dtype: torch.dtype = torch.float32) -> None:
    x = torch.randn(8, 32, device="cuda", dtype=dtype)
    y = torch.randn(8, 16, device="cuda", dtype=dtype)
    crit = nn.MSELoss()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()


@contextlib.contextmanager
def _count_item_calls():
    """Count calls to `torch.Tensor.item()` for the duration of the block."""
    counter = {"n": 0}
    orig_item = torch.Tensor.item

    def _counting_item(self, *args, **kwargs):
        counter["n"] += 1
        return orig_item(self, *args, **kwargs)

    torch.Tensor.item = _counting_item
    try:
        yield counter
    finally:
        torch.Tensor.item = orig_item


@pytest.mark.skipif(not is_fused_available(), reason="Triton fused CUDA path unavailable")
@pytest.mark.parametrize(
    "kwargs",
    [
        # Pure chaos decay branch (total_scalar_decay computed from chaos_contrib).
        dict(lr=1e-3, gamma=0.05, chaos_warmup=0, quantum_decay=0.0, weight_decay=0.0),
        # Weight-decay-only branch (chaos_gain == 0 -> wd_only_decay path).
        dict(lr=1e-3, gamma=0.0, chaos_warmup=0, quantum_decay=0.0, weight_decay=1e-2),
        # Quantum decay branch (apply_quantum=True -> qd_contrib used).
        dict(lr=1e-3, gamma=0.05, chaos_warmup=0, quantum_decay=2e-4, weight_decay=1e-4),
        # Lion mode, still routes through the same decay kernel args.
        dict(lr=1e-3, gamma=0.03, chaos_warmup=0, quantum_decay=0.0, weight_decay=0.0, lion_mode=True),
    ],
    ids=["chaos_decay", "wd_only", "quantum_decay", "lion"],
)
def test_fused_group_step_no_device_sync(kwargs: dict[str, Any]) -> None:
    """The fused path must not call `.item()` while computing decay scalars.

    Before the fix, `total_scalar_decay` and `qd_contrib` were pulled to the
    host with `.item()` inside `fused_param_step` on every single parameter,
    every single step — a blocking GPU sync each time. This test fails loudly
    against the old code (nonzero count) and passes against the fix (zero).
    """
    model = _tiny_model().to("cuda")
    opt = PsiLogic(model.parameters(), use_fused_cuda=True, use_foreach=False, **kwargs)

    # One warmup step outside the counted region: state init, Triton kernel
    # JIT-compilation, and CUDA context warmup all touch `.item()` in ways
    # unrelated to what we're testing (e.g. `int(...)` on Python ints isn't
    # an issue, but a cold Triton autotune pass might probe device props).
    _step_n(opt, model, n_steps=1)
    torch.cuda.synchronize()

    with _count_item_calls() as counter:
        _step_n(opt, model, n_steps=5)
        torch.cuda.synchronize()

    assert counter["n"] == 0, (
        f"fused_group_step triggered {counter['n']} `.item()` call(s) — "
        "a GPU/CPU sync leaked back into the decay path."
    )


@pytest.mark.skipif(not is_fused_available(), reason="Triton fused CUDA path unavailable")
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(lr=1e-3, gamma=0.05, chaos_warmup=0, quantum_decay=0.0, weight_decay=0.0),
        dict(lr=1e-3, gamma=0.0, chaos_warmup=0, quantum_decay=0.0, weight_decay=1e-2),
        dict(lr=1e-3, gamma=0.05, chaos_warmup=0, quantum_decay=2e-4, weight_decay=1e-4),
    ],
    ids=["chaos_decay", "wd_only", "quantum_decay"],
)
def test_fused_matches_scalar_after_sync_removal(kwargs: dict[str, Any]) -> None:
    """The tensor-pointer kernel args must produce identical results to
    the scalar reference path — same branches as the sync test above,
    checked for numerical equivalence rather than sync count.
    """
    torch.manual_seed(42)
    model_ref = _tiny_model().to("cuda")
    model_fused = copy.deepcopy(model_ref)

    opt_ref = PsiLogic(model_ref.parameters(), use_foreach=False, use_fused_cuda=False, **kwargs)
    opt_fused = PsiLogic(model_fused.parameters(), use_foreach=False, use_fused_cuda=True, **kwargs)

    x = torch.randn(8, 32, device="cuda")
    y = torch.randn(8, 16, device="cuda")
    crit = nn.MSELoss()

    for _ in range(20):
        opt_ref.zero_grad()
        opt_fused.zero_grad()
        loss_ref = crit(model_ref(x), y)
        loss_ref.backward()
        # Reuse model_ref's grads for model_fused: two independent backward
        # passes would each hit nondeterministic CUDA reduction order, which
        # is a separate (already-known, already-tolerated) source of drift
        # unrelated to what this test checks.
        for p_ref, p_fused in zip(model_ref.parameters(), model_fused.parameters()):
            p_fused.grad = p_ref.grad.clone()
        opt_ref.step()
        opt_fused.step()

    for p_ref, p_fused in zip(model_ref.parameters(), model_fused.parameters()):
        torch.testing.assert_close(p_ref, p_fused, rtol=1e-5, atol=1e-6)

    for p_ref, p_fused in zip(model_ref.parameters(), model_fused.parameters()):
        state_ref = opt_ref.state[p_ref]
        state_fused = opt_fused.state[p_fused]
        torch.testing.assert_close(state_ref["m"], state_fused["m"], rtol=1e-5, atol=1e-6)
        if "v" in state_ref:
            torch.testing.assert_close(state_ref["v"], state_fused["v"], rtol=1e-5, atol=1e-6)


def test_zero_scalar_cache_dtype_and_device() -> None:
    """Sanity check on the new `_zero_scalar` helper — no GPU required.

    Just confirms the cached zero tensor has the dtype/device the kernel
    expects, and that repeated calls for the same device return a tensor
    with the same value (whether or not it's the literal same object is an
    implementation detail we don't pin down here).
    """
    from psilogic._cuda.step import _zero_scalar

    device = torch.device("cpu")
    z1 = _zero_scalar(device)
    z2 = _zero_scalar(device)
    assert z1.dtype == torch.float32
    assert z1.device == device
    assert z1.shape == (1,)
    assert float(z1.item()) == 0.0
    assert float(z2.item()) == 0.0
