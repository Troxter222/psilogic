"""Tests for numerically safe optimizer state on low-precision parameters.

The AMP test in test_amp.py only exercises FP32 parameters under autocast,
where the optimizer never actually sees FP16 tensors. This file covers the
case the issue describes directly: parameters whose own dtype is FP16/BF16,
which previously received FP16/BF16 momentum, variance, and chaos state.
No GPU is required since the underlying bug and fix are dtype-based, not
device-based.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from psilogic import PsiLogic


def _tiny_model(dtype: torch.dtype) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)).to(dtype)


def test_state_is_fp32_for_fp16_params() -> None:
    model = _tiny_model(torch.float16)
    opt = PsiLogic(model.parameters(), lr=1e-3, use_foreach=False)

    x = torch.randn(4, 10, dtype=torch.float16)
    y = torch.randn(4, 2, dtype=torch.float16)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    opt.step()

    for p in model.parameters():
        state = opt.state[p]
        assert state["m"].dtype == torch.float32
        assert state["v"].dtype == torch.float32
        assert state["fast"].dtype == torch.float32
        assert state["slow"].dtype == torch.float32
        assert state["gn_avg"].dtype == torch.float32


def test_state_is_fp32_for_bf16_params() -> None:
    model = _tiny_model(torch.bfloat16)
    opt = PsiLogic(model.parameters(), lr=1e-3, use_foreach=False)

    x = torch.randn(4, 10, dtype=torch.bfloat16)
    y = torch.randn(4, 2, dtype=torch.bfloat16)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    opt.step()

    for p in model.parameters():
        state = opt.state[p]
        assert state["m"].dtype == torch.float32
        assert state["v"].dtype == torch.float32


def test_fp32_params_state_dtype_unchanged() -> None:
    """FP32 (and other non-low-precision) params keep prior behavior."""
    model = _tiny_model(torch.float32)
    opt = PsiLogic(model.parameters(), lr=1e-3, use_foreach=False)

    x = torch.randn(4, 10)
    y = torch.randn(4, 2)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    opt.step()

    for p in model.parameters():
        state = opt.state[p]
        assert state["m"].dtype == torch.float32
        assert state["v"].dtype == torch.float32


def test_fp16_training_runs_without_dtype_errors() -> None:
    """Regression test: chaos decay + Adam update must not raise on FP16.

    Exercises the code paths (unified decay, quantum decay, Adam update)
    that mix FP32 state with FP16 params, across enough steps to move past
    chaos warmup.
    """
    model = _tiny_model(torch.float16)
    opt = PsiLogic(
        model.parameters(),
        lr=1e-3,
        chaos_warmup=0,
        chaos_tau=0.01,
        quantum_decay=0.1,
        use_foreach=False,
    )
    x = torch.randn(4, 10, dtype=torch.float16)
    y = torch.randn(4, 2, dtype=torch.float16)
    criterion = nn.MSELoss()

    for _ in range(10):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_fp16_lion_mode_runs_without_dtype_errors() -> None:
    model = _tiny_model(torch.float16)
    opt = PsiLogic(
        model.parameters(),
        lr=1e-3,
        chaos_warmup=0,
        chaos_tau=0.01,
        lion_mode=True,
        use_foreach=False,
    )
    x = torch.randn(4, 10, dtype=torch.float16)
    y = torch.randn(4, 2, dtype=torch.float16)
    criterion = nn.MSELoss()

    for _ in range(10):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_checkpoint_roundtrip_preserves_fp32_state() -> None:
    """FP16-param state must still be FP32 after a save/load roundtrip,
    including the v1-schema migration path that backfills missing chaos
    state (fast/slow/gn_avg)."""
    model = _tiny_model(torch.float16)
    opt = PsiLogic(model.parameters(), lr=1e-3, use_foreach=False)
    x = torch.randn(4, 10, dtype=torch.float16)
    y = torch.randn(4, 2, dtype=torch.float16)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    opt.step()

    state_dict = opt.state_dict()
    # Simulate a pre-v0.4 (schema v1) checkpoint missing chaos state.
    for pstate in state_dict["state"].values():
        pstate.pop("fast", None)
        pstate.pop("slow", None)
        pstate.pop("gn_avg", None)
    state_dict.pop("psilogic_schema", None)

    model2 = copy.deepcopy(model)
    opt2 = PsiLogic(model2.parameters(), lr=1e-3, use_foreach=False)
    opt2.load_state_dict(state_dict)

    for p in model2.parameters():
        state = opt2.state[p]
        assert state["fast"].dtype == torch.float32
        assert state["slow"].dtype == torch.float32
        assert state["gn_avg"].dtype == torch.float32

    # And a further step must still run without raising a dtype error.
    opt2.zero_grad()
    loss2 = nn.MSELoss()(model2(x), y)
    loss2.backward()
    opt2.step()
