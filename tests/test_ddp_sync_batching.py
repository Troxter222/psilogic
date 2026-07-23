"""Tests for issue #27: batch DDP chaos synchronization into one packed
collective per parameter group per optimizer step, instead of one
all-reduce per parameter.

No real process group is needed: ``_maybe_sync_chaos`` is a thin wrapper
around ``torch.distributed.all_reduce``, so we fake ``torch.distributed``
just enough to (a) count how many collectives are issued, and (b) verify
that packing several parameters' chaos state into one buffer and reducing
it once produces the exact same per-parameter result as reducing each
parameter individually.
"""

from __future__ import annotations

from unittest import mock

import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic._cuda import fused_group_step


def _model(n_layers: int = 3) -> nn.Sequential:
    torch.manual_seed(0)
    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(8, 8), nn.ReLU()]
    layers.append(nn.Linear(8, 2))
    return nn.Sequential(*layers)


def _count_sync_calls(opt: PsiLogic) -> dict[str, int]:
    """Patch opt._maybe_sync_chaos to count calls while preserving behavior."""
    counter = {"n": 0}
    original = opt._maybe_sync_chaos

    def counting(states):
        counter["n"] += 1
        return original(states)

    opt._maybe_sync_chaos = counting
    return counter


def test_scalar_path_issues_one_sync_per_group() -> None:
    model = _model()
    n_params = len(list(model.parameters()))
    assert n_params > 1  # otherwise this test wouldn't distinguish anything

    opt = PsiLogic(model.parameters(), lr=1e-3, sync_chaos_ddp=True, use_foreach=False)
    x, y = torch.randn(4, 8), torch.randn(4, 2)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    counter = _count_sync_calls(opt)
    opt.step()

    assert counter["n"] == 1, (
        f"expected exactly one batched sync call for {n_params} params, got {counter['n']}"
    )


def test_fused_path_issues_one_sync_per_group() -> None:
    model = _model()
    n_params = len(list(model.parameters()))
    assert n_params > 1

    opt = PsiLogic(
        model.parameters(), lr=1e-3, sync_chaos_ddp=True, use_fused_cuda=True, use_foreach=False
    )
    x, y = torch.randn(4, 8), torch.randn(4, 2)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    counter = _count_sync_calls(opt)
    group = opt.param_groups[0]
    with torch.no_grad():
        opt._step_fused_cuda(group)

    assert counter["n"] == 1, (
        f"expected exactly one batched sync call for {n_params} params, got {counter['n']}"
    )
    for p in model.parameters():
        assert torch.isfinite(p).all()


def _fake_all_reduce_add_offset(offset: float):
    """A fake collective standing in for a real multi-rank all-reduce: adds
    a fixed offset to whatever buffer it's given, in place. Since it's
    linear and elementwise, applying it once to a packed buffer must equal
    applying it separately to each slice — which is exactly the property
    batching relies on.
    """

    def _fn(tensor, op=None):
        tensor.add_(offset)

    return _fn


def test_batched_sync_matches_per_param_sync_numerically() -> None:
    """The whole point of batching: packing N params' chaos state into one
    buffer and reducing it once must give bit-identical results to reducing
    each param's state individually, for any linear reduction.
    """
    model_batched = _model()
    model_individual = type(model_batched)(*list(model_batched))
    model_individual.load_state_dict(model_batched.state_dict())

    opt_batched = PsiLogic(
        model_batched.parameters(), lr=1e-3, sync_chaos_ddp=True, use_foreach=False
    )
    opt_individual = PsiLogic(
        model_individual.parameters(), lr=1e-3, sync_chaos_ddp=True, use_foreach=False
    )

    x, y = torch.randn(4, 8), torch.randn(4, 2)
    for opt, model in ((opt_batched, model_batched), (opt_individual, model_individual)):
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

    with mock.patch("torch.distributed.is_available", return_value=True), mock.patch(
        "torch.distributed.is_initialized", return_value=True
    ), mock.patch("torch.distributed.get_world_size", return_value=1), mock.patch(
        "torch.distributed.all_reduce", side_effect=_fake_all_reduce_add_offset(0.01)
    ):
        # Batched: one _maybe_sync_chaos call covering every param (current behavior).
        opt_batched.step()

        # Individual: force one _maybe_sync_chaos call per param, mimicking
        # the pre-fix per-parameter sync, by monkeypatching to call the
        # original per-state as it's produced.
        original_sync = opt_individual._maybe_sync_chaos

        def per_param_sync(states):
            for s in states:
                original_sync([s])

        opt_individual._maybe_sync_chaos = per_param_sync
        opt_individual.step()

    for p_batched, p_individual in zip(model_batched.parameters(), model_individual.parameters()):
        assert torch.allclose(p_batched, p_individual, atol=1e-6), (
            "batched vs per-parameter DDP sync produced different results"
        )


def test_fused_group_step_matches_scalar_step_call_pattern() -> None:
    """fused_group_step should also call the sync callback exactly once,
    regardless of parameter count, mirroring _step_scalar's behavior.
    """
    model = _model()
    n_params = len(list(model.parameters()))
    opt = PsiLogic(model.parameters(), lr=1e-3, sync_chaos_ddp=True, use_foreach=False)
    x, y = torch.randn(4, 8), torch.randn(4, 2)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    calls = []

    def recording_sync(states):
        calls.append(len(states))

    group = opt.param_groups[0]
    with torch.no_grad():
        fused_group_step(group, opt.state, sync_chaos_ddp=True, maybe_sync=recording_sync)

    assert calls == [n_params], (
        f"expected a single sync call covering all {n_params} params, got {calls}"
    )
