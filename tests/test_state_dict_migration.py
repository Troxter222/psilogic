"""state_dict schema versioning and v0.3-monolith migration tests."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic.optimizer import _STATE_DICT_SCHEMA_KEY, _STATE_DICT_SCHEMA_VERSION


def _warmed_up_optimizer() -> tuple[nn.Module, PsiLogic, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = nn.Linear(16, 4)
    opt = PsiLogic(model.parameters(), lr=1e-2)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    for _ in range(3):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
    return model, opt, x, y


def test_state_dict_is_tagged_with_schema_version():
    _, opt, _, _ = _warmed_up_optimizer()
    sd = opt.state_dict()
    assert sd[_STATE_DICT_SCHEMA_KEY] == _STATE_DICT_SCHEMA_VERSION


def test_round_trip_preserves_schema_handling():
    model, opt, x, y = _warmed_up_optimizer()
    # deepcopy emulates a torch.save/torch.load round trip (fresh tensors)
    sd = copy.deepcopy(opt.state_dict())

    model2 = nn.Linear(16, 4)
    opt2 = PsiLogic(model2.parameters(), lr=1e-2)
    model2.load_state_dict(model.state_dict())
    opt2.load_state_dict(sd)

    crit = nn.CrossEntropyLoss()
    for m, o in ((model, opt), (model2, opt2)):
        o.zero_grad()
        crit(m(x), y).backward()
        o.step()

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


def test_v1_checkpoint_without_schema_key_loads():
    """A v0.3-monolith checkpoint has no schema tag and lacks newer group keys."""
    model, opt, _, _ = _warmed_up_optimizer()
    sd = opt.state_dict()
    sd.pop(_STATE_DICT_SCHEMA_KEY)
    for group in sd["param_groups"]:
        group.pop("gamma_auto", None)  # key did not exist in v0.3

    model2 = nn.Linear(16, 4)
    opt2 = PsiLogic(model2.parameters(), lr=1e-2)
    opt2.load_state_dict(sd)

    for group in opt2.param_groups:
        assert group["gamma_auto"] is False

    crit = nn.CrossEntropyLoss()
    opt2.zero_grad()
    crit(model2(torch.randn(8, 16)), torch.randint(0, 4, (8,))).backward()
    opt2.step()  # must not raise


def test_v1_checkpoint_with_missing_chaos_state_loads():
    model, opt, _, _ = _warmed_up_optimizer()
    sd = opt.state_dict()
    sd.pop(_STATE_DICT_SCHEMA_KEY)
    for param_state in sd["state"].values():
        param_state.pop("gn_avg", None)

    model2 = nn.Linear(16, 4)
    opt2 = PsiLogic(model2.parameters(), lr=1e-2)
    opt2.load_state_dict(sd)

    for p in model2.parameters():
        assert "gn_avg" in opt2.state[p]

    crit = nn.CrossEntropyLoss()
    opt2.zero_grad()
    crit(model2(torch.randn(8, 16)), torch.randint(0, 4, (8,))).backward()
    opt2.step()
    for p in model2.parameters():
        assert not torch.isnan(p).any()


def test_future_schema_is_rejected():
    model, opt, _, _ = _warmed_up_optimizer()
    sd = opt.state_dict()
    sd[_STATE_DICT_SCHEMA_KEY] = _STATE_DICT_SCHEMA_VERSION + 1

    opt2 = PsiLogic(nn.Linear(16, 4).parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="schema"):
        opt2.load_state_dict(sd)
