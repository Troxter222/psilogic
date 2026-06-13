"""GPT parameter-group split and PsiLogicGPT convenience tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from psilogic import PsiLogic, PsiLogicGPT, gpt_param_groups, gpt_scratch_defaults
from tests.toy_models import ToyGPT


def _group_of(groups: list[dict], param: nn.Parameter) -> dict:
    for group in groups:
        if any(p is param for p in group["params"]):
            return group
    raise AssertionError("Parameter not found in any group")


def _named(model: nn.Module, name: str) -> nn.Parameter:
    return dict(model.named_parameters())[name]


class TestGptParamGroups:
    def test_full_coverage_no_duplicates(self):
        model = ToyGPT(tied=False)
        groups = gpt_param_groups(model, lr=3e-4)
        grouped = [p for g in groups for p in g["params"]]
        assert len(grouped) == len(set(map(id, grouped)))
        assert len(grouped) == sum(1 for _ in model.parameters())

    def test_embeddings_minimal_gamma_no_quantum_decay(self):
        model = ToyGPT()
        groups = gpt_param_groups(model, lr=3e-4)
        for name in ("wte.weight", "wpe.weight"):
            group = _group_of(groups, _named(model, name))
            assert group["gamma"] == 0.005
            assert group["quantum_decay"] == 0.0

    def test_blocks_gamma(self):
        model = ToyGPT()
        groups = gpt_param_groups(model, lr=3e-4)
        for name in ("h.0.c_attn.weight", "h.0.mlp_c_fc.weight"):
            assert _group_of(groups, _named(model, name))["gamma"] == 0.02

    def test_lm_head_gamma_when_untied(self):
        model = ToyGPT(tied=False)
        groups = gpt_param_groups(model, lr=3e-4)
        assert _group_of(groups, _named(model, "lm_head.weight"))["gamma"] == 0.01

    def test_tied_lm_head_lands_in_embedding_group_once(self):
        model = ToyGPT(tied=True)
        groups = gpt_param_groups(model, lr=3e-4)
        tied_weight = model.lm_head.weight
        owners = [g for g in groups if any(p is tied_weight for p in g["params"])]
        assert len(owners) == 1
        assert owners[0]["gamma"] == 0.005

    def test_norm_and_bias_no_decay(self):
        model = ToyGPT()
        groups = gpt_param_groups(model, lr=3e-4)
        for name in ("h.0.ln_1.weight", "h.0.c_attn.bias", "ln_f.bias"):
            assert _group_of(groups, _named(model, name))["weight_decay"] == 0.0

    def test_optimizer_accepts_groups(self):
        model = ToyGPT()
        groups = gpt_param_groups(model, lr=3e-4)
        opt = PsiLogic(groups, **gpt_scratch_defaults(1000))
        assert len(opt.param_groups) == len(groups)


class TestPsiLogicGPTConvenience:
    def test_module_builds_param_groups(self):
        opt = PsiLogicGPT(ToyGPT(), lr=3e-4)
        assert len(opt.param_groups) > 1
        gammas = {g["gamma"] for g in opt.param_groups}
        assert 0.005 in gammas and 0.02 in gammas

    def test_plain_params_still_accepted(self):
        model = ToyGPT()
        opt = PsiLogicGPT(model.parameters(), lr=3e-4)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["gamma"] == 0.02

    def test_training_loss_decreases(self):
        torch.manual_seed(0)
        model = ToyGPT()
        opt = PsiLogicGPT(model, lr=1e-2, chaos_warmup=0)
        crit = nn.CrossEntropyLoss()
        idx = torch.randint(0, 64, (4, 16))
        targets = torch.randint(0, 64, (4, 16))

        def loss_fn() -> torch.Tensor:
            logits = model(idx)
            return crit(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        with torch.no_grad():
            initial = loss_fn().item()
        for _ in range(15):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        assert loss_fn().item() < initial
