"""ViT parameter-group split and PsiLogicViT convenience tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from psilogic import PsiLogic, PsiLogicViT, vision_defaults, vit_param_groups
from tests.toy_models import ToyViT


def _group_of(groups: list[dict], param: nn.Parameter) -> dict:
    for group in groups:
        if any(p is param for p in group["params"]):
            return group
    raise AssertionError("Parameter not found in any group")


def _named(model: nn.Module, name: str) -> nn.Parameter:
    return dict(model.named_parameters())[name]


class TestVitParamGroups:
    def test_full_coverage_no_duplicates(self):
        model = ToyViT()
        groups = vit_param_groups(model, lr=1e-3)
        grouped = [p for g in groups for p in g["params"]]
        assert len(grouped) == len(set(map(id, grouped))), "Duplicate params across groups"
        assert len(grouped) == sum(1 for _ in model.parameters())

    def test_patch_embed_gets_minimal_gamma(self):
        model = ToyViT()
        groups = vit_param_groups(model, lr=1e-3)
        group = _group_of(groups, _named(model, "patch_embed.weight"))
        assert group["gamma"] == 0.005

    def test_tokens_get_minimal_gamma_and_no_decay(self):
        model = ToyViT()
        groups = vit_param_groups(model, lr=1e-3)
        for name in ("cls_token", "pos_embed"):
            group = _group_of(groups, _named(model, name))
            assert group["gamma"] == 0.005
            assert group["weight_decay"] == 0.0

    def test_attention_gamma(self):
        model = ToyViT()
        groups = vit_param_groups(model, lr=1e-3)
        for name in ("attn_qkv.weight", "attn_out_proj.weight"):
            assert _group_of(groups, _named(model, name))["gamma"] == 0.02

    def test_mlp_gamma(self):
        model = ToyViT()
        groups = vit_param_groups(model, lr=1e-3)
        for name in ("mlp_fc1.weight", "mlp_fc2.weight"):
            assert _group_of(groups, _named(model, name))["gamma"] == 0.03

    def test_norm_and_bias_no_decay(self):
        model = ToyViT()
        groups = vit_param_groups(model, lr=1e-3)
        for name in ("norm1.weight", "norm1.bias", "attn_qkv.bias", "head.bias"):
            assert _group_of(groups, _named(model, name))["weight_decay"] == 0.0

    def test_lion_blocks_split(self):
        """attention/MLP/head run Lion, patch embed and tokens stay on Adam."""
        model = ToyViT()
        groups = vit_param_groups(model, lr=1e-3, lion_blocks=True)
        assert _group_of(groups, _named(model, "attn_qkv.weight"))["lion_mode"] is True
        assert _group_of(groups, _named(model, "mlp_fc1.weight"))["lion_mode"] is True
        assert _group_of(groups, _named(model, "head.weight"))["lion_mode"] is True
        assert _group_of(groups, _named(model, "patch_embed.weight"))["lion_mode"] is False
        assert _group_of(groups, _named(model, "cls_token"))["lion_mode"] is False

    def test_optimizer_accepts_groups(self):
        model = ToyViT()
        groups = vit_param_groups(model, lr=1e-3)
        opt = PsiLogic(groups, **vision_defaults(100))
        assert len(opt.param_groups) == len(groups)


class TestPsiLogicViTConvenience:
    def test_module_builds_param_groups(self):
        opt = PsiLogicViT(ToyViT(), lr=1e-3)
        assert len(opt.param_groups) > 1
        gammas = {g["gamma"] for g in opt.param_groups}
        assert 0.005 in gammas and 0.02 in gammas and 0.03 in gammas

    def test_plain_params_still_accepted(self):
        model = ToyViT()
        opt = PsiLogicViT(model.parameters(), lr=1e-3)
        assert len(opt.param_groups) == 1
        assert opt.param_groups[0]["gamma"] == 0.04

    def test_training_loss_decreases(self):
        torch.manual_seed(0)
        model = ToyViT()
        opt = PsiLogicViT(model, lr=1e-2, chaos_warmup=0)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(8, 3, 8, 8)
        y = torch.randint(0, 10, (8,))

        with torch.no_grad():
            initial = crit(model(x), y).item()
        for _ in range(15):
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        assert crit(model(x), y).item() < initial

    def test_lion_blocks_training_step(self):
        torch.manual_seed(0)
        model = ToyViT()
        opt = PsiLogicViT(model, lr=1e-3, lion_blocks=True, chaos_warmup=0)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(4, 3, 8, 8)
        y = torch.randint(0, 10, (4,))
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
        for p in model.parameters():
            assert not torch.isnan(p).any()
