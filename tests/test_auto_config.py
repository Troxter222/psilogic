"""Zero-config `PsiLogic.auto` and architecture inference tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic, infer_architecture
from tests.toy_models import ToyCNN, ToyGPT, ToyViT


class TestInferArchitecture:
    def test_vit(self):
        assert infer_architecture(ToyViT()) == "vit"

    def test_gpt(self):
        assert infer_architecture(ToyGPT()) == "gpt"

    def test_cnn(self):
        assert infer_architecture(ToyCNN()) == "vision"

    def test_generic_mlp(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
        assert infer_architecture(model) == "generic"

    def test_nlp_encoder(self):
        class ToyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(100, 16)
                self.attention = nn.MultiheadAttention(16, 2, batch_first=True)
                self.classifier = nn.Linear(16, 2)

            def forward(self, idx):
                x = self.embeddings(idx)
                x, _ = self.attention(x, x, x)
                return self.classifier(x.mean(dim=1))

        assert infer_architecture(ToyEncoder()) == "nlp"


class TestAuto:
    def test_vision_preset_applied(self):
        opt = PsiLogic.auto(ToyCNN())
        assert isinstance(opt, PsiLogic)
        assert opt.defaults["gamma"] == 0.04
        assert opt.defaults["tau_scale"] == 2.5

    def test_gpt_gets_param_groups(self):
        opt = PsiLogic.auto(ToyGPT(), total_steps=1_000)
        assert len(opt.param_groups) > 1
        assert any(g["gamma"] == 0.005 for g in opt.param_groups)
        assert opt.defaults["gamma_T_max"] == 1_000

    def test_vit_gets_param_groups(self):
        opt = PsiLogic.auto(ToyViT())
        gammas = {g["gamma"] for g in opt.param_groups}
        assert 0.005 in gammas and 0.02 in gammas

    def test_overrides_respected_on_ungrouped_archs(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
        opt = PsiLogic.auto(model, lr=5e-3, gamma=0.07)
        assert opt.param_groups[0]["lr"] == 5e-3
        assert opt.param_groups[0]["gamma"] == 0.07

    def test_rejects_non_module(self):
        with pytest.raises(TypeError, match="nn.Module"):
            PsiLogic.auto([nn.Parameter(torch.randn(4))])

    def test_generic_model_trains(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
        opt = PsiLogic.auto(model, lr=1e-2)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(8, 16)
        y = torch.randint(0, 4, (8,))

        with torch.no_grad():
            initial = crit(model(x), y).item()
        for _ in range(10):
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        assert crit(model(x), y).item() < initial

    def test_gpt_model_trains(self):
        torch.manual_seed(0)
        model = ToyGPT()
        opt = PsiLogic.auto(model, lr=1e-2, chaos_warmup=0)
        crit = nn.CrossEntropyLoss()
        idx = torch.randint(0, 64, (4, 16))
        targets = torch.randint(0, 64, (4, 16))
        opt.zero_grad()
        logits = model(idx)
        crit(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).backward()
        opt.step()
        for p in model.parameters():
            assert not torch.isnan(p).any()
