"""HuggingFace / Lightning integration tests (framework-optional)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from psilogic import PsiLogic
from psilogic.integrations.hf import create_psilogic_optimizer, psilogic_trainer_class
from psilogic.integrations.lightning import configure_psilogic
from tests.toy_models import ToyGPT, ToyViT


class ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(100, 16)
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)
        self.out_proj = nn.Linear(16, 16)
        self.classifier = nn.Linear(16, 2)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(idx)
        attn = torch.softmax(self.q_proj(x) @ self.k_proj(x).transpose(-2, -1) / 4.0, dim=-1)
        x = self.out_proj(attn @ self.v_proj(x))
        return self.classifier(x.mean(dim=1))


class TestCreatePsiLogicOptimizer:
    def test_explicit_preset_without_args(self):
        opt = create_psilogic_optimizer(ToyEncoder(), None, preset="nlp")
        assert isinstance(opt, PsiLogic)
        assert len(opt.param_groups) > 1

    def test_auto_preset_detects_gpt(self):
        opt = create_psilogic_optimizer(ToyGPT(), None)
        assert any(g["gamma"] == 0.005 for g in opt.param_groups)
        assert opt.defaults["tau_scale"] == 3.0  # gpt_scratch_defaults

    def test_auto_preset_detects_vit(self):
        opt = create_psilogic_optimizer(ToyViT(), None)
        assert opt.defaults["tau_scale"] == 2.5  # vision_defaults

    def test_training_args_namespace_respected(self):
        args = SimpleNamespace(learning_rate=5e-4, weight_decay=0.05, max_steps=1_000)
        opt = create_psilogic_optimizer(ToyEncoder(), args, preset="nlp")
        assert opt.defaults["gamma_T_max"] == 1_000
        assert all(g["lr"] == 5e-4 for g in opt.param_groups)
        decayed = [g for g in opt.param_groups if g["weight_decay"] > 0]
        assert decayed and all(g["weight_decay"] == 0.05 for g in decayed)

    def test_explicit_kwargs_beat_args(self):
        args = SimpleNamespace(learning_rate=5e-4, weight_decay=0.05, max_steps=1_000)
        opt = create_psilogic_optimizer(
            ToyEncoder(), args, preset="nlp", lr=1e-3, total_steps=500, gamma=0.07
        )
        assert opt.defaults["gamma_T_max"] == 500
        assert opt.defaults["gamma"] == 0.07
        assert all(g["lr"] == 1e-3 for g in opt.param_groups)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="preset"):
            create_psilogic_optimizer(ToyEncoder(), None, preset="banana")

    def test_optimizer_steps(self):
        torch.manual_seed(0)
        model = ToyEncoder()
        opt = create_psilogic_optimizer(model, None, preset="nlp")
        idx = torch.randint(0, 100, (4, 8))
        loss = nn.functional.cross_entropy(model(idx), torch.randint(0, 2, (4,)))
        loss.backward()
        opt.step()
        for p in model.parameters():
            assert not torch.isnan(p).any()


class TestLightning:
    def test_configure_psilogic(self):
        opt = configure_psilogic(ToyEncoder(), preset="nlp", lr=3e-4, total_steps=200)
        assert isinstance(opt, PsiLogic)
        assert opt.defaults["gamma_T_max"] == 200

    def test_chaos_monitor_importable_without_lightning(self):
        # Import always succeeds; construction requires lightning.
        from psilogic.integrations.lightning import (
            _LIGHTNING_AVAILABLE,
            ChaosMonitorCallback,
        )

        if _LIGHTNING_AVAILABLE:
            cb = ChaosMonitorCallback(log_every_n_steps=10)
            assert cb.log_every_n_steps == 10
        else:
            with pytest.raises(ImportError):
                ChaosMonitorCallback()


class TestHFTrainer:
    def test_trainer_class_requires_transformers(self):
        transformers = pytest.importorskip("transformers")
        trainer_cls = psilogic_trainer_class()
        assert issubclass(trainer_cls, transformers.Trainer)
