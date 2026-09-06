"""Convenience classes and FairBench kwargs must track presets.py literals."""

from __future__ import annotations

from psilogic import PsiLogicGPT, PsiLogicNLP, PsiLogicViT, PsiLogicWhisper
from psilogic.presets import (
    as_fairbench_kwargs,
    glue_defaults,
    gpt_scratch_defaults,
    nlp_defaults,
    vision_defaults,
    whisper_defaults,
)
from tests.toy_models import ToyGPT, ToyViT


def _assert_group_matches_preset(opt, preset: dict, keys: tuple[str, ...]) -> None:
    group = opt.param_groups[0]
    for key in keys:
        assert group[key] == preset[key], f"{key}: {group[key]!r} != {preset[key]!r}"


class TestPresetSourceOfTruth:
    _KEYS = (
        "gamma",
        "chaos_tau",
        "adaptive_tau",
        "tau_scale",
        "max_cancel",
        "agc_clip",
        "grad_centralize",
        "quantum_decay",
    )

    def test_gpt_matches_gpt_scratch_defaults(self):
        preset = gpt_scratch_defaults(0)
        opt = PsiLogicGPT(ToyGPT().parameters(), lr=3e-4)
        _assert_group_matches_preset(opt, preset, self._KEYS)
        assert opt.param_groups[0]["weight_decay"] == preset["weight_decay"]

    def test_nlp_matches_nlp_defaults(self):
        preset = nlp_defaults(0)
        opt = PsiLogicNLP(ToyGPT().parameters(), lr=1e-3)
        _assert_group_matches_preset(opt, preset, self._KEYS)

    def test_vit_matches_vision_defaults(self):
        preset = vision_defaults(0)
        opt = PsiLogicViT(ToyViT().parameters(), lr=1e-3)
        _assert_group_matches_preset(opt, preset, self._KEYS)

    def test_whisper_matches_whisper_defaults(self):
        preset = whisper_defaults(0)
        opt = PsiLogicWhisper(ToyGPT().parameters(), lr=1e-5)
        _assert_group_matches_preset(opt, preset, self._KEYS)

    def test_fairbench_kwargs_subset(self):
        fb = as_fairbench_kwargs(vision_defaults())
        assert fb["grad_centralize"] is True
        assert "use_foreach" not in fb
        assert "gamma_T_max" not in fb
        assert glue_defaults(1000)["gamma_T_max"] == 1000
