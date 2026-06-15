# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-06

### Added

- **Modular package layout** — the `psilogic.py` monolith is split into
  `optimizer.py`, `_chaos.py`, `presets.py`, `param_groups.py`,
  `convenience.py` with no public API breaks.
- **`psilogic.debug`** diagnostics module: `chaos_stats(optimizer)`,
  `norm_history(optimizer, model)`, `layer_norms(model)` and the per-parameter
  `get_chaos_metrics(state)` export.
- **`vit_param_groups`** — ViT split with per-group gamma (patch embed 0.005,
  attention 0.02, MLP 0.03), norm/bias without weight decay, and an optional
  `lion_blocks=True` mode that runs Lion sign-momentum on transformer blocks
  while patch embeddings stay on Adam.
- **`gpt_param_groups`** — from-scratch LM split: embeddings γ=0.005 with
  quantum decay disabled, transformer blocks γ=0.02, LM head γ=0.01.
  Weight-tied heads are handled without duplication.
- **Chaos warm-in** — `chaos_warmup=-1` now auto-scales to
  `max(500, total_steps // 20)` and the chaos gain ramps in linearly over a
  quarter of the warmup window instead of switching on abruptly
  (`_chaos.effective_warmup`).
- **`PsiLogic.auto(model)`** — zero-config constructor that infers the
  architecture (ViT / GPT / NLP encoder / CNN / generic) and applies the
  matching preset and parameter groups.
- **Auto-γ** (`gamma_auto=True`) — reduces gamma when the slow EMA signals
  convergence (`_chaos.auto_gamma`).
- **DDP chaos sync** (`sync_chaos_ddp=True`) — all-reduces fast/slow chaos
  signals across ranks so every rank damps identically.
- **Step-time profiling** (`profile_step_time=True`) — records
  `last_step_time_ms` and `step_time_ms_ema` on the optimizer.
- **`state_dict` schema v2** with transparent migration of v0.3-monolith
  checkpoints (missing group keys filled from defaults, missing chaos-state
  tensors re-initialized to neutral values); future schemas are rejected with
  a clear error.
- **Presets**: `whisper_defaults()` (speech fine-tuning) and
  `glue_defaults()` (encoder fine-tuning at GLUE scale).
- **Convenience classes** now accept a bare `nn.Module` and build their param
  groups automatically (`PsiLogicNLP`, `PsiLogicGPT`, `PsiLogicViT`); new
  `PsiLogicWhisper` preset class.
- **Integrations** — `psilogic.integrations.hf.create_psilogic_optimizer` /
  `psilogic_trainer_class()` for HuggingFace Transformers and
  `psilogic.integrations.lightning.configure_psilogic` /
  `ChaosMonitorCallback` for PyTorch Lightning. Both are import-safe without
  the corresponding framework installed.
- **Benchmark suite** — `benchmark/run_benchmark.py` gains `cifar10`
  (ResNet-18) and `nanogpt` (char-GPT / Tiny Shakespeare) arenas plus a
  `--preset vit` shortcut; new `benchmark/imagenet/train_imagenet.py`
  (DDP-ready, bf16 AMP, cosine LR) and `benchmark/run_all.py`
  (`--suite v1` one-command reproduction).
- **Tests** — new coverage for the ViT/GPT param-group splits, chaos warmup
  auto-scaling, gradient accumulation correctness, step-time overhead,
  state-dict migration, zero-config auto mode, debug utilities and the
  framework integrations.

### Changed

- Hyperparameter validation raises `ValueError` instead of `AssertionError`
  (library best practice; asserts are stripped under `python -O`).
- `benchmark/run_benchmark.py` imports `PsiLogic` from the package instead of
  carrying an inline copy, and reads Telegram credentials from the
  `PSILOGIC_TG_TOKEN` / `PSILOGIC_TG_CHAT` environment variables instead of
  hardcoded values.
- The benchmark LR scheduler is a local cosine-with-warmup implementation
  (drops the `transformers` dependency for non-HF arenas).
- `use_foreach=True` now degrades gracefully when `torch._foreach_*` ops are
  unavailable in the running PyTorch build.

### Compatibility

- No breaking public API changes versus 0.3.2. Checkpoints saved by 0.3.x
  load through the schema-v1 migration path.

## [0.3.2]

- Initial public PyPI release: monolithic `psilogic.py`, CIFAR-10 / SST-2 /
  WikiText-2 benchmark results, Zenodo DOI.
