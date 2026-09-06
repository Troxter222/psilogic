# Contributing to PsiLogic

Thank you for helping improve PsiLogic. This project targets reproducible optimizer
research with production-quality engineering. Be excellent to each other.

## Contents

- [Development setup](#development-setup)
- [Workflow](#workflow)
- [PR checklist](#pr-checklist)
- [Project layout](#project-layout)
- [What goes where](#what-goes-where)
- [Code guidelines](#code-guidelines)
- [Tests](#tests)
- [Pre-commit hooks](#pre-commit-hooks)
- [Benchmarks](#benchmarks)
- [Releases](#releases)
- [Deprecation policy](#deprecation-policy)
- [Questions & security](#questions--security)

---

## Development setup

Minimal path (lint + tests + type-check):

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic
pip install -e ".[dev]"
pip install pre-commit && pre-commit install   # optional but recommended
```

Full local stack (integrations, CUDA extras, benchmark deps):

```bash
pip install -e ".[all]"
```

---

## Workflow

1. **Branch** from `main` with a descriptive name:
   - `fix/…` — bug fixes
   - `feat/…` — features
   - `docs/…` — documentation only
   - `bench/…` — FairBench / profiling
2. **Change** the smallest surface that solves the problem — match existing style and module boundaries.
3. **Test** locally (see [Tests](#tests)).
4. **Document** user-facing API or behavior changes in `CHANGELOG.md` under `[Unreleased]`.
5. **Open a PR** against `main` with a clear description and test plan.

Commit messages: short imperative summary (Conventional Commits style is welcome:
`feat:`, `fix:`, `docs:`, `test:`, `perf:`, `chore:`).

---

## PR checklist

Before requesting review:

- [ ] `ruff check` + `ruff format` clean on touched paths
- [ ] `pytest tests/ -v` passes locally (CPU is enough for CI-equivalent)
- [ ] `mypy psilogic` clean if you touched typed public API
- [ ] User-facing changes noted in `CHANGELOG.md` → `[Unreleased]`
- [ ] README / ROADMAP / PAPER updated if headlines, defaults, or gates changed
- [ ] No secrets (`.env`, tokens, credentials) and no huge binary dumps unless intentional FairBench artifacts
- [ ] Do **not** rewrite `benchmark/results/full/` without a full FairBench re-run + note in the PR

CI (`.github/workflows/ci.yml`) runs ruff + pytest on a Python/torch matrix and must stay green.

---

## Project layout

| Path | Purpose |
|:-----|:--------|
| `psilogic/` | Public installable package — keep imports stable within a major version |
| `psilogic/optimizer.py` | Core `PsiLogic` optimizer |
| `psilogic/_chaos.py`, `psilogic/_version.py` | Private internals (leading `_`) |
| `psilogic/_cuda/` | Optional Triton fused step |
| `psilogic/integrations/` | HuggingFace / Lightning (optional deps) |
| `psilogic/presets.py`, `param_groups.py`, `convenience.py`, `debug.py` | Presets, helpers, diagnostics |
| `tests/` | pytest suite; GPU tests marked `@pytest.mark.gpu` |
| `benchmark/` | FairBench harness (not shipped on PyPI) |
| `examples/` | Integration recipes |
| `scripts/` | Profilers, arXiv PDF build, research helpers |
| `arxiv/` | LaTeX source of truth for the paper (`paper.tex`) |
| `run_fairbench.sh` | Detached tmux launcher for long FairBench runs |

---

## What goes where

| Change type | Primary files |
|:------------|:--------------|
| Optimizer / API behavior | `psilogic/` + tests + `CHANGELOG.md` |
| Headline numbers / FairBench CSVs | `benchmark/results/` + sync `README.md` / `PAPER.md` / `ROADMAP.md` scorecard |
| Plans, gates, version map | `ROADMAP.md` |
| Paper narrative / tables | Prefer `arxiv/paper.tex` (SoT); keep `PAPER.md` notes in sync |
| Security policy | `SECURITY.md` (private reports only) |

---

## Code guidelines

- Use `from __future__ import annotations` in new modules.
- Validate hyperparameters with `ValueError`, not `assert`.
- Optional dependencies (HuggingFace, Lightning, Triton, DeepSpeed) must degrade gracefully when missing.
- New public symbols go in the module `__all__` and in `psilogic/__init__.py` if top-level.
- Prefer small, reviewable PRs over drive-by refactors.

---

## Tests

```bash
# Default CI-equivalent
ruff check psilogic tests benchmark examples
ruff format --check psilogic tests benchmark examples
pytest tests/ -v
mypy psilogic
```

| Situation | Run |
|:----------|:----|
| Touched `optimizer.py` / `_chaos.py` | Full `pytest tests/` |
| Touched fused CUDA path | `pytest tests/test_numerical_parity.py tests/test_step_overhead.py -v` (GPU if available) |
| Touched presets / convenience | `pytest tests/test_*preset*.py tests/test_auto_config.py -v` |
| Touched integrations | `pip install -e ".[integrations]"` then `pytest tests/test_integrations.py -v` |
| GPU-only tests | `pytest tests/ -m gpu -v` (auto-skip on CPU CI) |
| Multi-GPU | `pytest tests/ -m multi_gpu -v` |

Fusion debug tip: `PsiLogic(..., use_fused_cuda=False)` forces foreach/scalar fallback.

---

## Pre-commit hooks

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

| Hook | What it does |
|:-----|:-------------|
| `ruff` | Lint + auto-fix (`--fix`) |
| `ruff-format` | Code formatting |
| `mypy` | Type-check `psilogic/` |

---

## Benchmarks

Canonical harness is FairBench (`python -m fairbench` from `benchmark/`), not legacy
`benchmark/run_all.py`.

```bash
# CPU-friendly smoke (no downloads)
cd benchmark
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0

# Full protocol / flags
# see benchmark/README.md

# Long detachable run from repo root
./run_fairbench.sh
```

**When a full re-run is required:** changes that affect optimizer math, default
hyperparameters used by FairBench arenas, or claims about wall time / quality.
Unit tests alone are enough for refactors that preserve numerical parity
(`tests/test_numerical_parity.py`, `tests/test_adamw_equivalence.py`).

Do not commit new `benchmark/results/full/` aggregates without documenting the
machine, commit SHA, and whether `psilogic[cuda]` fusion was enabled.

---

## Releases

1. Bump `psilogic/_version.py`.
2. Move `[Unreleased]` notes in `CHANGELOG.md` into a dated section for that version.
3. Sync README / ROADMAP scorecard if headline numbers or defaults changed.
4. Tag `vMAJOR.MINOR.PATCH` on `main` — CI publish workflow builds and uploads to PyPI.
5. Update Zenodo / DOI metadata on major scientific releases when applicable.

---

## Deprecation policy

- Within a **minor** release (0.x → 0.y): prefer warnings + CHANGELOG notes; avoid silent
  behavior changes to public constructors when possible. Document intentional default
  changes (e.g. v0.6 safer AGC/GC defaults) clearly in README + CHANGELOG.
- **Major** (1.0+): breaking API changes require a deprecation window of at least one
  minor release with `DeprecationWarning` where practical, then removal.
- Private modules (`psilogic/_*.py`) may change without notice.

---

## Questions & security

- Bugs, features, benchmark reproduction: open a [GitHub issue](https://github.com/Troxter222/psilogic/issues).
- Security vulnerabilities: report **privately** — see [`SECURITY.md`](SECURITY.md)
  (latest PyPI release and `main` are supported; do not file public issues for vulns).
