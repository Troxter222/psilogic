# Contributing

Small, focused changes are easier to review. Match the code around what you touch. Don't reformat files you didn't need to edit.

## Setup

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic
pip install -e ".[dev]"
```

Optional, same checks as CI:

```bash
pip install pre-commit && pre-commit install
```

Integrations, CUDA extras, and benchmark deps:

```bash
pip install -e ".[all]"
```

## Sending a change

Branch from `main`. I use `fix/…`, `feat/…`, `docs/…`, `bench/…`.

Commit subject: short imperative line. Prefixes `feat:`, `fix:`, `docs:`, `test:`, `perf:`, `chore:` are fine.

Before review:

- `ruff check` and `ruff format` clean on the paths you changed
- `pytest tests/ -v` passes (CPU is enough to match CI)
- `mypy psilogic` if you changed typed public API
- user-facing behavior noted under `[Unreleased]` in `CHANGELOG.md`
- README / ROADMAP / PAPER updated if headlines, defaults, or gates changed
- no `.env`, tokens, or credentials, and no large binaries unless they are FairBench artifacts you meant to add
- don't replace `benchmark/results/full/` unless you actually re-ran FairBench and say so in the PR

CI is `.github/workflows/ci.yml` (ruff + pytest on a Python/torch matrix). Keep it green.

## Layout

| Path | What it is |
|:-----|:-----------|
| `psilogic/` | installable package. Public imports stay stable inside a major version |
| `psilogic/optimizer.py` | `PsiLogic` |
| `psilogic/_chaos.py`, `psilogic/_version.py` | private (leading `_`) |
| `psilogic/_cuda/` | optional Triton fused step |
| `psilogic/integrations/` | HuggingFace / Lightning, optional deps |
| `psilogic/presets.py`, `param_groups.py`, `convenience.py`, `debug.py` | presets, helpers, diagnostics |
| `tests/` | pytest. GPU tests are `@pytest.mark.gpu` |
| `benchmark/` | FairBench. Not shipped on PyPI |
| `examples/` | recipes |
| `scripts/` | profilers, arXiv PDF build, research helpers |
| `arxiv/` | paper source (`paper.tex`) |
| `run_fairbench.sh` | tmux launcher for a long FairBench run |

| You changed | Also update |
|:------------|:------------|
| optimizer or public API | `psilogic/`, tests, `CHANGELOG.md` |
| headline numbers or FairBench CSVs | `benchmark/results/` and the scorecard in README / PAPER.md / ROADMAP |
| plans, gates, version map | `ROADMAP.md` |
| paper text or tables | `arxiv/paper.tex` first, then the notes in `PAPER.md` |
| how to report a vulnerability | `SECURITY.md`. Reports stay private |

## Code

New modules start with `from __future__ import annotations`. Bad hyperparameters raise `ValueError`, not `assert`. HuggingFace, Lightning, Triton, and DeepSpeed must be skippable if they are not installed. A new public name goes in that module's `__all__`, and in `psilogic/__init__.py` if it is top-level.

## Tests

```bash
ruff check psilogic tests benchmark examples
ruff format --check psilogic tests benchmark examples
pytest tests/ -v
mypy psilogic
```

| You touched | Run |
|:------------|:----|
| `optimizer.py` or `_chaos.py` | `pytest tests/` |
| fused CUDA | `pytest tests/test_numerical_parity.py tests/test_step_overhead.py -v` (GPU if you have one) |
| presets or convenience helpers | `pytest tests/test_*preset*.py tests/test_auto_config.py -v` |
| integrations | `pip install -e ".[integrations]"` then `pytest tests/test_integrations.py -v` |
| GPU-only tests | `pytest tests/ -m gpu -v` (skipped on CPU CI) |
| multi-GPU | `pytest tests/ -m multi_gpu -v` |

If fusion is the thing you are debugging, `PsiLogic(..., use_fused_cuda=False)` forces the foreach/scalar path.

## Pre-commit

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

Hooks: ruff with `--fix`, ruff-format, mypy on `psilogic/`.

## Benchmarks

The harness is FairBench (`python -m fairbench` from `benchmark/`). `benchmark/run_all.py` is the old runner.

```bash
cd benchmark
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0
```

Flags and the full protocol: [benchmark/README.md](benchmark/README.md). From the repo root, `./run_fairbench.sh` starts a long run in tmux.

Re-run FairBench if you changed optimizer math, a default the arenas actually use, or a number you would cite as wall time or quality. A refactor that keeps the same numbers does not need a new run. Check `tests/test_numerical_parity.py` and `tests/test_adamw_equivalence.py`.

If you commit a new `benchmark/results/full/`, say which machine, which commit, and whether `psilogic[cuda]` fusion was on.

## Releases

1. Bump `psilogic/_version.py`.
2. Move `[Unreleased]` in `CHANGELOG.md` into a dated section for that version.
3. If headline numbers or defaults changed, update the README and ROADMAP scorecard.
4. Tag `vMAJOR.MINOR.PATCH` on `main`. The publish workflow builds the wheel and uploads to PyPI.
5. On a release you want cited, update the Zenodo record.

## Breaking changes

On 0.x, prefer a warning and a CHANGELOG note over a silent change to a public constructor. If a default changes on purpose (v0.6 turned AGC and gradient centralization off on the bare constructor), say so in README and CHANGELOG.

From 1.0, a breaking public API needs at least one minor release with `DeprecationWarning` where that is practical, then removal.

`psilogic/_*.py` can change without a deprecation cycle.

## Questions

Bugs, features, and reproduction: [GitHub issues](https://github.com/Troxter222/psilogic/issues).

Vulnerabilities: [SECURITY.md](SECURITY.md). Do not file those in public. Supported versions are the latest PyPI release and `main`.
