# Contributing to PsiLogic

Thank you for helping improve PsiLogic. This project targets reproducible optimizer research with production-quality engineering.

## Development Setup

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic
pip install -e ".[all]"
pre-commit install   # optional but recommended
```

## Workflow

1. **Branch** from `main` with a descriptive name (`fix/vit-preset`, `feat/hf-integration`).
2. **Change** the smallest surface that solves the problem — match existing style and module boundaries.
3. **Test** locally:
   ```bash
   ruff check psilogic tests benchmark examples
   ruff format psilogic tests benchmark examples
   pytest tests/ -v
   mypy psilogic
   ```
4. **Document** user-facing API changes in `CHANGELOG.md` under `[Unreleased]`.
5. **Open a PR** against `main` with a clear description and test plan.

## Project Layout

| Path | Purpose |
|------|---------|
| `psilogic/` | Public API — keep imports stable within a major version |
| `psilogic/_chaos.py`, `psilogic/_version.py` | Private internals (leading `_`) |
| `tests/` | pytest suite; mark GPU tests with `@pytest.mark.gpu` |
| `benchmark/` | Reproducibility harness (not shipped on PyPI) |
| `examples/` | Integration recipes |

## Code Guidelines

- Use `from __future__ import annotations` in new modules.
- Validate hyperparameters with `ValueError`, not `assert`.
- Optional dependencies (HuggingFace, Lightning) must degrade gracefully when not installed.
- New public symbols go in the relevant module's `__all__` and `psilogic/__init__.py` if top-level.
- Bump `psilogic/_version.py` and `CHANGELOG.md` for releases; CI publishes on `v*.*.*` tags.

## Running Benchmarks

See [`benchmark/README.md`](benchmark/README.md). Benchmark changes should not break `python benchmark/run_all.py --suite quick --steps 10` on CPU.

## Questions

Open a [GitHub issue](https://github.com/Troxter222/psilogic/issues) for bugs, feature requests, or benchmark reproduction questions.

Security vulnerabilities should be reported privately — see [`SECURITY.md`](SECURITY.md).
