# Publication preparation

Editorial revision of 19 October 2026. The changes are intended for one Git
commit; they do not introduce a new optimizer version or a PyPI release.

## Scope

Comments and docstrings now describe the operations directly. The review
removed repeated explanations, decorative section borders and claims that a
detector statistic establishes convergence. Function names, formulas, defaults,
error messages and executable statements were retained. Research scripts were
formatted consistently with the rest of the Python sources. Clear existing text
was left as written. Historical experiment logs and proof-check evidence were
not rewritten to match the new prose.

Research Markdown uses GitHub math delimiters. Norm delimiters use `\Vert{}`
so they do not split Markdown table cells. The theorem-status table separates
the manuscript proofs, supplementary checks and Lean coverage. The audit notes
identify AI-assisted work and distinguish it from external peer review.

The roadmap now includes the saved September H100 results. Training-run wall
time and optimizer-only step time remain separate measurements. The original
benchmark CSVs, configurations, numerical outcomes and paper tables are retained.
No training was run during this revision.

## Verification records

- `code_equivalence.json`: compares the executable AST of all 80 existing Python
  files before and after this preparation, removing only docstrings and source
  positions. All 14 package files are also compared with Git revision
  `b4917163d63d8a5cb2960e29af4b1bccdbf5798d`, covering the earlier optimizer
  comment edits. Other string literals, expressions and identifiers are retained.
- `math_preservation.json`: checks that all theorem, lemma, proposition,
  corollary, assumption, definition and proof environments in the theory source
  are byte-identical. The empirical Markdown source is unchanged.
- `markdown_math.json`: compares all 370 converted mathematical fragments after
  normalizing delimiter syntax and whitespace.
- `recurrence_results.json`, `proof_algebra_results.json` and
  `counterexample_results.json`: fresh numerical checks using a temporary copy
  of the edited package. The original research result files are retained.
- `lean_build_status.json` and `lean_build.log`: a fresh check of the unchanged
  Lean source with the pinned toolchain and Mathlib checkout.
- `checks.json`: final lint, formatting, type, test, link and PDF results,
  including environment limitations.
- `tests.log` records the original suite run, including the socket-permission
  failure; `ddp_retry.log` records its successful retry with local sockets
  permitted. `mypy.log` records the type check.
- `pdf_review.json`: page counts, output hashes, rendered-page inspection and
  checks that extracted word boxes remain within page bounds.

The original `../convergence/source_snapshot.json`, verification summaries and
Lean logs describe the earlier investigation. Their package hashes and source
line numbers precede this editorial revision. The archival
`check_source_integrity.py` deliberately requires the original bytes and Git
status; it is not a check of editorial equivalence.
The source hashes in `../../arxiv/generated/provenance.json` likewise identify
the original empirical audit inputs, before the comment edits.

## Dependency resolution

The existing local `uv.lock` omits the `deepspeed` extra declared by the current
`pyproject.toml`. It is excluded from Git rather than published as a validated
lockfile. Offline resolution could not complete because required index metadata
was missing from the cache. Package requirements were not changed.

The optional uv workflow in [CONTRIBUTING.md](../../CONTRIBUTING.md) resolves
from the current project metadata. Neither that resolution nor the local
development environment reconstructs the archived H100 training stack.

## Files to publish

Keep package and test sources, documentation, research sources, final PDFs,
saved benchmark snapshots, generated paper tables and figures, and the scripts
needed to rebuild them. `.gitignore` excludes local dependency resolution,
dataset caches, temporary review images, duplicate compiler outputs and LaTeX
intermediates. The final Lean logs are retained as proof-check evidence.

The three final articles are:

- [Empirical article](../../output/pdf/paper.pdf).
- [Convergence appendix](../convergence/manuscript.pdf).
- [Combined article](../../output/pdf/paper_unified.pdf).

Build from the repository root:

```sh
python3 scripts/build_arxiv_pdf.py --skip-results
python3 research/convergence/build_manuscript.py
python3 research/unified/build.py
```

`--skip-results` retains the saved tables and figures. Regenerating those assets
separately requires the analysis dependencies listed in `PAPER.md`.

## One commit

No staging, commit, push or release is performed by these preparation steps.
To include the prepared changes in one commit, review the working tree and run
these commands yourself from the repository root:

```sh
git status --short
git add -- .gitignore CHANGELOG.md CONTRIBUTING.md README.md ROADMAP.md PAPER.md \
  arxiv benchmark psilogic scripts tests research \
  output/pdf/paper.pdf output/pdf/paper_unified.pdf
git diff --cached --stat
git diff --cached --check
git commit -m "chore: prepare research and documentation for publication"
git push
```

Review the staged changes before committing, including the pre-existing paper
and benchmark additions. The lockfile and build intermediates should not appear
in that list. The push uses the current branch's configured upstream.
