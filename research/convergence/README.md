# PsiLogic convergence investigation

Start with **manuscript.pdf** (standalone research appendix) or **manuscript.tex**
(editable LaTeX). `theorem_status.md` and `assumptions.md` state exactly what the
results cover. The investigation used the implementation recorded in
`source_snapshot.json`; later publication edits change comments and docstrings.

The main result is a pathwise stationarity-energy bound for an actual selectable
Adam configuration retaining the full implemented detector and both moments,
with decay and preprocessing disabled. It yields deterministic convergence for
a sufficiently small constant step and conditional stochastic guarantees with
controlled error. It is not a theorem for the full default configuration,
arbitrary persistent noise, or floating-point execution. Counterexamples and
remaining limitations are documented alongside the proofs.

## Files

- `manuscript.tex`, `manuscript.pdf`, `references.bib`: standalone manuscript.
- `algorithm_specification.md`, `implementation_trace.md`, `backend_trace.md`:
  complete update equations, defaults, branches and exact source references.
- `source_snapshot.json`: revision, initial working-tree status and package hashes.
- `assumptions.md`, `theorem_status.md`: conditions and result-by-result scope.
- `proof_audit.md`, `independent_review.md`: gap register, repairs and second derivation.
- `counterexamples.md`, `counterexample_checks.py`, `counterexample_results.json`:
  analytic counterexamples and executable reproductions.
- `reference_verification.md`: checked primary bibliographic sources.
- `verification/`: recurrence and algebra checks, source integrity, test/build evidence.
- `lean/`: pinned formalization source and actual build status; only its listed
  scalar/finite-sum results may have kernel-check status, not the full theorem.

## Reproduce

From the repository root, using Python with the project's PyTorch dependencies
and NumPy/SymPy:

```sh
.venv/bin/python research/convergence/verification/check_recurrence.py
.venv/bin/python research/convergence/counterexample_checks.py
.venv/bin/python research/convergence/verification/check_proof_algebra.py
python3 research/convergence/build_manuscript.py
```

The build needs `pdflatex` and `bibtex` plus standard AMS, geometry, booktabs,
microtype, hyperref, natbib and enumitem packages. It does not require the
optional Latin Modern package. Build intermediates and transcripts are stored
under `verification/latex/` (ignored except for the build summary), and the
final PDF is copied to this directory. Compiler logs are generated locally.
The Lean build instructions and version pins are in `lean/README.md`.

For the existing selected regression suite:

```sh
.venv/bin/python -m pytest tests/test_psilogic.py tests/test_soft_chaos_trust.py \
  tests/test_chaos_warmup.py tests/test_numerical_parity.py \
  tests/test_adamw_equivalence.py tests/test_state_dict_migration.py -ra
```

Numerical checks use tolerances and cannot prove exact program equivalence.
CUDA-dependent tests are skipped when hardware is unavailable. The real-arithmetic
theorems do not claim exact convergence of rounded parameter updates.

## Original source snapshot

`source_snapshot.json` and the original verification logs describe the source
and working tree inspected during the investigation. Source-line references
in the implementation and backend traces refer to that snapshot. Publication
edits can move those lines without changing the update equations.

`verification/check_source_integrity.py` checks byte-for-byte identity and Git
status against the original snapshot. It is an archival check and is expected
to fail after editorial changes or a new commit. Do not replace the original
hashes to make it pass. The publication checks are recorded separately in
[`../publication/README.md`](../publication/README.md).
