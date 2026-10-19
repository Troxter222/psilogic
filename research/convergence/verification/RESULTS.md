# Original investigation results

These results describe the investigation snapshot in `../source_snapshot.json`.
They precede publication edits to comments and documentation. Current editorial
checks are recorded in [`../../publication/README.md`](../../publication/README.md).

- **LaTeX:** successfully compiled with pdflatex/BibTeX; 16-page PDF. Final build
  has no overflowing boxes or unresolved references. Commands, logs and hashes
  are in `latex/build_results.json`.
- **PDF:** all pages visually inspected; final changed pages also viewed at full
  size. All 8,462 extracted word boxes lie within page bounds. See `pdf_review.json`.
- **Independent recurrence:** 34 cases, 47,380 parameter/state comparisons passed
  at atol=rtol=2e-5 on actual scalar CPU and forced CPU foreach. This is numerical
  evidence, not a general equivalence proof.
- **Proof algebra:** exact symbolic/rational identities, 300 momentum-tracking
  sequences and eight smooth nonconvex test trajectories passed. Finite tests
  do not replace the mathematical proofs.
- **Counterexamples:** C1-C7 reproduced; backend RMS precision difference confirmed.
  See the separate complete analytic proofs in `../counterexamples.md`.
- **Existing tests:** 72 passed; 16 skipped because CUDA/Triton execution was
  unavailable. Raw output is `existing_tests.log`.
- **Lean:** ten declarations compiled and kernel-checked, exit 0, with Lean 4.19.0
  and pinned Mathlib. All axiom reports use only standard foundations and no
  placeholder or added axioms. The full convergence theorem and stochastic
  reasoning are not formalized. Exact checked source hash and raw output are
  in `../lean/build_status.json` and `../lean/build.log`.
- **Source integrity at the time of the investigation:** all 14 inspected package files matched their initial SHA-256
  hashes; Git revision and non-research Git status match the initial snapshot.
  Existing user changes were preserved.

`summary.json` contains a machine-readable aggregate. Status is complete with
explicit mathematical limitations, not an unconditional convergence claim.
The main theorem concerns a selectable real-arithmetic Adam configuration with
full detector/moments but no decay or preprocessing, bounded observations, and
an explicit small constant step. Its practical step restriction is conservative.
