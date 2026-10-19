# Executable correspondence checks

Run from the repository root:

```sh
.venv/bin/python research/convergence/verification/check_recurrence.py
```

The script imports the actual `PsiLogic` class but **does not call any optimizer
helper inside its reference recurrence**. It evaluates the reconstructed
mathematical recurrence in float64, then compares every parameter, momentum,
second moment, gradient norm average, fast/slow EMA, soft chaos value, and local
step counter with the actual CPU implementation after every step. The reference
uses out-of-place operations and explicit formulas. Absolute/relative tolerances
are both 2e-5 because production states are float32. Successful execution writes
`recurrence_results.json`, including hashes of the checked source files.

The 17 scenarios run through both CPU scalar and the explicitly invoked CPU
foreach path (34 cases total): Adam, Lion, AGC before centralization and quantum
decay, zero first gradient, explicit warmup/ramp/cosine endpoint, automatic
warmup with and without a known horizon, automatic gamma reduction and floor,
99% cancellation cap, absolute/adaptive gate stabilizers, disabled gamma with
nonzero quantum decay, zero learning rate, negative `p_ext`, missing gradients,
unequal local counters, and zero momentum with non-negligible epsilon. Branch
coverage assertions ensure the important boundary cases were actually reached.

Scope: numerical correspondence evidence for the enumerated CPU cases. Forced
CPU foreach exercises its mathematical implementation without claiming CUDA,
Triton, DDP, mixed precision, compiler transformation, or every input is covered.
It is not a machine-checked convergence proof or bitwise equivalence proof.

Initial run: **34 cases, 47,380 comparisons, all passed**, PyTorch 2.13.0+cu130,
CUDA unavailable. See JSON for casewise errors and source hashes.

## Proof-supporting algebra and inequalities

`check_proof_algebra.py` checks exact SymPy/Fraction identities for epsilon,
Young's inequality, and absorption constants; it tests 300 finite arbitrary
momentum-tracking sequences and eight smooth nonconvex synthetic trajectories
with deliberately current-gradient-dependent positive diagonal scaling.
`proof_algebra_results.json` records all results and the deterministic seed.
These are finite stress tests, not substitutes for the manuscript proofs.

`check_source_integrity.py` compares all 14 recorded package source SHA-256
hashes, revision and non-research Git status with the original snapshot.
It is an archival check: editorial changes and subsequent commits make it fail.
The original hashes and result logs are retained rather than updated to match
later edits. It does not hash unrelated user files or read their contents.

The selected pre-existing regression suite completed with **72 passed, 16
skipped** (CUDA unavailable); raw output is `existing_tests.log`. No GPU or
Triton execution guarantee is inferred from CPU checks.

`../build_manuscript.py` writes reproducible compiler commands, logs and source/
PDF hashes to `latex/`. Final PDF render inspection is recorded separately.
