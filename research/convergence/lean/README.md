# Kernel-checked Lean 4 / Mathlib scalar core

**Status: 10 theorem statements compiled and kernel-checked successfully.**
The final proof-check process exited 0 with no warnings. Every `#print axioms`
result lists only Mathlib/Lean's standard foundations: `propext`,
`Classical.choice`, and `Quot.sound`. There is no `sorry`, `admit`, `unsafe`
declaration, custom axiom, or `sorryAx` dependency in this formalization.

Pinned versions: Lean 4.19.0 (runtime commit `6caaee842e94`), Mathlib v4.19.0
(commit `c44e0c8ee63ca166450922a373c7409c5d26b00b`). See `build.log` for the exact
command, raw output, runtime, dependency commit, and source SHA256;
`build_status.json` records the same result in machine-readable form. The first
unsuccessful file build is retained as `build_attempt1.log`; its real-valued
definition needed the standard `noncomputable` declaration. `setup_log.md`
documents installation and dependency-fetch obstacles.

| Checked statement | Exact scope |
|---|---|
| `trust_bounds` | Clamp input and cap nonnegative imply cap-dependent lower bound and upper bound 1. |
| `trust_floor` | Internal 0.99 clamp gives unconditional real-arithmetic floor 0.01. |
| `normalized_ema_bound` | Nonnegative prior average and norm, positive epsilon give ratio in [0,100]. |
| `ema_interval` | A convex EMA update preserves [0,100]. |
| `bias_epsilon_identity` | Exact scalar bias-correction identity with epsilon rescaled in the corrected denominator. |
| `trust_denominator_identity` | Scaling an update equals scaling its denominator inversely, on the stated nonzero domain. |
| `scalar_descent` | Algebraically absorbs the smoothness error under the explicit step restriction. The smoothness inequality is a premise. |
| `finite_descent_sum` | Telescopes a finite sequence of assumed descent inequalities. |
| `weighted_stationarity_certificate` | Uses an explicit final objective lower bound in that finite telescoping result. |
| `average_stationarity_bound` | Gives a 1/N average certificate for a positive constant descent coefficient and N>0, from explicit descent and lower-bound premises. |

These results **do not formalize the complete convergence theorem**. The
application must separately prove that the descent premise holds and that `G2`
is the squared objective-gradient norm. Differentiability/smoothness,
conditional expectations, stochastic adaptation, asymptotic limits, the
sigmoid/tanh composition, Python execution, CUDA, DDP, and floating-point
arithmetic are outside this formalization. The final average certificate is a
meaningful finite algebraic core, not a stochastic convergence proof.

To repeat the exact successful check while the temporary environment exists,
run from the repository root:

```sh
python research/convergence/lean/verify_lean.py \
  --mathlib /tmp/psilogic-mathlib \
  --lake /tmp/lean-4.19.0-linux/bin/lake
```

For a fresh environment with elan installed, the included pinned Lake project
provides the portable build path:

```sh
cd research/convergence/lean
lake update
lake exe cache get Mathlib/Data/Real/Basic.lean \
  Mathlib/Algebra/BigOperators/Group/Finset/Basic.lean \
  Mathlib/Tactic/Linarith.lean Mathlib/Tactic/FieldSimp.lean \
  Mathlib/Tactic/Ring.lean Mathlib/Tactic/NormNum.lean
lake build
```

The recorded session check used `lake env lean` in the pinned Mathlib checkout,
not a fresh-network execution of that portable setup sequence. Downloaded
third-party artifacts are deliberately outside the research source directory.
