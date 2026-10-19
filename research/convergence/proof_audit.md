# Proof audit and gap register

## Evidence hierarchy

Recorded run counts and source hashes refer to the original investigation.
Later editorial checks are recorded in `../publication/README.md`.

1. Source inspection and recorded hashes establish which implementation was read.
2. Independent recurrence tests support numerical correspondence on tested CPU
   inputs; they do not prove program equivalence.
3. The manuscript contains the mathematical proofs under explicit hypotheses.
4. `independent_review.md` records a second derivation of the main inequalities.
5. Symbolic/rational and numerical stress checks support algebra and expose
   counterexamples; they do not formally verify the convergence theorem.
6. Only successfully built Lean declarations have kernel-check status. See
   `lean/README.md`, `lean/build.log`, and `theorem_status.md` for actual coverage.

## Implementation checks and second derivation

AI agents reconstructed the scalar recurrence in two separate passes and inspected
the Triton, fused and foreach paths, including stale generated build copies.
Before proof construction, the reference recurrence passed 34 cases and 47,380
comparisons; the scripts and recorded results are in `verification/`.

After the pathwise full-momentum bound was derived, a second derivation checked
the preconditioner spectrum, EMA weights, drift convolution, error absorption and
constants. It is recorded in `independent_review.md`. These checks were performed
by AI agents and do not constitute external peer review. The absorption
coefficient is at least 5a/16, which permits the weaker bound a/4 and the resulting
constants 9 and 13.

The second derivation used the looser valid bound
`a_old = c sqrt(1-beta2)/(G+eps)`. The final manuscript uses
`a = c/(G+eps/sqrt(1-beta2))`. The stronger denominator bound follows from
`v_t/(1-beta2^t) <= G²`. Every later proof
only needs `a I <= H_t <= M I`, so the same proof and constants apply. The
original derivation is retained to preserve the audit trail, not presented as
an exact transcription of the final manuscript.

## Repairs made after adversarial manuscript review

- **Convex lower-bound bug:** the initial convex corollary used a generic lower
  bound f_* as though it were an attained minimum. For f=x²/2, x_t=x_*=0,
  f_*=-1 and R=0 its original statement was false. The final statement explicitly
  requires f_*=f(x_*). The strongly convex discussion has the same qualification.
- **Integrability:** the simplified predictable-scaling proof now derives finite
  second moments inductively and objective integrability before taking conditional
  expectations.
- **Explicit domains:** trust bounds reference the analytic assumptions; the spike
  proposition states beta/epsilon domains, zero initial moments and positive step.
- **Clock wording:** local counters advance; only configured warmup and horizon
  are fixed integers.
- **Decay scope:** large finite decay products are legal inputs outside the main
  zero-decay theorem; they are not mislabeled as outside the analytic domain.
- **Counterexample scope:** one-step expected ascent is never labeled asymptotic
  divergence. The exact fixed-step cycle and GC invariant establish genuine
  nonstationarity for their specified configurations.

## Claim-by-claim register

| ID | Potential gap or rejected claim | Resolution/status |
|---|---|---|
| P01 | E[trust*g] = E[trust] E[g] | Rejected; trust is current-sample dependent. Main proof is pathwise. |
| P02 | Bounded positive damping implies convergence | Disproven as a standalone argument; exact detector can reverse conditional mean. |
| P03 | Adam convergence transfers automatically | Rejected; no external Adam theorem imported. |
| P04 | Bias-correction epsilon treated as constant | Repaired in specification: effective corrected epsilon is eps/sqrt(1-beta2^t). |
| P05 | First observation or initial drift dropped | d_1=0, n_1=xi_1 retained; theorem requires fresh moments. |
| P06 | Momentum drift uses wrong time index | Uses update t-1 and sums through T-1 before enlargement to T. |
| P07 | Noise/history independence assumed | Not needed for pathwise Jensen/column-sum energy bound. |
| P08 | Spectrum inferred only from trust | Bounded observations and eps>0 additionally establish a and M. |
| P09 | Error absorption circular | Explicit step condition bounds Q first; positive coefficient then permits division. |
| P10 | Cesaro convergence used as pointwise convergence | Pointwise claims require summable error or relative-error absorption. |
| P11 | Interchange of expectation and infinite sum hidden | Monotone convergence of nonnegative terms stated explicitly. |
| P12 | Clipping/centralization preserves oracle unbiasedness | Rejected; AGC mean reversal and GC invariant provided. |
| P13 | Default weight decay has same stationary objective | Rejected; it is not generally gradient descent on f+lambda‖x‖²/2 in adaptive metric. |
| P14 | Cancellation can be exactly zero | On intended domain trust>=.01; negative p_ext can instead amplify. |
| P15 | Constant step always converges | Disproven by exact two-cycle for beta=(0,0). Small-step theorem remains valid. |
| P16 | Full fixed-variance stochastic convergence | OPEN under weaker useful conditions; present bound has a residual term. |
| P17 | All optional modes covered | NO theorem for default decay, quantum decay, general Lion, asynchronous groups or arbitrary resumes. |
| P18 | Real formula verifies floating-point implementation | Rejected; rounding/stalling and layout restrictions disclosed. |
| P19 | Numerical or language-model audit equals formal verification | Rejected; Lean status is declaration-specific. |
| P20 | Matching iteration rate proves advantage over AdamW | Rejected; no superiority claim; growing-batch sample count disclosed. |
| P21 | Practical hyperparameters satisfy theorem | Not established; epsilon-dependent step restriction is extremely conservative. |
| P22 | Convex f_* is automatically attained minimum | Repaired explicitly; counterexample to initial statement recorded above. |

## Remaining mathematical obligations

There is no hidden unproved step in a claimed broader default-mode theorem:
no such theorem is asserted. The open research tasks are weaker realistic
conditions for persistent-noise Adam with this detector, a suitable target for
nonzero adaptive/decoupled decay, analysis of Lion and projected/clipped oracles,
nonuniform clocks and loaded states, and quantified arithmetic errors.
Formalizing normed-space smoothness, momentum tracking, and stochastic corollaries
in Lean remains separate from checking the scalar algebraic core.
