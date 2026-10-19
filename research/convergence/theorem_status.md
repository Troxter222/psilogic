# Theorem status

Labels refer to [the manuscript](manuscript.tex), which contains the proofs
under the stated assumptions. The proof column lists additional checks or
qualifications. Only the results identified in the Lean column are
kernel-checked; the [second derivation](independent_review.md) is AI-assisted.

| Label | Mathematical result | Scope | Proof/review | Lean coverage |
|---|---|---|---|---|
| lem:ema | Nonnegative norm EMA, ratio<=100, fast/slow<=100; zero-input asymptotics | Exact detector, positive epsilon | Recurrence checked | Normalization and interval lemmas kernel-checked; asymptotics not formalized |
| lem:trust | Trust in [c,1], c>=.01; cumulative multiplier bounds | Finite nonnegative detector domain | Exact cap checked | Trust interval and floor kernel-checked; cumulative bound informal |
| lem:scaling | Spectrum aI<=H_t<=MI, bounded corrected moment | Bounded observations, zero moments | Stronger a checked in second derivation | Scalar epsilon identity kernel-checked; vector spectrum not formalized |
| thm:predictable | O(T^-1/2) expected average stationarity | Modified predictable-scaling method | Integrability argument added after audit | No stochastic theorem formalized |
| lem:tracking | Bias-corrected momentum tracking energy | Arbitrary vector errors, no independence | Second derivation and symbolic/numerical checks | Not formalized |
| thm:main | Pathwise full-momentum energy bound, constants 9 and 13 | Selectable Adam configuration without decay or preprocessing | Second derivation checks spectrum, tracking and constants | Complete theorem not formalized |
| cor:det | O(1/T) deterministic average stationarity, gradient norm tends to zero | Main configuration, bounded exact gradients, small step | Follows from the main bound | Generic scalar finite-sum certificates checked; this corollary not formalized |
| cor:noise | Expected energy bound, vanishing/summable noise, growing batches | Additional MSE and batch hypotheses | Sample cost stated explicitly | Not formalized |
| cor:relative | O(1/T), mean-square and almost-sure stationarity | Strong explicit relative-error threshold | Relative-error absorption argument | Not formalized |
| prop:nomomentum | Sharper descent; linear gap under gradient dominance | beta1=0, deterministic, explicit small step | Descent proof in manuscript | Scalar descent absorption and finite average certificate kernel-checked |
| Convex proposition | O(T^-1/2) gap of average iterate | Attained minimum and bounded trajectory distance | Attainment hypothesis corrected after counterexample | Not formalized |
| prop:spike | Coordinatewise adaptive-step bound | beta2>beta1², zero state; decay excluded from bound | Cauchy-Schwarz argument | Not formalized |
| prop:cycle | Exact nonstationary two-cycle | Legal large constant-step beta=(0,0) mode | Exact and numerical reproduction | Not formalized |
| prop:bias / prop:scalarbias | Expected ascent / bounded-multiplier anti-alignment | Finite-step obstruction only | Analytic construction | Not formalized |
| prop:detectorbias | Exact detector reverses conditional mean | Reachable history, unbiased oracle | Rational/analytic construction and numerical reproduction | Not formalized |
| prop:gc | Permanent nonstationary stalling | Enabled GC, even with both decays and AGC | Invariant argument and Adam/Lion reproduction | Not formalized |

## Explicit non-results

No unconditional guarantee is established for the default positive-decay
configuration, arbitrary fixed-variance stochastic gradients, general Lion,
quantum decay, centralization/AGC, asynchronous groups, resumed state, GPU/DDP
program semantics, or finite precision. No superiority over AdamW is proved.

The final Lean run exited zero: **10 declarations kernel-checked** with Lean
4.19.0 and Mathlib commit `c44e0c8ee63ca166450922a373c7409c5d26b00b`.
`lean/build.log` contains the exact command and axiom reports;
`lean/build_status.json` records the checked source hash. No `sorryAx` or added
axioms are used; the reports list the standard `propext`, `Classical.choice`,
and `Quot.sound`. This does not formally verify the full convergence theorem.
