# Assumptions and theorem coverage

All mathematical results concern exact real arithmetic. Successful Python
construction does not establish these hypotheses. State tensors in actual
executions are normally fp32, including with fp64 parameters.

## Main theorem: selectable canonical Adam configuration

- A finite nonempty partition of parameters into nonempty real tensors; every
  tensor receives a gradient at every update. No external parameter mutations.
- Single-process scalar recurrence, ordinary zero initialized moments, the
  implemented first-step detector seed, and no arbitrary resumed checkpoint.
- `lion_mode=False`, `grad_centralize=False`, `agc_clip=0`,
  `weight_decay=0`, `quantum_decay=0`.
- Fixed common `betas=(beta,beta2)` with both entries in `[0,1)`, fixed common
  `eps>0`, and a common constant learning rate alpha.
- Full exact detector retained: finite nonnegative gamma and p_ext; max_cancel
  in `(0,1]`; finite threshold settings; fixed integer warmup and horizon;
  optional adaptive_tau, cosine scheduling and gamma_auto as implemented.
- f is bounded below by a finite f_* and has L-Lipschitz gradient, L>0.
- Every supplied concatenated gradient has norm at most a deterministic G.
- Define c as the smallest block value `1-min(max_cancel,.99)`,
  a = c / (G + eps / sqrt(1-beta2)), M=1/eps,
  K=beta/(1-beta), C_beta=1/(1-beta).
- Require `0 < alpha <= a/(4 L M^2 max(1,K))`.

For xi_t=g_t-grad f(x_t), the pathwise result is

    sum_{t=1}^T ||grad f(x_t)||^2
      <= 9 [f(x_1)-f_*]/(alpha*a)
         + 13 C_beta (M/a)^2 sum_{t=1}^T ||xi_t||^2.

No unbiasedness or independence is needed in this inequality. Its deterministic
specialization gives O(1/T) average squared gradient and gradient norm tending
to zero. It does not prove convergence of parameter values or global optimality.

The step restriction is very conservative: at L=G=1, beta=.9, beta2=.999,
eps=1e-8 and c=.95 it is approximately 2.64e-18. No claim is made that ordinary
training steps such as 1e-3 satisfy it. This is a sufficient condition, not a
necessary condition or an instability threshold.

## Stochastic conclusions

With deterministic x_1 and the main conditions holding almost surely:

| Additional hypothesis | Conclusion |
|---|---|
| E‖xi_t‖² <= sigma_t² | Expected finite-horizon energy bound with sum sigma_t² |
| Cesaro mean of sigma_t² tends to zero | Average expected squared gradient tends to zero |
| Sum sigma_t² finite | Mean-square and almost-sure gradient convergence |
| E[‖xi_t‖² given past] <= r² ‖grad f(x_t)‖², with 13 C_beta (M/a)² r² < 1 | O(1/T) average bound and mean-square/almost-sure gradient convergence |
| Fixed positive variance bound only | A nonvanishing upper-bound term; no general stationarity conclusion |

For minibatches, conditionally independent unbiased samples of variance at most
sigma² give variance at most sigma²/B. The pathwise bound G must still be
assumed. B=ceil(sqrt(T)) gives O(T^-1/2) in iterations at T*ceil(sqrt(T)) sample
evaluations. Neither fixed-batch optimality nor an advantage over AdamW follows.

## Simplified and specialized results

The predictable-scaling theorem replaces the current detector, momentum and
variance update by a past-measurable symmetric matrix B_t with
0<aI<=B_t<=MI. Under unbiased conditional gradients and bounded conditional
variance it gives O(T^-1/2) with a horizon-dependent step. This is a modified
method, not a mode implemented by a PsiLogic flag.

The no-momentum deterministic proposition is an actual configuration
(beta1=0), with a less restrictive step a/(L M²). Gradient dominance adds a
linear objective rate when alpha*a*mu<=1. For the strongly convex interpretation
f_* must equal the attained minimum. The convex full-momentum consequence
also requires f_*=f(x_*), an attained minimizer, and a trajectory-distance bound.

The coordinate spike bound needs beta2>beta1², beta1,beta2 in [0,1), epsilon>0,
zero moments, positive step, and trust in [0,1]. It bounds only the adaptive
step; it is not a noise-robustness or superiority theorem.

## Excluded from the main theorem

Positive weight decay (including the plain constructor default), quantum decay,
AGC, gradient centralization, Lion, missing/asynchronous gradients, arbitrary
per-group moment coefficients and step schedules, checkpoint state, distributed
collectives, finite precision, sparse/complex tensors, noncontiguous fused
layouts, mixed-device fused groups, and nonfinite inputs. Some may admit
separate theorems; none is silently covered here.
