# Independent review of the full-momentum restricted-configuration theorem

This note gives a second derivation of the main estimates from the implemented
Adam recurrence, checking the candidate bound with constants 9 and 13. Its
conclusions apply to that bound under the conditions below; later manuscript
revisions require a separate check.

AI agents prepared the derivation and the preceding source inspection and
deterministic implementation comparisons. The comparison scripts and recorded
results are in `verification/`. This is an AI-assisted mathematical check, not
external peer review or Lean verification.

**Verdict:** the stated constants and index shifts are valid under the explicit
conditions below. The estimate is pathwise and needs no independence between
the current gradient, trust, or second moment. It does not establish general
fixed-variance stochastic convergence, convergence of the default nonzero-decay
configuration, or a finite-precision theorem.

## 1. Conditions and indexing used in this audit

The update is the current Adam branch with `weight_decay=quantum_decay=agc_clip=0`,
`grad_centralize=False`, `lion_mode=False`, fresh zero moments, and all parameter
blocks receiving a gradient at every step. The first- and second-moment
coefficients are fixed common numbers $\beta\in[0,1)$ and
$\beta_2\in[0,1)$. Epsilon is fixed and strictly positive. The stepsize
$\alpha>0$ is constant. All arithmetic is real. The full implemented
detector, warmup, gamma schedule and optional auto-gamma may remain enabled;
their hyperparameters must lie in the admissible finite domain that ensures
block trust $a_t^j\in[c,1]$, for a fixed $c>0$. For example choose
$c=1-\min(C,.99)$ for a shared max-cancel setting $C\in(0,1]$, with
nonnegative `p_ext` and gamma. For different group caps use the smallest bound.

Let $x_t$ be the point where the gradient at iteration $t\ge1$ is
evaluated, and let $h_t=\nabla f(x_t)$. The observed gradient satisfies
$g_t=h_t+\xi_t$ and $\Vert{}g_t\Vert{}\le G<\infty$ on every considered sample
path. The objective is bounded below by $f_*$, differentiable, and has an
$L$-Lipschitz gradient for a chosen $L>0$. The constants
$G,L,c,\alpha,\epsilon,\beta,\beta_2$ are deterministic and uniform in the
horizon. Fix $\Delta=f(x_1)-f_*<\infty$.

This is a legal configuration of the current code. It is not the unmodified
default configuration: default weight decay is positive, several convenience
presets preprocess gradients, and optional Lion updates obey a different
recurrence. No proof below uses a claim that successful constructor validation
alone implies the required parameter domain.

## 2. Exact update and preconditioner spectrum

The source gives

$$
 m_t=\beta m_{t-1}+(1-\beta)g_t,\quad
 v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},\quad
 \widehat m_t=m_t/(1-\beta^t).
$$

For every coordinate $i$ in block $j$, define

$$
 (H_t)_{ii}=a_t^j\frac{\sqrt{1-\beta_2^t}}
                              {\sqrt{(v_t)_i}+\epsilon}.
$$

Then the exact update is $x_{t+1}=x_t-\alpha H_t\widehat m_t$. This
definition respects the implemented placement of epsilon before second-moment
bias correction; no constant-epsilon bias-corrected Adam formula is substituted.

Since $|(g_t)_i|\le G$, the moment expansion gives

$$
 0\le(v_t)_i=(1-\beta_2)\sum_{s=1}^t\beta_2^{t-s}(g_s)_i^2
 \le G^2(1-\beta_2^t)\le G^2.
$$

Therefore, writing

$$
 a=\frac{c\sqrt{1-\beta_2}}{G+\epsilon}>0,
 \qquad M=\frac1\epsilon,
 \qquad\kappa=\frac Ma,
$$

we obtain $aI\preceq H_t\preceq MI$. In particular $a\le M$ and
$\kappa\ge1$. The lower bound is deliberately loose, but uniform in time
and valid at the first update. It relies on the bounded observations and
positive epsilon; it is not obtained from cancellation bounds alone.

## 3. Bias-corrected averaging weights and the noise term

Define

$$
 w_{t,s}=\frac{(1-\beta)\beta^{t-s}}{1-\beta^t},\qquad 1\le s\le t.
$$

For $\beta=0$ this means $w_{t,t}=1$ and all earlier weights zero.
The weights are nonnegative and sum to one, so
$\widehat m_t=\sum_{s=1}^tw_{t,s}g_s$. Since
$1-\beta^t\ge1-\beta$, each weight satisfies
$w_{t,s}\le\beta^{t-s}$. Consequently

$$
 \sum_{t=s}^T w_{t,s}\le\sum_{r=0}^{T-s}\beta^r
 \le C_\beta:=\frac1{1-\beta}.
$$

Let $n_t=\sum_{s=1}^tw_{t,s}\xi_s$. Convexity of the squared Euclidean
norm and a finite rearrangement of sums yield, pathwise,

$$
 \sum_{t=1}^T\Vert{}n_t\Vert{}^2
 \le\sum_{t=1}^T\sum_{s=1}^tw_{t,s}\Vert{}\xi_s\Vert{}^2
 =\sum_{s=1}^T\Vert{}\xi_s\Vert{}^2\sum_{t=s}^Tw_{t,s}
 \le C_\beta\sum_{s=1}^T\Vert{}\xi_s\Vert{}^2.                 \tag{1}
$$

This estimate treats the errors as arbitrary vectors; it does not require
zero mean, independence across time, or independence from $H_t$.

## 4. Independently derived drift recurrence and convolution bound

Set

$$
 b_t=\sum_{s=1}^tw_{t,s}h_s,\qquad d_t=b_t-h_t,
 \qquad e_t=\widehat m_t-h_t=n_t+d_t.
$$

At the first update $b_1=h_1$, hence $d_1=0$. For $t\ge2$ write

$$
 \rho_t=\frac{\beta(1-\beta^{t-1})}{1-\beta^t}\in[0,\beta].
$$

The weights give $b_t=\rho_tb_{t-1}+(1-\rho_t)h_t$, and subtraction of
$h_t$ gives the exact identity

$$
 d_t=\rho_t\bigl(d_{t-1}+h_{t-1}-h_t\bigr).          \tag{2}
$$

Unrolling (2), with $\delta_s=h_{s-1}-h_s$, gives

$$
 d_t=\sum_{s=2}^t\left(\prod_{r=s}^t\rho_r\right)\delta_s,
 \qquad
 \Vert{}d_t\Vert{}\le\sum_{s=2}^t\beta^{t-s+1}\Vert{}\delta_s\Vert{}.
$$

For $\beta>0$, weighted Cauchy--Schwarz gives

$$
 \Vert{}d_t\Vert{}^2
 \le \left(\sum_{s=2}^t\beta^{t-s+1}\right)
       \sum_{s=2}^t\beta^{t-s+1}\Vert{}\delta_s\Vert{}^2
 \le K\sum_{s=2}^t\beta^{t-s+1}\Vert{}\delta_s\Vert{}^2,
 \quad K:=\frac\beta{1-\beta}.
$$

After summing over $t$, the remaining column sum is also at most $K$,
so

$$
 \sum_{t=1}^T\Vert{}d_t\Vert{}^2\le K^2\sum_{s=2}^T\Vert{}h_s-h_{s-1}\Vert{}^2. \tag{3}
$$

For $\beta=0$, (2) gives $d_t=0$ and (3) holds with $K=0$.
No initial-state error is missing: this argument explicitly uses zero initial
momentum and the first bias-corrected average $\widehat m_1=g_1$.

Lipschitz continuity and the actual preceding update give

$$
 \Vert{}h_s-h_{s-1}\Vert{}\le L\Vert{}x_s-x_{s-1}\Vert{}
 =L\alpha\Vert{}H_{s-1}\widehat m_{s-1}\Vert{}
 \le L\alpha M\Vert{}\widehat m_{s-1}\Vert{}.
$$

Thus, with $Q=K^2L^2\alpha^2M^2$,

$$
 \sum_{t=1}^T\Vert{}d_t\Vert{}^2
 \le Q\sum_{t=1}^{T-1}\Vert{}\widehat m_t\Vert{}^2
 \le Q\sum_{t=1}^T\Vert{}\widehat m_t\Vert{}^2.              \tag{4}
$$

Using $\Vert{}u+v\Vert{}^2\le2\Vert{}u\Vert{}^2+2\Vert{}v\Vert{}^2$ in $e_t=n_t+d_t$, (1)
and (4) imply

$$
 \sum_{t=1}^T\Vert{}e_t\Vert{}^2
 \le2C_\beta\Xi_T+2QS_T,
 \quad\Xi_T=\sum_{t=1}^T\Vert{}\xi_t\Vert{}^2,
 \quad S_T=\sum_{t=1}^T\Vert{}\widehat m_t\Vert{}^2.         \tag{5}
$$

## 5. Descent, absorption and all numerical constants

Smoothness gives

$$
 f(x_{t+1})\le f(x_t)-\alpha\langle h_t,H_t\widehat m_t\rangle
                   +\frac{L\alpha^2}{2}\Vert{}H_t\widehat m_t\Vert{}^2.
$$

Substitute $h_t=\widehat m_t-e_t$, use the spectral bounds, and apply
$M\Vert{}e_t\Vert{}\Vert{}\widehat m_t\Vert{}\le(a/2)\Vert{}\widehat m_t\Vert{}^2+
(M^2/(2a))\Vert{}e_t\Vert{}^2$. This yields

$$
 f(x_{t+1})\le f(x_t)
 -\alpha\left(\frac a2-\frac{L\alpha M^2}{2}\right)
                  \Vert{}\widehat m_t\Vert{}^2
 +\frac{\alpha M^2}{2a}\Vert{}e_t\Vert{}^2.                 \tag{6}
$$

The bound is pointwise in the sample path. In particular there is no step
replacing $\mathbb E[H_t\xi_t]$ by zero.

Summing (6), using $f(x_{T+1})\ge f_*$ and then (5), gives

$$
 \alpha\left(\frac a2-\frac{L\alpha M^2}{2}-\frac{M^2Q}{a}\right)S_T
 \le\Delta+\frac{\alpha M^2C_\beta}{a}\Xi_T.       \tag{7}
$$

Assume the proposed stepsize restriction

$$
 0<\alpha\le\frac{a}{4LM^2\max(1,K)}.             \tag{8}
$$

Then

$$
 \frac{L\alpha M^2}{2}\le\frac a8,
 \quad Q\le\frac{a^2}{16M^2},
 \quad\frac{M^2Q}{a}\le\frac a{16}.
$$

The coefficient inside parentheses in (7) is therefore at least
$a/2-a/8-a/16=5a/16$, which is strictly stronger than the asserted lower
bound $a/4$. It is valid to retain the simpler latter bound, obtaining

$$
 S_T\le\frac{4\Delta}{\alpha a}
       +4C_\beta\frac{M^2}{a^2}\Xi_T.             \tag{9}
$$

Finally, $h_t=\widehat m_t-e_t$ and (5) give

$$
 \sum_{t=1}^T\Vert{}h_t\Vert{}^2
 \le2S_T+2\sum_{t=1}^T\Vert{}e_t\Vert{}^2
 \le(2+4Q)S_T+4C_\beta\Xi_T.
$$

Since $a\le M$, (8) implies $Q\le1/16$ and hence
$2+4Q\le9/4$. Substituting (9), then using $M^2/a^2\ge1$, gives

$$
 \boxed{\displaystyle
 \sum_{t=1}^T\Vert{}\nabla f(x_t)\Vert{}^2
 \le\frac{9\Delta}{\alpha a}
       +13C_\beta\frac{M^2}{a^2}\sum_{t=1}^T\Vert{}\xi_t\Vert{}^2.} \tag{10}
$$

The constant 9 is $(9/4)\cdot4$. The constant 13 bounds
$9\kappa^2+4$ by $13\kappa^2$. Both constants are conservative;
sharpening them is unnecessary for correctness.

## 6. What the deterministic and stochastic conclusions actually imply

For exact deterministic gradients, $\xi_t=0$. Inequality (10) holds
uniformly for every $T$, so the nonnegative series
$\sum_t\Vert{}\nabla f(x_t)\Vert{}^2$ is finite. Consequently
$\Vert{}\nabla f(x_t)\Vert{}\to0$, and its squared norm averaged over the first
$T$ iterates is at most $9\Delta/(\alpha aT)$. This is not a proof
that $x_t$ converges, that its limit is a global minimizer, or that every
individual gradient has the same $O(1/T)$ bound.

For stochastic errors, suppose the required moments are finite. Taking
expectations in the pathwise inequality is legitimate and gives

$$
 \frac1T\sum_{t=1}^T\mathbb E\Vert{}\nabla f(x_t)\Vert{}^2
 \le\frac{9\Delta}{\alpha aT}
       +\frac{13C_\beta\kappa^2}{T}
         \sum_{t=1}^T\mathbb E\Vert{}\xi_t\Vert{}^2.         \tag{11}
$$

If $x_1$ is random, replace $\Delta$ by
$\mathbb E[f(x_1)-f_*]$ and assume that expectation finite.

Vanishing Cesàro mean of the error energy gives vanishing **average** expected
squared gradients. It does not, from this inequality alone, give pointwise
convergence of $\mathbb E\Vert{}\nabla f(x_t)\Vert{}^2$. Summable error energy,
$\sum_t\mathbb E\Vert{}\xi_t\Vert{}^2<\infty$, does give finite expected total
gradient energy. Monotone convergence then yields a finite gradient-energy
series almost surely, so gradients tend to zero almost surely and in mean
square. This is an explicit, justified use of an expectation/limit interchange.

Under only a constant error bound $\mathbb E\Vert{}\xi_t\Vert{}^2\le\sigma^2$,
(11) has the nonzero floor $13C_\beta\kappa^2\sigma^2$. It must not be
described as a convergence-to-stationarity theorem in that regime. The
inequality alone also does not prove that every such run fails to converge.

## 7. Relative-noise corollary

Suppose, for all $t$,
$\mathbb E\Vert{}\xi_t\Vert{}^2\le r^2\mathbb E\Vert{}\nabla f(x_t)\Vert{}^2$, and set
$A=13C_\beta\kappa^2$. A conditional version of this assumption also
suffices by the tower property. With $Ar^2<1$, expectation of (10) gives

$$
 (1-Ar^2)\sum_{t=1}^T\mathbb E\Vert{}\nabla f(x_t)\Vert{}^2
 \le\frac{9\Delta}{\alpha a}.
$$

Thus the average rate is $O(1/T)$, and the total expected squared-gradient
series is finite. As above, gradients converge to zero in mean square and
almost surely. No conditional unbiasedness is required for this corollary,
but the relative-error condition is strong and must be stated, not inferred
from standard bounded-variance noise. The strict inequality $Ar^2<1$
is essential for this absorption argument.

## 8. Growing minibatches: exact conditions and rates

To derive a batch-based variance bound, separately assume that, conditionally
on the past and current iterate, a batch consists of $B_t$ independent
oracle samples with mean $h_t$ and individual conditional variance at most
$\sigma^2$. The batch size must be known before these samples are drawn.
Then their average has error energy at most $\sigma^2/B_t$, because the
conditional cross terms vanish. That reasoning uses independence *between
oracle samples*, not independence of the adaptive update from the gradient.
The required pathwise observation bound $\Vert{}g_t\Vert{}\le G$ remains an
additional assumption; a variance bound alone does not imply it. A uniform
bound on each individual oracle gradient is one sufficient way to ensure it.

If $B_t\ge b t^r$ with $b>0,r>0$, comparison with an integral gives

$$
 \sum_{t=1}^Tt^{-r}\le
 \begin{cases}
 1+(T^{1-r}-1)/(1-r),&0<r<1,\\
 1+\log T,&r=1,\\
 1+1/(r-1),&r>1.
 \end{cases}
$$

Substitution into (11) yields average expected squared-gradient bounds
$O(T^{-r})$ for $0<r<1$, $O((1+\log T)/T)$ for $r=1$, and
$O(1/T)$ for $r>1$. In particular $r=1/2$ gives the
$O(T^{-1/2})$ *iteration* rate, but uses growing batches. Under comparable
upper and lower batch growth, total oracle samples are of order
$T^{r+1}$, so this is not a fixed-batch oracle-complexity improvement.
Only the $r>1$ bound ensures summable noise by this argument and hence
the stronger pointwise/a.s. conclusions stated above.

## 9. Adversarial checks and remaining scope

| Potential failure | Audit outcome |
|---|---|
| Factorizing current trust and stochastic gradient | Not used; all estimates through (10) are pathwise. |
| Bias-correction index missing at step 1 | Explicit weights and $d_1=0$ account for it. |
| Treating EMA error as independent noise | Not used; (1) is a deterministic convexity estimate. |
| Missing initial momentum term | Fresh $m_0=0$ is required; resumed arbitrary momentum is excluded. |
| Drift recurrence coefficient off by one | Product $\prod_{r=s}^t\rho_r$ has $t-s+1$ factors; the resulting constant is $K=\beta/(1-\beta)$. |
| Bounding current drift by the wrong iterate step | (4) uses moments through $T-1$, then safely enlarges to $T$. |
| Epsilon denominator or upper bound wrong | Uses actual uncorrected $\sqrt v+\epsilon$; upper spectral bound is $1/\epsilon$. |
| Trust lower bound silently assumes positive amplification | Nonnegative `p_ext` is explicit; code does not enforce it. |
| Absorption constants have wrong sign | Parenthesis in (7) is at least $5a/16$; claimed $a/4$ is valid. |
| Deterministic theorem contradicts noisy Adam failures | It uses a restrictive epsilon-dependent stepsize and bounded observed gradients; it does not assert fixed-variance stochastic convergence. |
| Treating a vanishing average as pointwise convergence | Rejected unless summability or the relative-noise argument supplies finite total energy. |
| Calling the growing-batch rate sample-efficient | Rejected; batch growth and sample complexity must be disclosed. |
| Extending automatically to weight decay, AGC, GC or Lion | Rejected; these are excluded in the theorem audited here. |
| Treating float32 parity checks as real-arithmetic proof | Rejected; numerical correspondence checks and mathematical proofs are distinct. |

The principal practical limitation is the conservative stepsize in (8): since
$M=1/\epsilon$, ordinary small epsilon may make the permitted stepsize
extremely small. This weakens usefulness, not validity. The theorem proves a
conditional guarantee for an existing configuration, not that usual training
hyperparameters satisfy it. No superiority over AdamW follows from it.

## 10. Follow-up manuscript checks

The following findings were recorded from the reviewing AI agent's follow-up
messages. They supplement the derivation above; the agent's review session
ended before this addendum was assembled.

1. The final, stronger spectrum lower bound
   `a = c/(G+eps/sqrt(1-beta2))` is valid: divide the exact denominator by
   sqrt(1-beta2^t), use corrected variance at most G², then bound the corrected
   epsilon by eps/sqrt(1-beta2). It is positive, at most M, and dominates a_old.
   The remainder of the proof uses only the spectrum inequality, so constants
   9 and 13 and their derivation are unchanged.
2. The initial convex corollary lacked `f_*=f(x_*)`. The reviewer produced
   f=x²/2, x_t=x_*=0, f_*=-1, R=0 as a counterexample. The final manuscript
   repairs this hypothesis and the strongly convex discussion.
3. The predictable-scaling theorem needed an explicit integrability argument.
   The final proof derives finite second moments and objective integrability
   inductively from a deterministic start, bounded matrix norm, Lipschitz gradient,
   and conditional variance bound. The reviewer checked this insertion.
4. The exact-detector counterexample's rational branch values and reachable
   quadratic construction were checked in the follow-up. Positive alpha is now
   stated explicitly in its construction.
5. Trust/spike proposition domains, advancing local clocks, decay-domain wording,
   and precise source-line endpoints were clarified as requested.

This audit is mathematical review, not Lean verification. No claim is made that
review by multiple language-model agents alone proves the theorem. The complete
inequalities and proofs remain available for expert inspection.
