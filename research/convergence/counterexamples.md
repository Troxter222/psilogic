# Counterexamples and proof obstructions

These results use the reconstructed exact-real recurrence, checked against the
executable source before this analysis. The numerical checks in
`counterexample_checks.py` are supporting calculations and implementation probes;
they are not the proofs. Unless stated otherwise, use dense scalar parameters,
no AGC, no centralization, no weight/quantum decay, and no distributed state
synchronization. Initial moment buffers are zero, and epsilon is positive.

## C1. A nonstationary exact two-cycle at constant learning rate

**Statement.** Set `betas=(0,0)`, `gamma=0`, `weight_decay=0`,
`quantum_decay=0`, `grad_centralize=False`, `agc_clip=0`, and Adam mode.
For any epsilon $e>0$ and constant learning rate $\eta>2e$, the
objective $f(x)=x^2/2$, exact gradient oracle, and initialization
$x_1=a=\eta/2-e>0$ give
$x_t=(-1)^{t-1}a$. Therefore the average squared gradient is exactly
$a^2>0$ at every horizon, and the iterates do not converge.

**Proof.** The beta settings imply $m_t=g_t=x_t$,
$v_t=g_t^2=x_t^2$, and both bias-correction factors equal one.
Gamma zero disables cancellation and quantum decay. Thus

$$
 F(x)=x-\eta x/(|x|+e).
$$

Since $a+e=\eta/2$, $F(a)=a-2a=-a$.
The function $F$ is odd, so $F(-a)=a$. Induction proves the cycle.
Because $\nabla f(x_t)=x_t$, every squared gradient equals $a^2$.
The objective is smooth with Lipschitz constant one, strongly convex, and
bounded below; gradients along this trajectory are bounded and noiseless.
All selected hyperparameters satisfy constructor validation. ∎

This disproves convergence over all legal constant learning rates, even in
a very simple subclass of the actual algorithm. It does not disprove a
sufficiently small-step or diminishing-step theorem, and it does not identify
the behavior of the default beta values.

## C2. Unbiased stochastic gradients can produce expected ascent

**Statement.** At the first step, set `betas=(0,0)`, epsilon $e=1/100$,
and $f(x)=(x+1/10)^2/2$ with $x_1=0$. Let

$$
g=10\text{ with probability }1/10,\qquad
g=-1\text{ with probability }9/10.
$$

The oracle is unbiased at this point because $\mathbb E g=1/10=f'(0)$.
Either disable chaos, or activate it with fixed positive gamma,
`chaos_warmup=0`, nonnegative `p_ext`, and the implemented damping cap.
For every learning rate $\eta>0$, this step strictly increases the
objective in expectation.

**Proof.** On the first step the norm-EMA helper sets fast and slow to one,
while disagreement is explicitly zero. The soft signal, scheduled gamma,
and trust consequently do not depend on which gradient was sampled.
Write the resulting trust as $\rho\in[0.01,1]$, with $\rho=1$
when chaos is off. Although the baseline norm depends on $|g|$, it is
not used in the first-step trust formula. The update is
$x_2=-\eta\rho U$, where $U=g/(|g|+e)$. Exactly,

$$
\mathbb E U
=\frac{100}{1001}-\frac{90}{101}
=-\frac{79990}{101101}<0.
$$

Expanding the quadratic gives

$$
\mathbb E[f(x_2)-f(0)]
=-\frac{\eta\rho}{10}\mathbb E U
 +\frac{\eta^2\rho^2}{2}\mathbb E U^2>0.
$$

Both terms are strictly positive. ∎

This is a **one-step anti-descent result**, not a proof of asymptotic
nonconvergence. It invalidates an argument that conditional unbiasedness of
$g$ automatically yields alignment after current-gradient normalization.
It remains valid with the exact active chaos mechanism on the first step.
An oracle unbiased everywhere for this quadratic is obtained by setting
$g(x)=x+1/10+\xi$, where $\xi=99/10$ with probability $1/10$
and $\xi=-11/10$ with probability $9/10$. Its variance is finite
and independent of $x$.

## C3. The actual detector can reverse the conditional mean gradient

This isolates dependence in cancellation itself, before Adam normalization.
Unlike a hypothetical bounded multiplier, the multiplier here is computed by
the exact implemented detector from a reachable history.

Set `betas=(0,0)`, epsilon $1/100$, `gamma=1`, `p_ext=1`,
`chaos_warmup=0`, `gamma_T_max=0`, `gamma_auto=False`,
`adaptive_tau=True`, `tau_scale=2`, and `max_cancel=1`.
After a first-step gradient $-1$, old momentum is $-1$ and
the baseline norm, fast EMA, and slow EMA are all one. At the second step
let $g=10$ with probability $1/10$ and $g=-1$ otherwise.
Again $\mathbb E[g\mid\mathcal F_1]=1/10$.

**Claim.** For the actual second-step trust $\rho(g)$,

$$
\mathbb E[\rho(g)g\mid\mathcal F_1]<-463/2000<0.
$$

**Proof.** On the positive branch, the updated baseline is $109/100$,
the normalized norm is $100/11$, and

$$
\mathrm{fast}_+=199/110,\quad
\mathrm{slow}_+=1189/1100,\quad
\frac{\mathrm{fast}_+}{\mathrm{slow}_++e}=199/120>1,\quad
d_+=2001/2002>99/100.
$$

Consequently the sigmoid excess is greater than $1/2$, the slow
state exceeds one, and

$$
s_+>\tfrac34\cdot\tfrac12(\tfrac12+\tfrac{99}{100})=447/800.
$$

Here $\tanh(1)>3/4$: this is equivalent to $\exp(2)>7$, and
$\exp(1)>1+1+1/2+1/6=8/3$ implies
$\exp(2)>64/9>7$. The cap is $99/100>447/800$, so
$\rho_+=1-\min(s_+,99/100)<353/800$.

On the negative branch, the updated baseline remains one and

$$
\mathrm{fast}_-=1009/1010,\quad
\mathrm{slow}_-=10099/10100,\quad
\frac{\mathrm{fast}_-}{\mathrm{slow}_-+e}=1009/1020<1,\quad
d_-=1/202<1/200.
$$

The excess is less than $1/2$, and $\tanh(\mathrm{slow}_-)<1$.
Thus $s_-<(1/2)(1/2+1/200)=101/400$ and
$\rho_->299/400$. Combining the bounds,

$$
\mathbb E[\rho g\mid\mathcal F_1]
=\rho_+-\tfrac9{10}\rho_-
<\tfrac{353}{800}-\tfrac9{10}\tfrac{299}{400}
=-\tfrac{463}{2000}.
$$

No independence or factorization has been used. ∎

**Attainability with an unbiased oracle for a smooth lower-bounded objective.**
Fix any positive learning rate $\eta$. Let $\rho_1>0$ denote the
deterministic first-step trust under the above settings and gradient $-1$.
Starting at zero, the first update is
$\delta=\eta\rho_1/(1+e)>0$. Set
$L=11/(10\delta)$ and
$f(x)=\tfrac L2(x-1/L)^2$. Then $f'(0)=-1$ and
$f'(\delta)=1/10$. Define the oracle to return the exact gradient
except at $x=\delta$, where it returns the stated two-point distribution.
This oracle is unbiased at every point and has conditional variance at most
$1089/100$. It realizes the required history with probability one.
The objective is globally smooth and bounded below.

This example also yields anti-alignment for the complete normalized direction:

$$
\mathbb E\left[\rho\frac{g}{|g|+e}\mid\mathcal F_1\right]
<\frac1{10}\frac{353}{800}
  -\frac9{10}\frac{299}{400}\frac{99}{100}<0,
$$

since $10/(10+e)<1$ and $1/(1+e)>99/100$.
As in C2, expanding the quadratic proves strict conditional expected objective
increase at this second step. This remains a finite-step obstruction; it does
not by itself establish the asymptotic behavior of the constructed oracle.

## C4. Gradient centralization can remove the entire useful gradient

**Statement.** Let a parameter have shape $1\times2$, initialize
$X_1=(0,0)$, and minimize
$f(X)=\tfrac12\Vert{}X-(1,1)\Vert{}_F^2$ using the exact gradient.
Enable gradient centralization and initialize moments as implemented.
For every learning-rate schedule, both Adam and Lion variants remain at zero,
provided scalar operations remain finite and denominators are well-defined.
This holds with whole-tensor AGC and with nonzero weight decay, quantum decay,
and chaos parameters. Yet $\Vert{}\nabla f(X_t)\Vert{}_F^2=2$ for all $t$.

**Proof.** Suppose $X_t=0$ and old momentum and variance are zero.
The raw gradient is $(-1,-1)$. AGC, if active, multiplies both entries
by the same finite scalar; the resulting row is still constant.
Centralization subtracts the row mean, giving $g_t=(0,0)$.
All moment updates therefore remain zero, and both the Adam direction and
Lion sign direction are zero. Each decay operation multiplies the zero
parameter by a finite scalar or vector, leaving it zero. Thus $X_{t+1}=0$
and the moments remain zero. These statements hold initially; induction
establishes the invariant. The original objective gradient at zero is always
$(-1,-1)$, with squared norm two. ∎

This is an asymptotic nonstationarity counterexample even for a strongly convex
objective, noiseless gradients, and diminishing learning rates. It rules out
an unconditional full-gradient stationarity theorem covering enabled
centralization. A theorem for the projected gradient or for an objective
whose gradient lies in the centralization subspace is a different statement.

## C5. AGC does not preserve unbiasedness or gradient alignment

At parameter $x=1$, set `agc_clip=1`; the implemented clipping threshold
is then one. For $f(x)=(x-9/10)^2/2$, the oracle from C2 is unbiased at
this point. AGC maps $10$ to $1$ and $-1$ to $-1$.
Hence $\mathbb E[\mathrm{AGC}(g)]=1/10-9/10=-4/5$, which has the
opposite sign to $f'(1)=1/10$. This follows directly from the clipping
formula, including its norm floors, since both gradient norms exceed
$10^{-6}$ and the parameter norm exceeds $10^{-3}$.
With beta values zero and chaos disabled, the normalized expected direction
is $-4/[5(1+e)]<0$, and the quadratic expansion as in C2 proves expected
ascent. This is a bias/one-step obstruction, not an asymptotic theorem.

## C6. Decoupled weight decay changes the stationarity target

Take $f(x)=(x-a)^2/2$ with $a\ne0$, initialize $x=a$, and
set $\lambda>0$, $\eta>0$, zero initial moments, quantum decay zero,
and positive epsilon. The first gradient and adaptive direction are zero.
The implemented weight decay alone gives
$x^+=(1-\eta\lambda)a$, so
$f(x^+)-f(a)=\eta^2\lambda^2a^2/2>0$.
Thus a minimum of $f$ need not be a fixed point of the update. This
does not rule out convergence to a decay-shifted equilibrium.

Moreover the adaptive update with decoupled decay generally cannot be
rewritten as applying the same adaptive method to
$f+\lambda\Vert{}x\Vert{}^2/2$: the decay term is not multiplied by the same
time-dependent preconditioner or trust. Any proposed regularized objective
requires its own derivation.

## C7. Constructor acceptance is weaker than theorem hypotheses

* **Zero epsilon.** Epsilon is not validated in `_validate_hyperparameters`.
  Set epsilon zero, gamma zero, a zero gradient, and positive learning rate
  in Adam mode. Then $m=v=0$ and the update evaluates $0/0$.
  The real recurrence is undefined and scalar PyTorch returns NaN.
* **Negative `p_ext`.** The implementation clamps damping only above. With
  active chaos, a positive soft signal, and `p_ext=-M<0`, trust equals
  $1+M\gamma_{\rm eff}hs>1$. Varying $M$ makes it arbitrarily
  large. The usual upper bound $\rho\le1$ needs the additional
  nonnegativity hypothesis; it is not guaranteed by parameter validation.
* **Large decay products.** Nonnegative weight decay and learning rate do
  not imply $1-\eta\lambda\ge0$. The analogous quantum factor can
  also become negative. Therefore the word “decay” does not itself establish
  a contraction, boundedness, or preservation of parameter signs.
* **NaNs and infinities.** Some simple comparison validations accept NaNs
  or infinity. The real-arithmetic theory assumes finite hyperparameters
  and states explicitly. It does not certify nonfinite executions.

## What these examples establish

| Example | Mathematically established | Does not establish |
|---|---|---|
| C1 | Exact nonstationary periodic orbit for legal fixed-step subclass | Failure for every step schedule/beta pair |
| C2 | Unbiased current-gradient normalization can cause expected ascent | Asymptotic stochastic divergence |
| C3 | Exact implemented cancellation can reverse conditional mean direction | Asymptotic failure for every stochastic oracle |
| C4 | Full GC-enabled algorithm can remain nonstationary forever | Failure of a projected-gradient theorem |
| C5 | AGC can reverse the mean gradient and cause expected ascent | Failure under explicitly controlled clipping bias |
| C6 | Positive decay can move away from an original-objective minimizer | Failure to converge to another equilibrium |
| C7 | Accepted configurations can violate positivity/finiteness premises | Failure under appropriate explicit premises |

These counterexamples forbid a uniform theorem for every accepted option and
parameter. They do not contradict restricted convergence theorems whose
assumptions explicitly exclude the exhibited mechanisms.
