# Exact mathematical specification of the current implementation

This specification was reconstructed from executable source at Git revision
`b4917163d63d8a5cb2960e29af4b1bccdbf5798d`, including the local working tree.
It is not a transcription of the earlier paper. Line references below refer to
the source snapshot in `source_snapshot.json`. The equations are a real-arithmetic
interpretation; finite precision, alternate kernels and checkpoint caveats are
separate below and in `implementation_trace.md`.

## 1. Indexing, inputs and initial state

Write the parameter vector as tensor blocks $x^j\in\mathbb R^{d_j}$.
All norms in the detector and AGC are the Euclidean norm of the entire tensor,
not a unitwise or layerwise norm. The tensor's original shape is retained for
gradient centralization. Each block belongs to one parameter group, which owns
its hyperparameters. At a global optimizer call $n$, a block is updated only
when `param.grad is not None`. Its local counter $k=k_j(n)$ counts such calls;
bias correction, warmup and the cosine schedule use this counter, not $n$.
Skipped blocks receive neither weight decay nor a state update. A *zero* gradient
is an active observation and increments the counter.

The gradient supplied by the training loop is $u_k^j$. It may already have
been clipped, accumulated, unscaled or averaged across data-parallel ranks.
PsiLogic itself does not sample this gradient or establish its unbiasedness.

At the first active observation the state is lazily created with
$k=0,m_0=v_0=0\in\mathbb R^{d_j}$ and
$A_0=F_0=S_0=s_0=0\in\mathbb R$. Here `m`, `v`, `gn_avg`, `fast`, `slow`,
`soft_chaos` implement $m,v,A,F,S,s$, respectively. All these tensors are
allocated in float32 irrespective of the parameter dtype. The counter is a
Python integer. `soft_chaos` is diagnostic state; its previous value is never an
input to the next update (`optimizer.py:113–126,591–608`).

For the mathematical domain used here, assume nonempty real tensors, finite
inputs and hyperparameters, $\epsilon>0$,
$0\le\beta_1,\beta_2<1$, $\eta_k,\lambda,\gamma,q,c\ge0$,
$p\ge0$, and $0<C\le1$. These are explicit analysis restrictions.
The constructor checks only some of them, and group overrides are not rechecked.
Here $\eta$=`lr`, $\lambda$=`weight_decay`, $q$=`quantum_decay`,
$c$=`agc_clip`, $p$=`p_ext`, $C$=`max_cancel`.

## 2. Gradient preprocessing

Set

$$
 a_k^{\rm AGC}=\begin{cases}
  \min\left(1,\frac{c\max(\Vert{}x_{k-1}\Vert{},10^{-3})}
                        {\max(\Vert{}u_k\Vert{},10^{-6})}\right),&c>0,\\
  1,&c\le0,
 \end{cases}\qquad h_k=a_k^{\rm AGC}u_k.
$$

If `grad_centralize=False`, or the tensor has at most one dimension, let
$g_k=h_k$. Otherwise subtract the mean over dimensions $1,\ldots,r-1$
within every slice of dimension 0: $g_k=\mathcal P h_k$. This is an
orthogonal linear projection in exact arithmetic. The scalar path leaves
`param.grad` untouched; the foreach path applies AGC directly to `param.grad`.
Quantum decay uses $h_k$, before centralization, while all moments, norms and
disagreement use $g_k$ (`optimizer.py:96–110,595–600,704–714`).

There is no separate global norm clipping operation inside `PsiLogic.step`.
FairBench can apply external `clip_grad_norm_` before calling it; this changes
the oracle $u_k$, not these equations.

## 3. Schedule and activation

Let $H$=`gamma_T_max` and $W_{\rm cfg}$=`chaos_warmup`. The resolved
warmup is

$$
 W=\begin{cases}
 W_{\rm cfg},&W_{\rm cfg}\ge0,\\
 \max(500,\lfloor H/20\rfloor),&W_{\rm cfg}<0,\ H>0,\\
 200,&W_{\rm cfg}<0,\ H\le0.
 \end{cases}
$$

Thus every negative warmup setting selects automatic behavior, although the
docstring specifically advertises `-1`. Define

$$
 w_k=\begin{cases}0,&k\le W,\\
 \min(1,(k-W)/\max(1,\lfloor W/4\rfloor)),&k>W,
 \end{cases}
 \quad
 \chi_k=\begin{cases}
 \tfrac12[1+\cos(\pi\min(k/H,1))],&H>0,\\
 1,&H\le0.
 \end{cases}
$$

Then $\gamma_k^{\rm sch}=\gamma\chi_k$ and $q_k^{\rm sch}=q\chi_k$.
The learning rate is not scheduled internally. A training loop may change it.
The chaos branch is active only when $w_k>0$ and
$\gamma_k^{\rm sch}>0$. For $H>0$, it is inactive for every $k\ge H$,
and warmup can occupy the entire requested training horizon.
(`_chaos.py:35–61,114–124`; `optimizer.py:613–617`.)

## 4. Norm history, direction disagreement and soft chaos

Let $r_k=\Vert{}g_k\Vert{}/\sqrt{\max(d,1)}$. On the first active observation,

$$
 A_1=r_1,\qquad F_1=S_1=1.
$$

In particular $F_1=S_1=1$ when $g_1=0$. For $k\ge2$,

$$
 A_k=.99A_{k-1}+.01r_k,\qquad z_k=\frac{r_k}{A_k+\epsilon},
 \qquad F_k=.9F_{k-1}+.1z_k,\qquad S_k=.99S_{k-1}+.01z_k.
$$

The denominator uses the **new** $A_k$. These states update even while the
chaos branch is inactive (`_chaos.py:263–282`; `optimizer.py:623–640`).

Before the momentum update, define

$$
 D_k=\begin{cases}0,&k=1,\\
 \frac12\left[1-\operatorname{clip}_{[-1,1]}
 \frac{\langle g_k,m_{k-1}\rangle}
 {\Vert{}g_k\Vert{}\Vert{}m_{k-1}\Vert{}+\epsilon}\right],&k>1.
 \end{cases}
$$

It is computed only when a block's chaos branch may be active (foreach may
compute it for additional blocks in the same group). With zero gradient or
zero previous momentum at $k>1$, $D_k=1/2$, not zero.
(`_chaos.py:127–151`; `optimizer.py:610–617,739–751`.)

If DDP chaos synchronization is enabled and a process group exists, replace
the new $F_k,S_k$ by their rank means before the remaining formulas. Neither
$A_k$ nor $D_k$ is synchronized. This requires aligned active parameter
lists across ranks and does not by itself guarantee identical trust factors.
All subsequent $F_k,S_k$ notation denotes the possibly synchronized states.

With $\sigma(z)=(1+e^{-z})^{-1}$, set

$$
 E_k=\begin{cases}
 \sigma\!\left(\dfrac{4}{\max(\texttt{tau\_scale}-1,.25)}
                \left[\dfrac{F_k}{S_k+\epsilon}-1\right]\right),
                     &\texttt{adaptive\_tau=True},\\
 \sigma\!\left(4\left[\dfrac{S_k}{\max(\texttt{chaos\_tau},\epsilon)}-1\right]\right),
                     &\texttt{adaptive\_tau=False},
 \end{cases}
 \qquad s_k=\tanh(S_k)\frac{E_k+D_k}{2}.
$$

This is a continuous soft gate, not a hard spike detector. The two symptoms
need not be probabilistically independent. A separate diagnostic labels
`soft_chaos > 0.5` a spike, but never affects the update.
(`_chaos.py:154–186`; `debug.py:51–59`.)

If `gamma_auto=True`, let

$$
 \gamma_k^{\rm eff}=\gamma_k^{\rm sch}\times
 \begin{cases}
 1,&k\le1\text{ or }S_k\ge .1,\\
 \max(S_k/.1,.1),&k>1\text{ and }S_k<.1.
 \end{cases}
$$

Otherwise $\gamma_k^{\rm eff}=\gamma_k^{\rm sch}$. A nonpositive scheduled
gamma is returned unchanged. This scaling reads the new slow state after DDP
synchronization (`_chaos.py:64–83,241–260`; `optimizer.py:644–646`).

When chaos is active, define

$$
 C_*=\min(C,.99),\qquad
 a_k=1-\min\{s_k\gamma_k^{\rm eff}p w_k,C_*\},\qquad
 Q_k=q_k^{\rm sch}w_k(1-s_k).
$$

When inactive, $a_k=1,Q_k=0,s_k=0$ (the last is the stored diagnostic
value). For $q_k^{\rm sch}\le0$, the code omits quantum decay, so $Q_k$
is interpreted as zero. Under the stated domain, $s_k\in[0,1]$ and
$a_k\in[1-C_*,1]$; in particular $a_k\ge .01$. This bound is on the
multiplier, not on the complete adaptive direction or its correlation with
the gradient. The source only clips damping from **above**. For accepted but
negative `p_ext`, $a_k$ can exceed one.

## 5. Adam branch: the canonical default

The moments are

$$
 m_k=\beta_1m_{k-1}+(1-\beta_1)g_k,\qquad
 v_k=\beta_2v_{k-1}+(1-\beta_2)g_k^{\odot2}.
$$

Set $B_{1,k}=1-\beta_1^k$, $B_{2,k}=1-\beta_2^k$. For nonnegative
weight decay, the complete update is

$$
 \boxed{x_k=(1-\eta_k\lambda)
       [\mathbf1-\eta_k Q_k\tanh(|h_k|)]\odot x_{k-1}
       -\eta_k a_k\frac{\sqrt{B_{2,k}}}{B_{1,k}}
                \frac{m_k}{\sqrt{v_k}+\epsilon}.}
$$

Products, absolute values, square roots, $\tanh$ and fractions in this formula
are coordinatewise. If weight decay is nonpositive (possible via group
override), code omits it entirely. The quantum-decay factor is applied after
weight decay and before the adaptive step. Neither multiplicative decay factor
is forced to be nonnegative: the code does not constrain $\eta\lambda$
or $\eta Q_k$. The quantum factor is omitted unless chaos is active even
if `quantum_decay > 0`.

In terms of bias-corrected moments, the adaptive direction is

$$
 a_k\frac{m_k/B_{1,k}}
 {\sqrt{v_k/B_{2,k}}+\epsilon/\sqrt{B_{2,k}}}.
$$

The epsilon is added to $\sqrt{v_k}$ **before** bias correction. Thus at
positive epsilon this is not exactly a formulation of AdamW with a constant
epsilon added to the bias-corrected denominator. (`optimizer.py:499–519,
549–558,619–621`.)

## 6. Lion branch

For `lion_mode=True`, use the same preprocessing, detector, schedules, decay,
and multiplier, but compute

$$
 \ell_k=\operatorname{sign}(\beta_1m_{k-1}+(1-\beta_1)g_k),\qquad
 x_k=(1-\eta_k\lambda)[\mathbf1-\eta_kQ_k\tanh(|h_k|)]\odot x_{k-1}
       -\eta_ka_k\ell_k,
 \qquad m_k=\beta_2m_{k-1}+(1-\beta_2)g_k.
$$

Here $\operatorname{sign}(0)=0$; there is no moment bias correction, and
`v` remains allocated but unused. In particular the stored Lion momentum uses
$\beta_2$, whereas its sign direction uses $\beta_1$.
(`optimizer.py:543–548,619–621,890–896`.)

## 7. Filtration and dependence

Let $\mathcal F_{n-1}$ contain initial parameters/state, past oracle draws,
past external scheduler decisions and all observations before global call
$n$. For a conventional stochastic analysis require learning rates and
hyperparameters at call $n$, and the choice of which blocks receive gradients,
to be $\mathcal F_{n-1}$-measurable. If the active set depends on the new
sample it is instead part of $\mathcal F_n$, and local-time sampling requires
separate assumptions. Let $\mathcal F_n$ additionally include all current
oracle draws and any DDP collective outputs.

Before the update $x,m,v,A,F,S$ are $\mathcal F_{n-1}$-measurable.
Current $u,h,g,r,z,D,A_k,F_k,S_k,s_k,a_k,m_k,v_k,x_k$ are
$\mathcal F_n$-measurable. The current multiplier and diagonal adaptive
preconditioner are generally **not** $\mathcal F_{n-1}$-measurable.
An assumption $\mathbb E[u_k\mid\mathcal F_{n-1}]=\nabla f(x_{n-1})$
does not imply that $g_k$ is unbiased after nonlinear AGC or external
clipping, nor that $a_kg_k$ is unbiased. Centralization gives the projected
conditional mean $\mathcal P\nabla f$, not the full gradient. The parameters
of EMA filters do not create independence. No factorization of conditional
expectations involving current trust or normalization is licensed by the code.

## 8. Execution branches and arithmetic scope

The scalar path is the public CPU path and the path chosen during compilation.
Foreach batches the same real arithmetic for dense supported tensors. Uniform
local counters and homogeneous device/dtype permit vectorized detector updates;
heterogeneous counters use a per-block loop. Fused CUDA has both multi-tensor
and per-tensor Triton implementations and fallback paths; see the separate fused
audit included with this research bundle for exact source references.

Real arithmetic does not imply bitwise equivalence. Norm reductions, parameter
storage casts, intermediate dtypes, fused arithmetic and approximate elementary
functions differ by path. Low precision gradient norms can overflow before
conversion to float32 despite float32 moment allocation. In scalar code the
parameter is written after weight decay, again after quantum decay and again
after the additive update; a fused kernel may retain intermediates in float32
until one final store. The foreach AGC mutation is an observable difference if
the caller reuses a gradient buffer. At `lr=0`, foreach skips Adam's final
division while scalar still evaluates it; nonfinite inputs can therefore behave
differently. No real-arithmetic convergence result here certifies all finite
precision trajectories.

Checkpoint migration creates missing `fast`/`slow` as one and missing
`gn_avg`/`soft_chaos` as zero without resetting existing counters. Thus resumed
legacy state is not necessarily the fresh initialization above. The inherited
PyTorch loader may cast existing saved tensors to parameter dtype; PsiLogic's
`_fill_missing_state` only fills missing keys and does not recast all existing
state. A fresh-state theorem must either exclude these resumes or separately
assume the required state bounds/dtypes (`optimizer.py:378–421`).

The method is not AMSGrad: there is no running coordinatewise maximum of second
moments. There is no internal line search, projection onto a feasible set,
unbiasedness test, NaN/Inf guard, or sparse/complex-gradient convergence contract.
