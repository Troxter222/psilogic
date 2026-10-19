# Backend execution and arithmetic trace

This report records executable source inspection for the local working tree at
Git revision `b4917163d63d8a5cb2960e29af4b1bccdbf5798d`. It does not assert
GPU-tested equivalence. All line references below refer to that inspected
working tree. No canonical optimizer source was modified during this audit.
Later publication edits to comments and docstrings move some of those lines;
the function names identify the corresponding code in the current tree.
No applicable `AGENTS.md` was found in the repository/ancestor chain or
`psilogic/_cuda`. The source-of-truth report maintained separately records the
full initial working-tree status.

## Dispatch and supported scope

| Decision | Executable source | Consequence |
|---|---|---|
| Compilation forces scalar | `psilogic/optimizer.py:928–937` | `torch.compiler.is_compiling()` bypasses foreach/fused dispatch. |
| CUDA detection is `any(p.is_cuda)` | `psilogic/optimizer.py:939–945` | Selection does not require every parameter to be on CUDA, nor every active parameter to be CUDA. |
| Fused has priority | `psilogic/optimizer.py:940–951` | Requires constructor-level and group-level fused flags plus some CUDA parameter. |
| Fused availability | `psilogic/_cuda/__init__.py:8–30` | Requires importable Triton and `torch.cuda.is_available()`. This is not a device architecture, dtype, or layout check. |
| Unavailable fused fallback | `psilogic/optimizer.py:963–974` | Calls `_step_foreach` directly, regardless of `group['use_foreach']` or `_FOREACH_AVAILABLE`. |
| Active parameters | `psilogic/_cuda/step.py:466–476` | Missing gradients skip every update, including decay and local time advancement. |
| Batched fused eligibility | `psilogic/_cuda/step.py:478–508` | Same local step and same `(device,dtype)` are required for batched orchestration; otherwise per-tensor fused orchestration is used. |
| Multi-tensor dtype/size | `psilogic/_cuda/step.py:127–136,153–173`; `psilogic/_cuda/kernels.py:604–613` | `float32`, `float16`, `bfloat16`; gradient/raw-gradient dtype must match parameter; tensor lengths/block offsets must fit signed 32-bit indices. |
| Fallback after table rejection | `psilogic/_cuda/step.py:627–668` | Rejection chooses per-tensor Triton launches, not the scalar Python optimizer. |
| Zero-size tensor kernels | `psilogic/_cuda/kernels.py:567–569,683–684` | No parameter update launch, but host state processing has already occurred. |

The mathematically corresponding fused configuration therefore requires dense,
contiguous parameter and state tensors, ordinary real floating dtypes, and CUDA
devices supported by the installed Triton stack. These conditions are sufficient
scope restrictions for the analysis, not validations supplied by the code.
Sparse gradients are not specifically rejected or implemented by the fused path.

**Layout defect visible in executable indexing.** `_maybe_contiguous` applies
only to gradient buffers (`step.py:51–52,350–353,518–525`), while the kernels
load/store parameters and moments as `ptr + offsets`, with no strides
(`kernels.py:249–252,287,321–343,373`). State initialization preserves parameter
memory format through `zeros_like` (`optimizer.py:119–120`). A transposed dense
parameter consequently pairs a row-major contiguous gradient with physical
parameter storage order; a sliced/non-dense parameter can additionally have
physical gaps. Such inputs do not implement the stated logical-coordinate
recurrence. The same layout assumption affects the multi-chaos momentum
reduction (`kernels.py:398–415`). This is a source-level finding; actual CUDA
execution requires separate hardware validation.

**Mixed-device limitation.** A CPU/CUDA group passes the top-level `any CUDA`
check and reaches per-tensor fused launches. No check routes CPU parameters back
to scalar processing (`step.py:482–507,417–437`). A group whose CUDA parameters
have no gradients can even leave an all-CPU active set on the fused path. The
source therefore does not justify a guarantee for mixed-device groups.

## Real-arithmetic recurrence implemented by the active kernels

Fix one active parameter block, with pre-step parameter $x$, momentum $m$,
variance $v$, raw gradient $z$, local time $t$, learning rate $\eta$,
weight decay $\lambda$, and $\epsilon$. All vector products below are
coordinatewise. Quantities are interpreted in exact real arithmetic in this
section; the next section records why this is not a bitwise implementation
claim.

Preprocessing computes

$$
c=\begin{cases}
\min\{1,a\max(\Vert{}x\Vert{},10^{-3})/\max(\Vert{}z\Vert{},10^{-6})\},&a>0,\\
1,&a\le0,
\end{cases}\qquad r=cz,\qquad g=P r,
$$

where $P=I$ when gradient centralization is disabled or rank is at most one;
otherwise it subtracts the mean within each leading-axis slice. The raw gradient
fed into quantum decay is $r$, **after AGC but before centralization**
(`optimizer.py:96–110`; `step.py:339–353,510–525`). No in-kernel centralization
is used by canonical orchestration: every active launch passes
`grad_centralize=False` (`step.py:293,423`).

The shared chaos helper functions compute the detector and schedules. The
orchestrator reads disagreement against **old** momentum before any kernel
mutation (`step.py:357–363,541–558`), updates the normalized gradient-norm EMAs
(`step.py:365–374,560–567`), optionally synchronizes fast and slow state, then
computes trust $\rho$ and quantum coefficient $q_*$
(`step.py:376–408,569–618`). In the active chaos branch,

$$
\rho=1-\min\{s\gamma_{\rm eff}p_{\rm ext}h,
                 \min(\texttt{max\_cancel},0.99)\},\qquad
q_*=q_{\rm eff}h(1-s).
$$

Here $s$ is the soft chaos signal and $h$ the warmup gain. The branch
requires positive warmup gain and scheduled gamma, with per-tensor auto-gamma
as applicable. Quantum decay additionally requires $q_{\rm eff}>0$.
Otherwise $\rho=1$, $q_*=0$ and stored soft chaos is zero. The clamp has
only an upper limit; an upper trust bound of one additionally needs
$p_{\rm ext}\ge0$. Full detector equations are mapped in the main algorithm
specification (`psilogic/_chaos.py:114–207,210–282`).

Adam branch:

$$
m^+=\beta_1m+(1-\beta_1)g,\quad
v^+=\beta_2v+(1-\beta_2)g^2,\quad
\alpha_t=\eta\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},
$$

$$
x^+=(1-\eta\lambda)(1-\eta q_*\tanh|r|)x
       -\alpha_t\frac{\rho m^+}{\sqrt{v^+}+\epsilon}.
$$

Sources: `step.py:410–415,620–625`; per-tensor kernel
`kernels.py:259–285`; multi-tensor kernel `kernels.py:345–371`.
Weight decay is applied before quantum decay, giving the product of factors,
including its $\eta^2\lambda q_*\tanh|r|$ cross term. The trust factor
scales the adaptive step only, not either decay. Epsilon is added to the
**uncorrected** square-root variance before the bias-correction multiplier;
equivalently the epsilon in a fully bias-corrected denominator is
$\epsilon/\sqrt{1-\beta_2^t}$, rather than constant $\epsilon$.

Lion branch:

$$
u=\beta_1m+(1-\beta_1)g,\quad
x^+=(1-\eta\lambda)(1-\eta q_*\tanh|r|)x
       -\eta\rho\operatorname{sign}(u),\quad
m^+=\beta_2m+(1-\beta_2)g,
$$

with $\operatorname{sign}(0)=0$, variance left unchanged, and no bias
correction (`kernels.py:261–266,276–280,347–352,362–366`). Both momentum
expressions use old $m$. Launchers set variance/momentum-update flags to
`not lion` (`kernels.py:572–573,687–688`).

The scalar and foreach paths have these same exact-real update formulas for
finite, well-defined inputs and the supported layouts. Their finite-precision
rounding and gradient side effects are different.

## Finite-precision differences and conditional branches

1. **State precision.** All normal momentum, variance, fast, slow, and baseline
   norm buffers are initialized in `torch.float32`, even for fp64 parameters
   (`optimizer.py:113–126`). This makes a real-arithmetic theorem an idealized
   theorem; exact convergence of rounded parameters does not follow from it.

2. **Kernel arithmetic.** Fused kernels cast data and scalar hyperparameters to
   fp32 before arithmetic and cast the final parameter store back to its
   storage dtype (`kernels.py:239–252,331–343`). A float64 group is rejected by
   multi-tensor dispatch but can reach the per-tensor kernel, which still
   converts parameter values and gradients to fp32. Python scalar paths retain
   fp64 parameter/gradient operations where PyTorch promotion allows them.

3. **Rounding boundaries.** Scalar/foreach decay modifies parameter storage
   in separate in-place operations (`optimizer.py:499–500,517–519,885–888`)
   before the final parameter update (`optimizer.py:543–558,890–917`). Fused
   kernels keep these intermediate parameter values in fp32 registers and
   store once. This differs especially for fp16/bf16. Fused multiply/add
   lowering and reductions are not asserted to match PyTorch rounding.

4. **Quantum nonlinear function.** Python calls `torch.tanh(raw_grad.abs())`
   while Triton evaluates $2\sigma(2|r|)-1$ in fp32
   (`kernels.py:272–274,358–360`; `optimizer.py:519,835`). These are equal
   over real numbers. In finite precision, subtraction near one can erase
   very small nonzero values, and the sigmoid implementation/rounding need
   not match tanh. In Python, tanh is evaluated at gradient dtype.

5. **Gradient mutation.** Scalar and per-parameter fused AGC return a new
   clipped tensor and preserve `param.grad` (`optimizer.py:96–104`;
   `step.py:339–342`). Homogeneous fused and foreach AGC multiply original
   gradient tensors in place (`step.py:510–516`; `optimizer.py:702–709`).
   Therefore repeated `step()` calls without replacing/clearing gradients
   may have different trajectories even in exact arithmetic. Equivalence
   presumes the same fresh externally supplied gradients at every call.

6. **Gradient RMS rounding.** Scalar helper divides a norm tensor by a Python
   scalar (`_chaos.py:273`), keeping a low-precision gradient norm's dtype.
   Batched orchestration divides the norm vector by an fp32 tensor
   (`step.py:561–564`; `optimizer.py:778–786`), producing fp32. These need not
   provide bitwise equal normalized inputs, despite the helper comment at
   `_chaos.py:224–228`. A distinction also exists between Python-float and
   fp32 representations of $\sqrt n$.

7. **Chaos reduction.** Current homogeneous fused groups use a Triton sum of
   $g^2,gm,m^2$, accumulated in fp32, when chaos is active and $t>1$
   (`step.py:545–548`; `kernels.py:375–421`). Tensors are split into at most
   64 partial reductions, based on the largest block size
   (`step.py:208–209,230–264`). The non-atomic kernel uses a deterministic
   summation layout for a fixed launch, but no cross-backend bitwise
   equivalence follows. Gradient norms are rounded to fp16/bf16 before the
   cosine denominator; momentum norms and inner products remain fp32.
   At step one, the shared disagreement helper returns zero, not one half
   (`_chaos.py:146–147`).

8. **Auto-gamma arithmetic.** Scalar/per-tensor `auto_gamma` converts a state
   scalar to a Python float and computes host arithmetic
   (`_chaos.py:78–83`); batched mode computes an fp32 tensor expression
   (`_chaos.py:255–260`). Real formulas agree for finite nonnegative states,
   but intermediate rounding differs.

9. **Distributed collective grouping.** Scalar and homogeneous batched paths
   synchronize all active states together. Heterogeneous/asynchronous fused
   fallback invokes `maybe_sync([state])` for each parameter
   (`step.py:374,482–507`), so collective count/order differ from the scalar
   batched group call (`optimizer.py:640`). In every case only fast/slow are
   averaged, not baseline norm, momentum, or local disagreement
   (`optimizer.py:450–466`). Correct execution requires consistent active
   parameters and collective order across ranks; general distributed
   equivalence is not established by this source audit.

10. **Degenerate and nonfinite values.** Foreach skips Adam parameter
    `addcdiv_` entirely when learning rate is zero (`optimizer.py:898`),
    whereas scalar and fused still evaluate the division. This matters for
    zero/invalid denominators and nonfinite states. Parameter validation does
    not validate epsilon, `p_ext`, or all parameters for finiteness
    (`optimizer.py:69–93`). Finite nonnegative data and epsilon strictly
    positive must therefore be explicit mathematical hypotheses, rather than
    claimed consequences of constructor acceptance.

## Exposed legacy kernel entry points

`launch_leader_sums`, `launch_centralize_moment`, and `launch_decay_adam` remain
defined (`kernels.py:424–531`) but are not called by canonical `step.py`.
They are separate low-level APIs, not another automatic optimizer dispatch.

The leader-sum kernel uses atomic fp32 additions (`kernels.py:26–38`), so
floating-point summation order can vary. The legacy centralization launcher and
the optional centralization flag on `launch_fused_step` require both
`n_leaders > 1` and `elems_per_leader > 1`
(`kernels.py:469,592`). A rank-two tensor with one leading row is centralized
by the canonical Python helper but not by that low-level conditional if called
directly. Canonical orchestration avoids this discrepancy by centralizing in
Python unconditionally by rank and passing `False` to kernels.

The two-stage kernel stores centralized gradients to their original dtype
before its second kernel; single-launch fusion can retain a higher-precision
register value. Accordingly the docstring's blanket “numerically identical”
claim (`kernels.py:559–564`) is not an established finite-precision identity
for every direct low-level invocation, especially when in-kernel
centralization is enabled with low-precision storage. Only the real algebraic
correspondence is asserted here.

## Build-directory copies and canonical import

`build/lib/psilogic/optimizer.py`, `_chaos.py`, and `_cuda/__init__.py` are
byte-identical to the canonical counterparts at inspection. The build copy of
`_cuda/kernels.py` lacks 82 lines adding the multi-chaos reduction and its
launcher; the build copy of `_cuda/step.py` has the older per-tensor reduction
loop (canonical difference: 93 inserted, 18 removed lines). The active
parameter-update kernels otherwise match.

Packaging selects `psilogic*` from the repository root
(`pyproject.toml:88–90`), and a runtime import from this workspace resolves
`psilogic.__file__` to the root `psilogic/__init__.py`. Thus `build/lib` is a
stale build artifact, not the authoritative implementation. It would require
separate arithmetic accounting if explicitly placed earlier on `sys.path`.

## Verification scope

Initial runtime probe: `/opt/miniconda3/bin/python`, PyTorch `2.13.0+cu130`,
Triton importable, `torch.cuda.is_available() == False`; source imported from
the canonical workspace. Consequently actual CUDA kernels cannot be certified
by GPU execution in this environment. CPU probes or Triton interpreter probes,
if present in `verification/`, are labeled as such and do not prove GPU
equivalence. Existing parity tests use numerical tolerances and conditional
CUDA skips (`tests/test_numerical_parity.py:252–438`), not an exact equivalence
proof.

A direct `TRITON_INTERPRET=1` CPU launch was also attempted. The installed
interpreter failed before executing the kernel with a NumPy/Triton boolean
conversion error (`numpy data itemsize (32 bits) exceeds dtype
primitive_bitwidth (1 bits) for triton type int1`). This failed attempt is not
reported as a kernel test. The separate `counterexample_checks.py` records an
executed CPU example of the scalar/batched RMS rounding discrepancy.
