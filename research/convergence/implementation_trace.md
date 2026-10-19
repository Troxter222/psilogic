# Source-to-equation traceability

## Source identity and scope

This trace describes the original investigation snapshot. Line numbers below
precede publication edits to comments and docstrings; use the named functions
to locate the corresponding code in the current tree.

The source of truth is the local `psilogic/` directory at revision
`b4917163d63d8a5cb2960e29af4b1bccdbf5798d`. The pre-research working tree had
modified `.gitignore`, `PAPER.md`, `README.md`, `arxiv/paper.tex`, and
`scripts/build_arxiv_pdf.py`; the optimizer package itself was not modified.
Untracked files included generated paper figures/tables, benchmark result
summaries, output PDF artifacts, `scripts/paper_results.py`,
`scripts/render_paper.py`, and `uv.lock`. None supplied a second imported
optimizer implementation. Dataset/checkpoint contents were not read.
`source_snapshot.json` records provenance; research artifacts created later are
not part of the original working-tree status.

An ignored `build/lib/psilogic/` copy was found by an explicit source-file scan.
All its Python files are byte-identical to the root package except
`_cuda/step.py` and `_cuda/kernels.py`. Those alternate build artifacts need
their own fused audit and are not the root import entry point. An ignored
`tmp/paper-before-revision/build_arxiv_pdf.py` is an old paper rendering script,
not an optimizer. No C/C++/CUDA extension source outside the Python Triton
module was found in the relevant project-source scan. Compiled bytecode and
large generated binaries were not treated as authoritative source.

The packaging configuration finds only `psilogic*` (`pyproject.toml:88–91`).
`psilogic/__init__.py:11–26` exports the core class, convenience subclasses,
builders and presets. HF/Lightning integration constructors ultimately call
the same class; they do not define alternative update equations.

## Canonical execution chain

| Executable location | Mathematical responsibility |
|---|---|
| `psilogic/optimizer.py:283–339` | Constructor defaults and global backend/DDP switches |
| `psilogic/optimizer.py:919–951` | Optional closure, then branch selection per group; compilation uses scalar |
| `psilogic/optimizer.py:939–951` | CUDA present: fused preferred, then foreach; ordinary CPU uses scalar |
| `psilogic/optimizer.py:963–974` | Fused availability check and dispatch; unavailable fused directly calls foreach |
| `psilogic/optimizer.py:591–608` | Skip absent gradients, preprocess, lazy initialization and per-block counter |
| `psilogic/optimizer.py:96–110` | Whole-tensor AGC and slice-mean centralization |
| `psilogic/optimizer.py:113–126` | Fresh zero float32 moments and scalar detector state |
| `psilogic/optimizer.py:613–621` | Check schedule, read old momentum for disagreement, update Adam moments |
| `psilogic/optimizer.py:623–640` | Update norm EMAs on every active observation, then optional DDP average |
| `psilogic/optimizer.py:643–678` | Schedule/auto-gamma, decays, trust and final Adam/Lion step |
| `psilogic/optimizer.py:499–527` | Weight decay first; quantum decay only inside active chaos; trust |
| `psilogic/optimizer.py:543–558` | Lion sign direction and post-step momentum, or bias-corrected Adam update |
| `psilogic/_chaos.py:35–61` | Negative warmup configuration, known/unknown horizon policy and ramp |
| `psilogic/_chaos.py:114–124` | Clamped cosine gamma and quantum-decay schedule |
| `psilogic/_chaos.py:263–282` | New norm average in normalization denominator; first-step fast/slow equal one |
| `psilogic/_chaos.py:127–151` | Stabilized cosine with old momentum, clipped to [-1,1], first-step exception |
| `psilogic/_chaos.py:154–186` | Adaptive-ratio or absolute-slow sigmoid, multiplied by tanh(slow) |
| `psilogic/_chaos.py:64–83` | Optional auto-gamma using current slow state |
| `psilogic/_chaos.py:189–207` | Upper-only damping clamp and hard maximum .99 |
| `psilogic/optimizer.py:680–917` | Foreach batching of preprocessing, moments, detector, decays and update |
| `psilogic/optimizer.py:427–466` | DDP arithmetic mean of fast and slow only |
| `psilogic/optimizer.py:369–421` | Schema metadata, migration and missing-state fill |
| `psilogic/debug.py:51–59` | Read-only diagnostic spike threshold; not part of optimization |

The exact recurrence, initialization, dimensions and filtration are in
`algorithm_specification.md`. All equation references there point to this source
snapshot. No optimizer source file was edited to make it fit the analysis.

## Defaults and accepted configuration

The plain constructor has these defaults (`optimizer.py:283–306`):

| Setting | Default |
|---|---|
| `lr`, `betas`, `weight_decay` | `1e-3`, `(0.9,0.999)`, `1e-4` |
| `gamma`, `p_ext`, `quantum_decay` | `0.05`, `1.0`, `0.0` |
| `eps`, `grad_centralize`, `agc_clip` | `1e-8`, `False`, `0.0` |
| `chaos_tau`, `chaos_warmup` | `0.5`, `-1` |
| `adaptive_tau`, `tau_scale`, `max_cancel` | `True`, `2.0`, `0.05` |
| `gamma_T_max`, `gamma_auto`, `lion_mode` | `0`, `False`, `False` |
| `use_foreach`, `use_fused_cuda` | `True`, `True` |
| `sync_chaos_ddp`, `profile_step_time` | `False`, `False` |

With unknown horizon the actual automatic warmup is 200 active observations,
followed by a 50-observation linear ramp, not the 500 mentioned without the
unknown-horizon qualification in one class docstring. The warmup uses local
block time; blocks with absent gradients can activate at different global calls.

`_validate_hyperparameters` (`optimizer.py:69–93`) checks nonnegative `lr`,
`weight_decay`, `gamma`, `quantum_decay`, `agc_clip`; betas in `[0,1)`; and
`max_cancel` in `(0,1]`. It does not validate `eps`, `p_ext`, the detector scales,
integer schedule arguments, or finiteness of every numeric argument. `NaN < 0`
is false, so several nominal nonnegativity checks also admit NaN. These checks
are applied to constructor arguments before the inherited optimizer merges
parameter-group dictionaries; arbitrary group overrides are not passed through
this validator. Mathematical admissibility therefore cannot be inferred merely
from successful construction.

`sync_chaos_ddp` and `profile_step_time` are optimizer-wide flags, not defaults
placed into every group. The fused switch additionally requires the global
constructor switch to be true, so a group cannot turn fused mode on if it was
globally disabled. `lion_mode` and the other stored defaults can differ by group.

## Presets, architecture inference and group precedence

`presets.py:45–72` supplies common `(0.9,0.999)` betas, `p_ext=1`, `eps=1e-8`,
`chaos_tau=.40`, automatic warmup, adaptive tau, requested horizon and foreach.
The mode-specific values are:

| Preset (source) | gamma | quantum decay | tau scale | max cancel | AGC | centralize | weight decay |
|---|---:|---:|---:|---:|---:|---|---:|
| NLP (`presets.py:75–89`) | .03 | .0002 | 2 | .05 | .01 | yes | .0001 |
| Vision (`presets.py:92–107`) | .04 | 0 | 2.5 | .04 | .02 | yes | .0001 |
| GPT (`presets.py:110–129`) | .02 | 0 | 2 | .03 | 0 | no | .1 |
| Whisper (`presets.py:132–147`) | .02 | .0001 | 2 | .04 | .01 | yes | .01 |
| GLUE (`presets.py:150–161`) | .03 | .0002 | 2 | .05 | .01 | yes | .01 |

Convenience subclasses use `setdefault`, so explicit constructor keywords
override preset values. They omit `gamma_T_max` and `use_foreach` from the
`apply_preset` injection (`presets.py:26–37`) and pass the horizon separately
(`convenience.py:135–222`). A supplied module is converted into task-specific
parameter groups; a supplied parameter iterable is used directly.

Architecture inference (`convenience.py:53–77`) checks parameter name markers
and module types in priority order: ViT markers plus attention; embedding plus
attention (GPT if GPT name markers, NLP otherwise); convolution (vision);
embedding (NLP); generic. `PsiLogic.auto` chooses the preset and builder
(`convenience.py:101–121`). Default learning rates are `.001` except GPT `.0003`.
Generic auto adds only the requested horizon. This is a heuristic about module
names, not a mathematical inference about the loss.

Parameter groups explicitly supply values that override constructor defaults:

* NLP (`param_groups.py:83–159`): embeddings gamma `.01`, attention `.03`,
  other `.03`; substring matching removes weight decay from selected bias/norm
  parameters. Embedding match has priority and is not automatically no-decay.
* ViT (`param_groups.py:162–269`): token/patch gamma `.005`, attention `.02`,
  MLP/rest `.03`; tokens and all tensors of dimension at most one have zero
  weight decay. With `lion_blocks=True`, attention/MLP/rest use Lion while
  tokens/norm-bias/patch groups explicitly use Adam.
* GPT (`param_groups.py:272–357`): dimension-at-most-one tensors have zero
  decay; embedding gamma `.005` and quantum decay explicitly zero; head gamma
  `.01`; blocks/default `.02`. Named parameter deduplication handles tied weights.

Only trainable named parameters enter these builders; empty groups are removed
and coverage is checked. Shared keyword entries for group-owned settings are
stripped. In particular, a constructor `gamma` override does not replace gamma
already set by a builder. In `PsiLogic.auto`, a constructor weight-decay override
also does not replace a builder's already-set decay. HF construction instead
passes the chosen decay into its builder (`integrations/hf.py:87–96`). Its auto
mapping routes a generic model to NLP and CNN vision to ViT, unlike plain
`PsiLogic.auto`. HF's NLP default LR is `2e-5`, not `.001`.
Lightning delegates to HF construction (`integrations/lightning.py:45–60`).

## Foreach, DDP and external training-loop behavior

Foreach filters active blocks, applies AGC **in-place to gradient buffers**,
then centralizes (`optimizer.py:698–714`). It reads disagreement before
updating moments (`739–757`). Uniform counters and homogeneous device/dtype
select the vectorized EMA branch (`759–839`); otherwise the per-block branch
uses each counter separately (`840–883`). Both use the same real recurrence.
Parameters receive decays before the additive step (`885–917`). Foreach skips
Adam's final update if `lr<=0`, unlike the scalar path; the admissible positive-LR
finite domain is the comparison intended here.

DDP averages only updated fast/slow state (`optimizer.py:427–466`), not the norm
average, gradients, moments, or disagreement. If all ranks start equal and
receive equal gradients from an external DDP wrapper, deterministic real
arithmetic preserves equality. Without those conditions averaging two scalar
states alone does not force equal damping or parameters. Collective ordering
also presupposes matching active tensor lists across ranks.

FairBench imports the package (`benchmark/fairbench/optimizers.py:148–153`) and
passes its knobs to `PsiLogic` (`237–241`); it does not define a hidden alternate
PsiLogic. Its loop performs gradient zeroing, AMP forward/backward, optional
global norm clipping after unscale, scaler step/update and external scheduler
step (`benchmark/fairbench/runner.py:227–238`). Accumulation, skipped scaler
steps, optimizer closure side effects and scheduler policy belong to the oracle
and global-time model, not an unmentioned internal recurrence.

## Important distinctions from comments or historical descriptions

1. Cancellation dampens the Adam/Lion additive update; it is not a separate
   chaos-dependent multiple of the parameter. Quantum decay is a distinct,
   optional, coordinatewise multiplicative factor.
2. The gate is continuous. Diagnostic hard spike labels have no update effect.
3. Two detector symptoms are not statistically independent by construction.
4. Normalized slow state need not vanish whenever raw gradients become small;
   the recent-gradient scale is in its denominator. No convergence certificate
   follows from `gamma_auto` being named "convergence-aware".
5. Epsilon placement differs from constant-epsilon bias-corrected AdamW.
6. Positive trust does not make an update a descent direction and does not
   restore stochastic unbiasedness.
7. Nonzero quantum decay is gated by positive gamma and completed warmup.
8. The source accepts unsupported pathological settings and group overrides;
   all convergence statements must impose their own explicit valid domain.
9. Float32 state allocation on fresh creation does not establish universal
   finite-precision stability or enforce float32 state after every checkpoint
   load (`optimizer.py:397–421`).

## Verification boundary

Independent symbolic reconstruction agrees on the scalar recurrence, the
preprocessing order, the new norm-average denominator, old-momentum disagreement,
epsilon placement, warmup/schedule counters and nested quantum branch. Small
deterministic recurrence comparisons and branch probes are recorded in
`verification/`; these test correspondence, not theorems of stochastic
convergence. Fused source inspection and any lack of real CUDA execution must
remain separately disclosed. Fresh initialization, valid dense real inputs and
the stated exact-arithmetic model are assumptions, not runtime enforcement.
