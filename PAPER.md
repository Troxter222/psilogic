<!--
  Author notes / Markdown working copy for the PsiLogic preprint.

  STATUS: on arXiv (arXiv:2607.16268) — keep notes in sync when bumping the PDF.

  Source of truth for the PDF: arxiv/paper.tex
  Build: python scripts/build_arxiv_pdf.py  ->  psilogic-arxiv.pdf
  Categories: cs.LG (primary), cs.AI (secondary).

  Process: edit paper.tex for camera-ready claims; mirror tables/abstract here
  so README / ROADMAP / notes stay aligned. Do not let PAPER.md and paper.tex diverge.
-->

# PsiLogic: Chaos-Aware Active Cancellation for Adam with a Fair Cross-Domain Benchmark

**Ali Sultonov**  
Independent Researcher  
troxtergrif@gmail.com  
https://github.com/Troxter222/psilogic

**Status:** arXiv preprint [arXiv:2607.16268](https://arxiv.org/abs/2607.16268) · Software DOI [10.5281/zenodo.18739857](https://doi.org/10.5281/zenodo.18739857)  
**PDF SoT:** [`arxiv/paper.tex`](arxiv/paper.tex) · **Notes last synced:** 2026-09-06 (v0.6 package defaults + fusion caveat)

### Pre-update checklist (before next arXiv bump)

- [ ] Tables 1–3 match `benchmark/results/full/aggregate.csv` / `significance.csv`
- [ ] Figures under `arxiv/figures/` exist and match Fig. 1–4 captions
- [ ] BibTeX / citation block matches README
- [ ] Wall-time table notes pre-fusion vs fused re-run commit SHA
- [ ] `PAPER.md` abstract ↔ `arxiv/paper.tex` abstract diffed

---

## Abstract

Adaptive optimizers such as Adam and AdamW apply the same update rule regardless of
whether training is in a chaotic early phase or near convergence. We introduce
**PsiLogic** (ΨLogic), an optimizer that augments Adam with a *dynamic Active Cancellation
Term* gated by a dual exponential moving average (EMA) of scale-normalized gradient norms.
The resulting *chaos detector* strengthens damping when gradient statistics are unstable
and fades to zero as training stabilizes, providing an implicit warmup without a hand-tuned
schedule.

We evaluate PsiLogic against Adam, AdamW, and Lion using **FairBench** — a reproducible
benchmark protocol with per-optimizer learning-rate sweeps, identical initialization per seed,
and Welch *t*-tests. On an NVIDIA H100 80GB reference run (4 arenas, 3 seeds, 2000 steps,
bf16 AMP), PsiLogic records the best validation metric on **NLP perplexity** and **ViT
accuracy**, **beats Adam and numerically ties AdamW on ResNet**, and **ties Adam/AdamW on
diffusion**. Concretely: NLP perplexity **7.79 ± 0.18** vs **8.17 ± 0.08** (AdamW,
*p* = 0.049); ViT top-1 **0.244 ± 0.006** vs **0.223 ± 0.002** (AdamW, *p* = 0.015);
ResNet top-1 **0.222 ± 0.001** vs Adam **0.172 ± 0.004** (*p* = 0.001) and AdamW
**0.219 ± 0.005** (*p* = 0.44, n.s.); diffusion MSE **0.02009 ± 0.00045** vs
**0.01987 ± 0.00006** (*p* = 0.49, n.s.). Peak GPU memory is comparable across optimizers;
PsiLogic incurs **1.2–1.8×** wall-clock overhead on the Jun 2026 H100 run (**pre-Triton
fusion**). Package v0.5+ ships an optional fused CUDA backend that does not change
optimizer math; a fused FairBench re-run is pending.

We release an open-source PyTorch implementation, the full FairBench harness, and all raw
CSV outputs to support independent verification.

**Keywords:** optimization, Adam, adaptive learning rate, deep learning, reproducibility

---

## 1. Introduction

The choice of optimizer affects convergence speed, generalization, and training stability in
deep learning. Adam (Kingma & Ba, 2015) and AdamW (Loshchilov & Hutter, 2019) dominate
practice, yet their corrective signal does not adapt to *how confused the model currently is*.
At initialization, gradients are large and noisy; near convergence, they are small and stable.
Standard Adam treats both regimes with structurally similar updates.

We propose **PsiLogic**, which adds a chaos-conditioned damping term to the Adam update.
The term is strongest when a dual EMA of normalized gradient norms signals instability, and
vanishes automatically as training settles. PsiLogic is designed as a drop-in replacement
for `torch.optim.Adam` with optional task presets (`PsiLogicNLP`, `PsiLogicGPT`,
`PsiLogicViT`, `PsiLogicWhisper`).

**Contributions.**

1. **PsiLogic** — chaos-gated Active Cancellation on Adam, with unified decay and optional GC/AGC.
2. **FairBench** — per-optimizer LR sweep, matched inits, multi-arena tasks, Welch *t*-tests.
3. **Reference H100 run** — public CSVs under `benchmark/results/full/`, with explicit ties,
   non-significant results, and wall-time overhead.

We do **not** claim universal dominance over AdamW or Lion.

---

## 2. Related Work

**Adam / AdamW.** Adam maintains bias-corrected first- and second-moment estimates for
per-parameter adaptive rates (Kingma & Ba, 2015). AdamW decouples weight decay from the
gradient step (Loshchilov & Hutter, 2019) and is the de facto standard for Transformers.

**Lion.** Lion (Chen et al., 2023) uses sign-based updates with coupled weight decay. It can
be memory-efficient but often requires careful LR tuning and underperforms on some
from-scratch language modeling tasks.

**Recent adaptive / schedule-free methods.** Sophia (Liu et al., 2023) uses a lightweight
Hessian diagonal for second-order-ish steps. Schedule-Free AdamW (Defazio et al., 2024)
removes explicit LR schedules via averaging. Prodigy (Mishchenko & Defazio, 2023) adapts
distance-to-optimum estimates online. Muon (Jordan et al., 2024) orthogonalizes 2D updates.
PsiLogic is complementary: it keeps Adam moments and adds a *chaos-gated* multiplicative
damping term rather than replacing the preconditioner or schedule.

**Stability mechanisms.** Gradient centralization (Yong et al., 2020) and adaptive gradient
clipping (Brock et al., 2021) improve training stability. PsiLogic optionally integrates both
(enabled in task presets; **off** on the bare v0.6+ constructor — see §3.6). Learning-rate
warmup (Goyal et al., 2017) reduces early-step damage; PsiLogic provides a related effect via
chaos-gated damping.

**Optimizer evaluation.** Fair comparison requires matched tuning budgets. FairBench gives
each optimizer its own LR search rather than a single shared LR, reducing tuning bias that
has historically confounded optimizer comparisons.

---

## 3. Method

### 3.1 Update Rule

PsiLogic extends Adam with an Active Cancellation Term:

```
θ_{t+1} = θ_t
         − η · m̂_t / (√v̂_t + ε)           [Adam step]
         − η · γ · P · chaos_t · θ_t         [active cancellation, when gated on]
```

Adam moments follow the standard definitions with bias correction. Weight decay is applied
through a unified per-step shrinkage coefficient (Section 3.3).

### 3.2 Chaos Detector

Let `gn_t = ‖∇_t‖₂ / √(numel)` be the scale-normalized gradient norm. We maintain:

```
fast_t = 0.90 · fast_{t-1} + 0.10 · gn_t     [τ ≈ 10 steps]
slow_t = 0.99 · slow_{t-1} + 0.01 · gn_t     [τ ≈ 100 steps]

ratio_t = fast_t / (slow_t + ε)
chaos_t = tanh(slow_t) · (1 + 0.5 · tanh(relu(ratio_t − 1)))
```

In adaptive mode (default), cancellation activates when `fast_t > τ_scale · slow_t`
(`τ_scale = 2.0`), detecting relative spikes in gradient chaos. As `slow_t → 0` at
convergence, `chaos_t → 0` and PsiLogic reduces toward Adam-like behavior without the
extra cancellation.

### 3.3 Unified Decay

To avoid compounding shrinkage from weight decay, cancellation, and auxiliary penalties,
all multiplicative decay is collapsed into a single coefficient per step. A hard clamp
`c_coeff ≤ max_cancel` (default 0.05) limits per-step weight reduction during high-loss
initialization. Optional cosine decay on γ is supported via `gamma_T_max`.

### 3.4 Algorithm

```
Algorithm 1: PsiLogic (simplified)

for t = 1 … T:
    g ← ∇L(θ); optionally apply AGC and gradient centralization
    update Adam moments m, v
    update fast_t, slow_t from ‖g‖₂
    if chaos gate active:
        θ ← θ · (1 − η·λ − min(chaos_t·η·γ·P, max_cancel))
    else:
        θ ← θ · (1 − η·λ)
    θ ← θ − η · m̂ / (√v̂ + ε)
```

Full implementation: https://github.com/Troxter222/psilogic (`psilogic/optimizer.py`).

### 3.5 Comparison with Baselines

| Property | Adam | AdamW | Lion | **PsiLogic** |
|:---------|:----:|:-----:|:----:|:------------:|
| Per-parameter adaptive rates | ✓ | ✓ | ✗ | ✓ |
| Unified decay (with chaos) | ✗ | ✗ | ✗ | ✓ |
| Chaos-aware damping | ✗ | ✗ | ✗ | **✓** |
| Implicit early-phase damping | ✗ | ✗ | ✗ | **✓** |
| Batched `foreach` CUDA kernels | partial | ✓ | ✗ | ✓ |
| Optional fused Triton step (v0.5+) | ✗ | ✗ | ✗ | ✓ |

### 3.6 Implementation note (package defaults & backends)

**Math vs backends.** The foreach and Triton fused CUDA paths implement the same update as
the scalar reference. Fusion does not change FairBench *quality* metrics; it only affects
step time. Enable with `pip install "psilogic[cuda]"` (`use_fused_cuda=True` when available).

**v0.6 safer defaults.** The bare constructor `PsiLogic(params, lr=...)` uses
`agc_clip=0.0` and `grad_centralize=False`. FairBench NLP follow-ups showed that enabling
AGC + GC on TinyStories GPT-scratch hurt vs AdamW. Task helpers (`PsiLogicNLP`,
`PsiLogicViT`, …) and presets may still enable mild AGC/GC. Headline FairBench quality
numbers remain those of the Jun 2026 reference configs in `benchmark/results/full/`.

---

## 4. FairBench Evaluation

### 4.1 Protocol

All headline numbers in this paper come from one reference run on **NVIDIA H100 80GB HBM3**
(PyTorch 2.4.1+cu124, CUDA 12.4). Configuration is frozen in
`benchmark/results/full/config.json`. Cite the software DOI and prefer pinning the git
commit that produced the CSVs when reproducing.

| Stage | Description |
|:------|:------------|
| **Stage 1 — LR sweep** | 7 log-spaced LRs from 10⁻⁵ to 10⁻²; 500 steps each; best val metric wins |
| **Stage 2 — Evaluation** | Selected LR; 2000 steps; seeds {0, 1, 2}; identical init per seed |
| **Shared** | batch=64, bf16 AMP, grad_clip=1.0, cosine LR, 100-step warmup |
| **Statistics** | Mean ± std; Welch *t*-test (PsiLogic vs each baseline) |

PsiLogic uses fixed per-arena presets; **only LR is tuned**, as for all baselines.

### 4.2 Arenas

| Arena | Model | Dataset | Metric |
|:------|:------|:--------|:-------|
| NLP | Small GPT | TinyStories | Perplexity ↓, val loss ↓ |
| ViT | ViT-Tiny patch16 224 | CIFAR-100 @ 224² | Top-1 acc ↑ |
| ResNet | ResNet-18 | Tiny ImageNet 200 | Top-1 acc ↑ |
| Diffusion | DDPM + UNet | CelebA @ 64² | Val MSE ↓ |

### 4.3 Main Results (Table 1)

Canonical source: `benchmark/results/full/aggregate.csv`.

| Arena | Metric | Adam | AdamW | Lion | **PsiLogic** |
|:------|:-------|:----:|:-----:|:----:|:------------:|
| NLP | Perplexity ↓ | 13.66±0.22 | 8.17±0.08 | 21.04±1.41 | **7.79±0.18** |
| NLP | Val loss ↓ | 2.614±0.016 | 2.101±0.010 | 3.045±0.068 | **2.053±0.023** |
| ViT | Val acc ↑ | 0.079±0.003 | 0.223±0.002 | 0.213±0.002 | **0.244±0.006** |
| ResNet | Val acc ↑ | 0.172±0.004 | 0.219±0.005 | 0.205±0.007 | **0.222±0.001** |
| Diffusion | Val MSE ↓ | **0.01987±0.00006** | **0.01987±0.00006** | 0.02175±0.00025 | 0.02009±0.00045 |

**Selected LRs:** NLP — all `3.16×10⁻⁴`; ViT — Adam `3.16×10⁻⁵`, AdamW/PsiLogic `3.16×10⁻⁴`,
Lion `10⁻⁴`; ResNet — Adam/Lion `10⁻⁴`, AdamW/PsiLogic `3.16×10⁻⁴`; Diffusion —
Adam/AdamW/PsiLogic `10⁻³`, Lion `10⁻⁴`.

**Scorecard.** Wins: NLP (PPL), ViT. Beat Adam / tie AdamW: ResNet. Tie: diffusion.

### 4.4 Statistical Significance (Table 2)

Welch *t*-test: PsiLogic vs baseline. \* *p* < 0.05, \*\* *p* < 0.01, \*\*\* *p* < 0.001;
n.s. = not significant. Source: `benchmark/results/full/significance.csv`.

| Arena | Metric | vs Adam | vs AdamW | vs Lion |
|:------|:-------|:--------|:---------|:--------|
| NLP | Perplexity | \*\*\* | \* | \*\* |
| NLP | Val loss | \*\*\* | n.s. (*p*=0.054) | \*\*\* |
| ViT | Val acc | \*\*\* | \* | \*\* |
| ResNet | Val acc | \*\* | n.s. (*p*=0.44) | \* |
| Diffusion | Val MSE | n.s. | n.s. | \* |

**Interpretation.** Against AdamW, ViT and NLP perplexity are significant; ResNet and
diffusion are not. Against Lion, PsiLogic is significant on quality arenas above except
where diffusion already loses to Adam/AdamW.

### 4.5 Compute Cost (Table 3)

Canonical peak VRAM / wall time from the same Jun 2026 H100 reference CSVs
(**pre-fusion**). A/W/L/P = Adam / AdamW / Lion / PsiLogic.

| Arena | Peak VRAM (MB) A/W/L/P | Wall time (s) A/W/L/P | PsiLogic / AdamW |
|:------|:----------------------|:---------------------|:----------------:|
| NLP | 458 / 458 / 445 / 458 | 46.6 / 45.9 / 38.2 / 55.2 | 1.20× |
| ViT | 1229 / 1229 / 1208 / 1229 | 95.2 / 98.5 / 98.6 / 176.7 | **1.79×** |
| ResNet | 823 / 825 / 777 / 823 | 45.3 / 47.6 / 46.1 / 67.4 | 1.42× |
| Diffusion | 3780 / 3780 / 3768 / 3781 | 94.2 / 95.2 / 91.6 / 168.3 | **1.77×** |

VRAM differences are ≤ 3% except Lion on some arenas (lower). Step-time overhead is the
main practical cost on this baseline. Refresh Fig. 4 / this table after the fused H100 re-run.

### 4.6 Figures

Rendered in the arXiv PDF from `arxiv/figures/` (see `arxiv/paper.tex`):

| Figure | File | Caption |
|:-------|:-----|:--------|
| Fig. 1 | `vit_val_val_acc.png` | ViT validation accuracy learning curves (mean ± std) |
| Fig. 2 | `nlp_val_perplexity.png` | NLP perplexity over training |
| Fig. 3 | `resnet_val_val_acc.png` | ResNet top-1 accuracy |
| Fig. 4 | `vit_train_step_time_s.png` | ViT per-step wall time (pre-fusion overhead) |

**Docs debt (deferred, not blocking claims):** additional side-by-side learning-curve
composite (ΨLogic vs Lion vs AdamW) and GPT `chaos_warmup` ablation figure — tracked in
[ROADMAP.md](ROADMAP.md); mark as future work until produced.

---

## 5. Ablations and Component Analysis

Prior ablations on a synthetic MLP task (v0.3.x) showed that gradient centralization and
adaptive gradient clipping each independently improve stability when combined with the chaos
term. A *mirror ablation* demonstrated that dynamically mirroring PsiLogic's cancellation
magnitude as AdamW weight decay does not fully reproduce PsiLogic's per-parameter behavior,
indicating the chaos signal is not equivalent to a single global weight-decay schedule.

These ablations **predate FairBench** and are **not** used for headline claims. Component
unit tests live in `tests/`. Extended FairBench-scale ablations (γ, `max_cancel`,
`chaos_warmup`, GC/AGC on/off under the FairBench protocol) are **deferred to a future
revision** (see ROADMAP Phase 3B) rather than implied as complete in this preprint.

---

## 6. Discussion

**Why chaos damping helps.** Large early gains on ViT (0.244 vs 0.079 Adam) suggest the chaos
term suppresses destructive early updates when gradient statistics are volatile. Under fair
LR tuning, NLP perplexity still favors PsiLogic over AdamW.

**Implicit warmup.** The cancellation term reduces effective step size during chaotic phases,
similar in spirit to LR warmup but driven by online gradient statistics.

**Seed stability.** ResNet shows the lowest cross-seed standard deviation among optimizers
(±0.001 on accuracy), which may matter for production training pipelines.

### Limitations

1. **Small seed count** — 3 seeds; ResNet-vs-AdamW and diffusion-vs-AdamW are n.s.
2. **Short training budget** — 2000 steps per arena; not ImageNet- or LLM-scale.
3. **Step-time overhead** — up to 1.79× vs AdamW on ViT (Jun 2026 H100, **pre-fusion**).
   v0.5+ Triton fusion targets ≤1.25× on Ampere+ without changing math; fused FairBench
   re-run pending.
4. **Diffusion** — no quality win over Adam/AdamW at this budget.
5. **No convergence proof** — empirical stability only.
6. **Independent evaluation** — not yet replicated by external groups.
7. **Package default drift** — v0.6 bare defaults differ from some preset/FairBench configs;
   always record the exact preset when comparing to this paper.

### Threats to validity

- Single accelerator vendor/SKU (NVIDIA H100) for the reference run.
- Fixed 2000-step horizon may favor methods that start fast over long-horizon winners.
- Per-optimizer LR sweep still uses a shared secondary recipe (warmup, clip, cosine).
- Arena models are small relative to production LLMs / ImageNet-1k training.

### Compute / environment note

The reference FairBench suite is a multi-arena H100 job (LR sweeps + 3 seeds × 4 optimizers
× 4 arenas). Exact GPU-hours depend on queueing and retries; treat wall times in Table 3 as
the public cost signal per evaluation stage, and prefer re-reporting total GPU-hours when
publishing a fused re-run.

---

## 7. Reproducibility Statement

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic && pip install -e ".[benchmark]" && pip install -r benchmark/requirements.txt

# Optional long run in tmux (resume-friendly)
./run_fairbench.sh

# Or step-by-step:
cd benchmark
python -m fairbench.download --data-root ./data
python -m fairbench --data-root ./data --output-dir results/full

# Smoke test (CPU-friendly)
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0
```

Reference outputs: `benchmark/results/full/{aggregate,summary,significance,config}.csv|json`  
Software DOI: [10.5281/zenodo.18739857](https://doi.org/10.5281/zenodo.18739857)  
PyPI: `pip install psilogic`  
When citing numbers, pin the git commit that matches the committed CSVs.

---

## 8. Conclusion

PsiLogic augments Adam with a chaos-gated Active Cancellation term that is strong during
unstable training and vanishes at convergence. Under FairBench on NVIDIA H100, it wins NLP
perplexity and ViT accuracy, ties or beats baselines on ResNet depending on the comparator,
ties on diffusion, and reports wall-time overhead honestly. Future work: fused overhead
validation, larger seed counts and training length, FairBench-scale ablations, and
independent replication at scale.

---

## References

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., et al. (2023). Symbolic
discovery of optimization algorithms. *NeurIPS*.

Defazio, A., et al. (2024). The road less scheduled. *arXiv* (Schedule-Free optimizers).

Jordan, K., et al. (2024). Muon: An optimizer for hidden representations. *arXiv*.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.

Liu, H., et al. (2023). Sophia: A scalable stochastic second-order optimizer for language
model pre-training. *arXiv*.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.

Mishchenko, K., & Defazio, A. (2023). Prodigy: An expeditiously adaptive parameter-free
learner. *arXiv*.

Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). Gradient centralization. *ECCV*.

Brock, A., et al. (2021). High-performance large-scale image recognition without
normalization. *ICML*.

Goyal, P., et al. (2017). Accurate, large minibatch SGD. *arXiv:1706.02677*.

Dosovitskiy, A., et al. (2021). An image is worth 16×16 words. *ICLR*.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning. *CVPR*.

Sultonov, A. (2026). PsiLogic software & FairBench artifacts.
[doi:10.5281/zenodo.18739857](https://doi.org/10.5281/zenodo.18739857).

---

## Appendix A. Per-Seed ViT Accuracy

| Seed | Adam | AdamW | Lion | **PsiLogic** |
|-----:|:----:|:-----:|:----:|:------------:|
| 0 | 0.078 | 0.226 | 0.214 | **0.238** |
| 1 | 0.083 | 0.222 | 0.211 | **0.247** |
| 2 | 0.076 | 0.221 | 0.213 | **0.249** |

Full per-seed tables: `benchmark/results/full/summary.csv`.

## Appendix B. Archived Experiments

Pre-FairBench results (CIFAR-10 A40, BERT, AG News, etc.) are archived in `OLD_RESULTS.md`
and are **not** used for claims in this preprint.

## Appendix C. LR Sweep Grid

Stage-1 candidates (shared log grid):  
`{1e-5, 3.16e-5, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2}`  
(500 steps each; best validation metric selects Stage-2 LR). Per-arena winners are listed
under Table 1. Full sweep logs: `benchmark/results/full/` and `benchmark/logs.txt`.

---

*Ali Sultonov · Independent Researcher · arXiv:2607.16268 · https://github.com/Troxter222/psilogic*
