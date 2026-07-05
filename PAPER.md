<!--
  arXiv preprint source (Markdown notes).
  PDF source of truth: arxiv/paper.tex
  Build: python scripts/build_arxiv_pdf.py  ->  psilogic-arxiv.pdf
  Categories: cs.LG (primary), cs.AI (secondary).
-->

# PsiLogic: Chaos-Aware Active Cancellation for Adam with a Fair Cross-Domain Benchmark

**Ali Sultonov**  
Independent Researcher  
troxtergrif@gmail.com  
https://github.com/Troxter222/psilogic

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
bf16 AMP), PsiLogic achieves the best validation metric in **three of four arenas**: NLP
perplexity **7.79 ± 0.18** vs **8.17 ± 0.08** (AdamW, *p* = 0.049), ViT top-1 accuracy
**0.244 ± 0.006** vs **0.223 ± 0.002** (AdamW, *p* = 0.015), and ResNet top-1 accuracy
**0.222 ± 0.001** vs **0.172 ± 0.004** (Adam, *p* = 0.001). On diffusion, validation MSE
is statistically tied with Adam/AdamW (*p* = 0.49). ResNet accuracy vs AdamW is a numerical
tie without significance at three seeds (*p* = 0.44). Peak GPU memory is comparable across
optimizers; PsiLogic incurs **1.2–1.8×** wall-clock overhead on transformer-heavy arenas.

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
for `torch.optim.Adam` with optional task presets (`PsiLogicNLP`, `PsiLogicGPT`, `PsiLogicViT`).

**Contributions.**

1. **PsiLogic optimizer** — chaos-gated Active Cancellation on top of Adam, with unified
   decay, optional gradient centralization (GC), and adaptive gradient clipping (AGC).
2. **FairBench** — a bias-mitigated evaluation protocol: per-optimizer LR sweep, identical
   weights per seed, multi-arena tasks, and Welch *t*-tests.
3. **Reference H100 benchmark** — reproducible CSVs and learning-curve plots committed at
   `benchmark/results/full/`, showing competitive or superior quality on NLP, ViT, and ResNet
   with explicit reporting of non-significant and negative results.

We do **not** claim universal dominance over AdamW or Lion. We report limitations — including
step-time overhead and ties on diffusion and ResNet-vs-AdamW — explicitly.

---

## 2. Related Work

**Adam / AdamW.** Adam maintains bias-corrected first- and second-moment estimates for
per-parameter adaptive rates (Kingma & Ba, 2015). AdamW decouples weight decay from the
gradient step (Loshchilov & Hutter, 2019) and is the de facto standard for Transformers.

**Lion.** Lion (Chen et al., 2023) uses sign-based updates with coupled weight decay. It can
be memory-efficient but often requires careful LR tuning and underperforms on some
from-scratch language modeling tasks.

**Stability mechanisms.** Gradient centralization (Yong et al., 2020) and adaptive gradient
clipping (Brock et al., 2021) improve training stability. PsiLogic optionally integrates both.
Learning-rate warmup (Goyal et al., 2017) reduces early-step damage; PsiLogic provides a
related effect via chaos-gated damping.

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
convergence, `chaos_t → 0` and PsiLogic reduces toward AdamW-like behavior.

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

---

## 4. FairBench Evaluation

### 4.1 Protocol

All headline numbers in this paper come from one reference run on **NVIDIA H100 80GB HBM3**
(PyTorch 2.4.1+cu124, CUDA 12.4). Configuration is frozen in
`benchmark/results/full/config.json`.

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

### 4.4 Statistical Significance (Table 2)

Welch *t*-test: PsiLogic vs baseline. \* *p* < 0.05, \*\* *p* < 0.01, \*\*\* *p* < 0.001;
n.s. = not significant.

| Arena | Metric | vs Adam | vs AdamW | vs Lion |
|:------|:-------|:--------|:---------|:--------|
| NLP | Perplexity | \*\*\* | \* | \*\* |
| NLP | Val loss | \*\*\* | n.s. (*p*=0.054) | \*\*\* |
| ViT | Val acc | \*\*\* | \* | \*\* |
| ResNet | Val acc | \*\* | n.s. (*p*=0.44) | \* |
| Diffusion | Val MSE | n.s. | n.s. | \* |

**Interpretation.** PsiLogic is best on NLP (perplexity), ViT, and ResNet vs Adam. Against
AdamW, ViT and NLP perplexity are significant; ResNet and diffusion are not. Against Lion,
PsiLogic is significant on all arenas except diffusion MSE vs Adam/AdamW.

### 4.5 Compute Cost (Table 3)

| Arena | Peak VRAM (MB) A/W/L/P | Wall time (s) A/W/L/P | PsiLogic / AdamW time |
|:------|:----------------------|:---------------------|:---------------------:|
| NLP | 458 / 458 / 445 / 458 | 46.6 / 45.9 / 38.2 / 55.2 | 1.20× |
| ViT | 1229 / 1229 / 1208 / 1229 | 95.2 / 98.5 / 98.6 / 176.7 | **1.79×** |
| ResNet | 823 / 825 / 777 / 823 | 45.3 / 47.6 / 46.1 / 67.4 | 1.42× |
| Diffusion | 3780 / 3780 / 3768 / 3781 | 94.2 / 95.2 / 91.6 / 168.3 | **1.77×** |

A = Adam, W = AdamW, L = Lion, P = PsiLogic. VRAM differences are ≤ 3% except Lion on
ResNet/NLP (lower). Step-time overhead is the main practical cost today.

### 4.6 Figures

Rendered in the arXiv PDF from `arxiv/figures/` (see `arxiv/paper.tex`):

| Figure | File | Caption |
|:-------|:-----|:--------|
| Fig. 1 | `vit_val_val_acc.png` | ViT validation accuracy learning curves (mean ± std) |
| Fig. 2 | `nlp_val_perplexity.png` | NLP perplexity over training |
| Fig. 3 | `resnet_val_val_acc.png` | ResNet top-1 accuracy |
| Fig. 4 | `vit_train_step_time_s.png` | ViT per-step wall time (overhead illustration) |

---

## 5. Ablations and Component Analysis

Prior ablations on a synthetic MLP task (v0.3.x) showed that gradient centralization and
adaptive gradient clipping each independently improve stability when combined with the chaos
term. A *mirror ablation* demonstrated that dynamically mirroring PsiLogic's cancellation
magnitude as AdamW weight decay does not fully reproduce PsiLogic's per-parameter behavior,
indicating the chaos signal is not equivalent to a single global weight-decay schedule.

These ablations predate FairBench; component tests are maintained in `tests/`. Extended
FairBench ablations (γ, `max_cancel`, `chaos_warmup`) are planned.

---

## 6. Discussion

**Why chaos damping helps.** Large early gains on ViT (0.244 vs 0.079 Adam) suggest the chaos
term suppresses destructive early updates when gradient statistics are volatile. Under fair
LR tuning, NLP perplexity still favors PsiLogic over AdamW.

**Implicit warmup.** The cancellation term reduces effective step size during chaotic phases,
similar in spirit to LR warmup but driven by online gradient statistics.

**Reproducibility.** ResNet shows the lowest cross-seed standard deviation among optimizers
(±0.001 on accuracy), which may matter for production training pipelines.

**Limitations (stated explicitly).**

1. **Small seed count** — 3 seeds; some comparisons (ResNet vs AdamW, diffusion vs AdamW) are
   not statistically significant.
2. **Short training budget** — 2000 steps per arena; not ImageNet- or LLM-scale.
3. **Step-time overhead** — up to 1.79× vs AdamW on ViT in the Jun 2026 H100
   baseline; **v0.5+** adds optional Triton fusion (`psilogic[cuda]`, `use_fused_cuda=True`)
   targeting ≤1.25× without changing optimizer math. Re-run FairBench on GPU to refresh Fig. 4.
4. **Diffusion** — no quality win over Adam/AdamW at this budget.
5. **No convergence proof** — empirical stability only.
6. **Independent evaluation** — results have not yet been replicated by external groups.

---

## 7. Reproducibility Statement

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic && pip install -e ".[benchmark]" && pip install -r benchmark/requirements.txt
cd benchmark
python -m fairbench.download --data-root ./data
python -m fairbench --data-root ./data --output-dir results/full
```

Reference outputs: `benchmark/results/full/{aggregate,summary,significance}.csv`  
Software DOI: 10.5281/zenodo.18739857  
PyPI: `pip install psilogic`

---

## 8. Conclusion

PsiLogic augments Adam with a chaos-gated Active Cancellation term that is strong during
unstable training and vanishes at convergence. Under FairBench on NVIDIA H100, it achieves
the best validation metric in three of four cross-domain arenas, with honest reporting of
ties and overhead. Future work: reduce step-time cost, increase seed count and training
length, and seek independent replication at scale.

---

## References

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., et al. (2023). Symbolic
discovery of optimization algorithms. *NeurIPS*.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.

Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). Gradient centralization. *ECCV*.

Brock, A., et al. (2021). High-performance large-scale image recognition without
normalization. *ICML*.

Goyal, P., et al. (2017). Accurate, large minibatch SGD. *arXiv:1706.02677*.

Dosovitskiy, A., et al. (2021). An image is worth 16×16 words. *ICLR*.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning. *CVPR*.

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

---

*Ali Sultonov · Independent Researcher · arXiv preprint · https://github.com/Troxter222/psilogic*
