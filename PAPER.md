<!--
  Working copy. PDF source: arxiv/paper.tex
  Build: python scripts/build_arxiv_pdf.py
  cs.LG, cs.AI.
-->

# PsiLogic: Chaos-Aware Active Cancellation for Adam with a Fair Cross-Domain Benchmark

**Ali Sultonov**  
Independent Researcher  
troxtergrif@gmail.com  
https://github.com/Troxter222/psilogic

**Status:** arXiv preprint [arXiv:2607.16268](https://arxiv.org/abs/2607.16268) · Software DOI [10.5281/zenodo.18739857](https://doi.org/10.5281/zenodo.18739857)  
**PDF source:** [`arxiv/paper.tex`](arxiv/paper.tex) · notes synced 2026-09-06 (v0.6 defaults, fusion caveat)

Before the next arXiv bump: tables 1–3 against `benchmark/results/full/aggregate.csv` and `significance.csv`, figures in `arxiv/figures/` against the captions, citation block against README, wall-time note against the commit that produced the CSVs (pre-fusion vs fused), and this abstract against `arxiv/paper.tex`.

## Abstract

Adam and AdamW use the same update whether early gradients are noisy or the run has already settled. PsiLogic (ΨLogic) adds a dynamic Active Cancellation term to Adam. A dual EMA of scale-normalized gradient norms (the chaos detector) turns that term up when the statistics are unstable and lets it go to zero as training settles. Early steps are damped without a separate warmup schedule.

We compare PsiLogic with Adam, AdamW, and Lion under FairBench. Each optimizer gets its own learning-rate sweep, runs that share a seed start from the same initialization, and differences are Welch *t*-tests. The reference run is an NVIDIA H100 80GB (4 arenas, 3 seeds, 2000 steps, bf16 AMP).

PsiLogic has the best validation metric on NLP perplexity and ViT accuracy. On ResNet it beats Adam; the gap vs AdamW is not significant. On diffusion it ties Adam and AdamW. NLP perplexity 7.79 ± 0.18 vs 8.17 ± 0.08 (AdamW, *p* = 0.049). ViT top-1 0.244 ± 0.006 vs 0.223 ± 0.002 (AdamW, *p* = 0.015). ResNet top-1 0.222 ± 0.001 vs Adam 0.172 ± 0.004 (*p* = 0.001) and AdamW 0.219 ± 0.005 (*p* = 0.44). Diffusion MSE 0.02009 ± 0.00045 vs 0.01987 ± 0.00006 (*p* = 0.49). Peak GPU memory is about the same. On that June 2026 H100 run, before Triton fusion, wall time was 1.2–1.8× AdamW. v0.5+ has an optional fused CUDA backend with the same math. A fused FairBench re-run is not in this preprint.

Code, the FairBench harness, and the raw CSVs are in the repo.

**Keywords:** optimization, Adam, adaptive learning rate, deep learning, reproducibility

## 1. Introduction

The optimizer changes convergence speed, stability, and often the final metric. Adam (Kingma & Ba, 2015) and AdamW (Loshchilov & Hutter, 2019) are what most training code uses. Their update does not depend on whether the current gradient statistics look unstable or already quiet. At initialization gradients are large and noisy. Near convergence they are small. Adam applies the same kind of step in both cases.

PsiLogic adds a chaos-conditioned damping term to the Adam update. The term is largest when a dual EMA of normalized gradient norms indicates a spike, and it goes to zero on its own as training settles. It is a drop-in for `torch.optim.Adam`, with optional task presets (`PsiLogicNLP`, `PsiLogicGPT`, `PsiLogicViT`, `PsiLogicWhisper`).

Contributions:

1. Chaos-gated Active Cancellation on Adam, with unified decay and optional GC/AGC.
2. FairBench: per-optimizer LR sweep, matched inits, four arenas, Welch *t*-tests.
3. The H100 reference run under `benchmark/results/full/`, including the ties and the non-significant comparisons.

Diffusion ties Adam/AdamW. ResNet vs AdamW is not significant at three seeds. Step time on the June 2026 run is slower than AdamW. Those results are in the tables below.

## 2. Related Work

**Adam / AdamW.** Adam keeps bias-corrected first- and second-moment estimates and uses them as per-parameter rates (Kingma & Ba, 2015). AdamW splits weight decay out of the gradient step (Loshchilov & Hutter, 2019). That is the usual setup for Transformers.

**Lion.** Lion (Chen et al., 2023) uses a sign update and coupled weight decay. It uses less memory. It often needs its own LR, and on some from-scratch language-model runs it trails AdamW.

**Other adaptive and schedule-free methods.** Sophia (Liu et al., 2023) uses a cheap Hessian diagonal. Schedule-Free AdamW (Defazio et al., 2024) drops the explicit LR schedule and averages instead. Prodigy (Mishchenko & Defazio, 2023) estimates distance-to-optimum online. Muon (Jordan et al., 2024) orthogonalizes 2D updates. PsiLogic keeps the Adam moments and multiplies in a chaos-gated damping term. It does not replace the preconditioner or the schedule.

**Stability.** Gradient centralization (Yong et al., 2020) and adaptive gradient clipping (Brock et al., 2021) are the usual extra stabilizers. PsiLogic can use both. They are on in the task presets and off on the bare v0.6+ constructor (see §3.6). Learning-rate warmup (Goyal et al., 2017) is the scheduled version of the same idea: don't take a full step while early gradients are still noise. The chaos term does that from the gradient statistics, not from a step counter.

**Evaluation.** A shared learning rate is a common way one optimizer looks better than it is. FairBench gives each optimizer its own LR search.

## 3. Method

### 3.1 Update Rule

PsiLogic is Adam plus an Active Cancellation term:

```
θ_{t+1} = θ_t
         − η · m̂_t / (√v̂_t + ε)           [Adam step]
         − η · γ · P · chaos_t · θ_t         [active cancellation, when gated on]
```

Adam moments are the usual ones, with bias correction. Weight decay goes through one per-step shrinkage coefficient (Section 3.3).

### 3.2 Chaos Detector

`gn_t = ‖∇_t‖₂ / √(numel)` is the scale-normalized gradient norm.

```
fast_t = 0.90 · fast_{t-1} + 0.10 · gn_t     [τ ≈ 10 steps]
slow_t = 0.99 · slow_{t-1} + 0.01 · gn_t     [τ ≈ 100 steps]

ratio_t = fast_t / (slow_t + ε)
chaos_t = tanh(slow_t) · (1 + 0.5 · tanh(relu(ratio_t − 1)))
```

In adaptive mode (the default), cancellation turns on when `fast_t > τ_scale · slow_t` (`τ_scale = 2.0`). That is a relative spike, not a raw threshold on the gradient. As `slow_t → 0` at convergence, `chaos_t → 0`, and the extra term drops out. What is left is close to AdamW.

### 3.3 Unified Decay

Weight decay, cancellation, and any auxiliary penalty should not multiply. The product `(1 − ηλ)(1 − c)` over-shrinks early, when both terms are large. PsiLogic folds them into one coefficient per step and clamps it: `c_coeff ≤ max_cancel` (default 0.05). Optional cosine decay on γ is `gamma_T_max`.

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

Implementation: https://github.com/Troxter222/psilogic (`psilogic/optimizer.py`).

### 3.5 Comparison with Baselines

| Property | Adam | AdamW | Lion | PsiLogic |
|:---------|:----:|:-----:|:----:|:--------:|
| Per-parameter adaptive rates | ✓ | ✓ | ✗ | ✓ |
| Unified decay (with chaos) | ✗ | ✗ | ✗ | ✓ |
| Chaos-aware damping | ✗ | ✗ | ✗ | ✓ |
| Implicit early-phase damping | ✗ | ✗ | ✗ | ✓ |
| Batched `foreach` CUDA kernels | partial | ✓ | ✗ | ✓ |
| Optional fused Triton step (v0.5+) | ✗ | ✗ | ✗ | ✓ |

### 3.6 Package defaults and backends

The foreach path and the Triton fused CUDA path implement the same update as the scalar reference. Fusion does not change FairBench quality metrics. It changes step time. `pip install "psilogic[cuda]"` turns it on (`use_fused_cuda=True` when Triton is available).

From v0.6, `PsiLogic(params, lr=...)` uses `agc_clip=0.0` and `grad_centralize=False`. FairBench NLP follow-ups showed AGC + GC on TinyStories GPT-from-scratch hurt relative to AdamW. Task helpers (`PsiLogicNLP`, `PsiLogicViT`, and the others) and presets can still turn mild AGC/GC on. The headline numbers in this note are the June 2026 configs in `benchmark/results/full/`, not necessarily the bare v0.6 constructor.

## 4. FairBench Evaluation

### 4.1 Protocol

All headline numbers are from one run on an NVIDIA H100 80GB HBM3 (PyTorch 2.4.1+cu124, CUDA 12.4). The config is `benchmark/results/full/config.json`. When reproducing, pin the git commit that produced the CSVs, and cite the software DOI.

| Stage | Description |
|:------|:------------|
| Stage 1, LR sweep | 7 log-spaced LRs from 10⁻⁵ to 10⁻²; 500 steps each; best val metric wins |
| Stage 2, evaluation | Selected LR; 2000 steps; seeds {0, 1, 2}; identical init per seed |
| Shared | batch=64, bf16 AMP, grad_clip=1.0, cosine LR, 100-step warmup |
| Statistics | Mean ± std; Welch *t*-test (PsiLogic vs each baseline) |

PsiLogic uses fixed per-arena presets. Only LR is tuned, same as the baselines.

### 4.2 Arenas

| Arena | Model | Dataset | Metric |
|:------|:------|:--------|:-------|
| NLP | Small GPT | TinyStories | Perplexity ↓, val loss ↓ |
| ViT | ViT-Tiny patch16 224 | CIFAR-100 @ 224² | Top-1 acc ↑ |
| ResNet | ResNet-18 | Tiny ImageNet 200 | Top-1 acc ↑ |
| Diffusion | DDPM + UNet | CelebA @ 64² | Val MSE ↓ |

### 4.3 Main Results (Table 1)

Source: `benchmark/results/full/aggregate.csv`.

| Arena | Metric | Adam | AdamW | Lion | PsiLogic |
|:------|:-------|:----:|:-----:|:----:|:--------:|
| NLP | Perplexity ↓ | 13.66±0.22 | 8.17±0.08 | 21.04±1.41 | **7.79±0.18** |
| NLP | Val loss ↓ | 2.614±0.016 | 2.101±0.010 | 3.045±0.068 | **2.053±0.023** |
| ViT | Val acc ↑ | 0.079±0.003 | 0.223±0.002 | 0.213±0.002 | **0.244±0.006** |
| ResNet | Val acc ↑ | 0.172±0.004 | 0.219±0.005 | 0.205±0.007 | **0.222±0.001** |
| Diffusion | Val MSE ↓ | **0.01987±0.00006** | **0.01987±0.00006** | 0.02175±0.00025 | 0.02009±0.00045 |

**Selected LRs.** NLP: all `3.16×10⁻⁴`. ViT: Adam `3.16×10⁻⁵`, AdamW/PsiLogic `3.16×10⁻⁴`, Lion `10⁻⁴`. ResNet: Adam/Lion `10⁻⁴`, AdamW/PsiLogic `3.16×10⁻⁴`. Diffusion: Adam/AdamW/PsiLogic `10⁻³`, Lion `10⁻⁴`.

Wins: NLP perplexity, ViT. Beat Adam, not significant vs AdamW: ResNet. Tie: diffusion.

### 4.4 Statistical Significance (Table 2)

Welch *t*-test, PsiLogic vs baseline. \* *p* < 0.05, \*\* *p* < 0.01, \*\*\* *p* < 0.001. n.s. = not significant. Source: `benchmark/results/full/significance.csv`.

| Arena | Metric | vs Adam | vs AdamW | vs Lion |
|:------|:-------|:--------|:---------|:--------|
| NLP | Perplexity | \*\*\* | \* | \*\* |
| NLP | Val loss | \*\*\* | n.s. (*p*=0.054) | \*\*\* |
| ViT | Val acc | \*\*\* | \* | \*\* |
| ResNet | Val acc | \*\* | n.s. (*p*=0.44) | \* |
| Diffusion | Val MSE | n.s. | n.s. | \* |

Against AdamW, ViT and NLP perplexity are significant. ResNet and diffusion are not. NLP val loss vs AdamW is *p* = 0.054, so that one is not significant either, even though the mean is lower. Against Lion, the quality arenas above are significant; on diffusion Lion is the one that loses to Adam/AdamW, and PsiLogic is closer to those two than Lion is.

### 4.5 Compute Cost (Table 3)

Peak VRAM and wall time from the same June 2026 H100 CSVs, before fusion. A/W/L/P = Adam / AdamW / Lion / PsiLogic.

| Arena | Peak VRAM (MB) A/W/L/P | Wall time (s) A/W/L/P | PsiLogic / AdamW |
|:------|:----------------------|:---------------------|:----------------:|
| NLP | 458 / 458 / 445 / 458 | 46.6 / 45.9 / 38.2 / 55.2 | 1.20× |
| ViT | 1229 / 1229 / 1208 / 1229 | 95.2 / 98.5 / 98.6 / 176.7 | 1.79× |
| ResNet | 823 / 825 / 777 / 823 | 45.3 / 47.6 / 46.1 / 67.4 | 1.42× |
| Diffusion | 3780 / 3780 / 3768 / 3781 | 94.2 / 95.2 / 91.6 / 168.3 | 1.77× |

VRAM is within 3%, except Lion on some arenas (lower). The practical cost on this run is step time. Figure 4 and this table should be replaced after a fused H100 re-run.

### 4.6 Figures

Rendered in the arXiv PDF from `arxiv/figures/` (see `arxiv/paper.tex`):

| Figure | File | Caption |
|:-------|:-----|:--------|
| Fig. 1 | `vit_val_val_acc.png` | ViT validation accuracy (mean ± std) |
| Fig. 2 | `nlp_val_perplexity.png` | NLP perplexity |
| Fig. 3 | `resnet_val_val_acc.png` | ResNet top-1 accuracy |
| Fig. 4 | `vit_train_step_time_s.png` | ViT per-step wall time (pre-fusion) |

Not in this version: a side-by-side learning-curve figure (PsiLogic vs Lion vs AdamW) and a GPT `chaos_warmup` ablation figure. Tracked in [ROADMAP.md](ROADMAP.md). Do not cite them until they exist.

## 5. Ablations

Earlier ablations on a synthetic MLP (v0.3.x) found that gradient centralization and adaptive gradient clipping each help stability when they are combined with the chaos term. A mirror ablation set AdamW's weight decay to PsiLogic's cancellation magnitude at each step. That did not reproduce PsiLogic's per-parameter behavior, so the chaos signal is not the same thing as one global weight-decay schedule.

Those runs predate FairBench and are not used for the claims above. Component tests are in `tests/`. FairBench-scale ablations of γ, `max_cancel`, `chaos_warmup`, and GC/AGC on/off are not in this preprint. They are Phase 3B in ROADMAP.

## 6. Discussion

The ViT gap vs Adam (0.244 vs 0.079) is large and shows up early. The chaos term is a plausible reason: it cuts the step when gradient statistics are still swinging. After a per-optimizer LR sweep, NLP perplexity still favors PsiLogic over AdamW.

The cancellation term also shrinks the effective step during those swings. That is similar to LR warmup, except the amount comes from the gradient statistics instead of a step counter.

On ResNet, PsiLogic has the smallest cross-seed standard deviation of the four (±0.001 on accuracy). Three seeds is a thin estimate of that.

### Limitations

1. Three seeds. ResNet vs AdamW and diffusion vs AdamW are not significant.
2. 2000 steps per arena. Not ImageNet-scale and not LLM-scale.
3. Step time up to 1.79× AdamW on ViT (June 2026 H100, before fusion). The chaos detector is a few scalars. The extra time on that run is in the step implementation. v0.5+ Triton fusion is aimed at ≤1.25× on Ampere+ and does not change the math. A fused FairBench re-run is not in yet.
4. Diffusion: no quality win over Adam/AdamW at this budget.
5. No convergence proof. The stability claims are empirical.
6. Nobody outside this repo has re-run it.
7. From v0.6 the bare constructor differs from some presets and from some FairBench configs. Record the preset if you compare against this note.

The reference run is one NVIDIA H100. A fixed 2000-step horizon can favor a method that starts fast. The LR sweep is per optimizer, but warmup, clip, and cosine are shared. The models are small next to a production LLM or ImageNet-1k.

GPU-hours for the full suite depend on queue time and retries. Table 3 is the public per-stage cost. A fused re-run should report total GPU-hours as well.

## 7. Reproducibility

```bash
git clone https://github.com/Troxter222/psilogic
cd psilogic && pip install -e ".[benchmark]" && pip install -r benchmark/requirements.txt

# long run in tmux, can resume
./run_fairbench.sh

# or step by step
cd benchmark
python -m fairbench.download --data-root ./data
python -m fairbench --data-root ./data --output-dir results/full

# smoke test
python -m fairbench --smoke-test --device cpu --no-amp --num-workers 0
```

Reference outputs: `benchmark/results/full/{aggregate,summary,significance,config}.csv|json`  
Software DOI: [10.5281/zenodo.18739857](https://doi.org/10.5281/zenodo.18739857)  
PyPI: `pip install psilogic`  
Pin the git commit that matches the CSVs you cite.

## 8. Conclusion

PsiLogic adds a chaos-gated Active Cancellation term to Adam. The term is large while gradient statistics are unstable and goes to zero at convergence. On the H100 FairBench run it wins NLP perplexity and ViT accuracy, beats Adam and ties AdamW on ResNet, and ties on diffusion. Wall time on that run was 1.2–1.8× AdamW, before fusion. Still open: a fused re-run, more seeds, longer training, FairBench-scale ablations, and a replication that is not this repo.

## References

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., et al. (2023). Symbolic discovery of optimization algorithms. *NeurIPS*.

Defazio, A., et al. (2024). The road less scheduled. *arXiv* (Schedule-Free optimizers).

Jordan, K., et al. (2024). Muon: An optimizer for hidden representations. *arXiv*.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.

Liu, H., et al. (2023). Sophia: A scalable stochastic second-order optimizer for language model pre-training. *arXiv*.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.

Mishchenko, K., & Defazio, A. (2023). Prodigy: An expeditiously adaptive parameter-free learner. *arXiv*.

Yong, H., Huang, J., Hua, X., & Zhang, L. (2020). Gradient centralization. *ECCV*.

Brock, A., et al. (2021). High-performance large-scale image recognition without normalization. *ICML*.

Goyal, P., et al. (2017). Accurate, large minibatch SGD. *arXiv:1706.02677*.

Dosovitskiy, A., et al. (2021). An image is worth 16×16 words. *ICLR*.

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning. *CVPR*.

Sultonov, A. (2026). PsiLogic software & FairBench artifacts. [doi:10.5281/zenodo.18739857](https://doi.org/10.5281/zenodo.18739857).

## Appendix A. Per-Seed ViT Accuracy

| Seed | Adam | AdamW | Lion | PsiLogic |
|-----:|:----:|:-----:|:----:|:--------:|
| 0 | 0.078 | 0.226 | 0.214 | **0.238** |
| 1 | 0.083 | 0.222 | 0.211 | **0.247** |
| 2 | 0.076 | 0.221 | 0.213 | **0.249** |

Full per-seed tables: `benchmark/results/full/summary.csv`.

## Appendix B. Archived Experiments

Pre-FairBench results (CIFAR-10 on an A40, BERT, AG News, and others) are in `OLD_RESULTS.md`. They are not used for claims in this preprint.

## Appendix C. LR Sweep Grid

Stage-1 candidates, shared log grid: `{1e-5, 3.16e-5, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2}`. 500 steps each. Best validation metric picks the Stage-2 LR. Winners are listed under Table 1. Sweep logs: `benchmark/results/full/` and `benchmark/logs.txt`.

---

*Ali Sultonov · Independent Researcher · arXiv:2607.16268 · https://github.com/Troxter222/psilogic*
