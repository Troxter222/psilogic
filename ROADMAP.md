# ΨLogic roadmap

Updated 2026-09-06. Current release: v0.6.0. Target for a stable v1.0: Q4 2027.

`[ ]` todo · `[~]` in progress · `[x]` done. A **gate** blocks the next phase.

## Status

| Gate | Status | Evidence |
|:-----|:-------|:---------|
| **0** | passed | CI green, modular package, PyPI v0.4 |
| **1A** | passed | FairBench ViT 0.244 vs Lion 0.213 ([`aggregate.csv`](benchmark/results/full/aggregate.csv)) |
| **1B** | passed | FairBench NLP PPL 7.79 vs AdamW 8.17 |
| **1C** | in progress | Triton fusion shipped (v0.5); H100 FairBench fused re-run pending |
| **2** | open | ImageNet R50 + GPT-2 medium |
| **3** | open | ≤15% A100 overhead + published ablations |
| **4** | open | arXiv live, public W&B, HF docs |
| **5** | open | v1.0, ≥5/7 arenas, 1B run |

Gate 1A note: early tables targeted CIFAR Top-1 ≥ 0.48 under a different recipe. The passed gate is FairBench ViT (ViT-Tiny / CIFAR-100, 2000 steps, per-optimizer LR sweep). Absolute accuracy is lower; the comparison is fairer. Don't mix the two targets.

```
Sep 2026           Gate 1C — fused FairBench re-run, refresh overhead numbers
Oct 2026 – Mar 27  Gate 2  — ImageNet-1k (CV-1) + GPT-2 medium (NLP-2)
Apr–Jun 2027       Phase 3 — ablations, Muon hybrid explore
Jul–Sep 2027       Phase 4 — arXiv submit (paper.tex SoT), HF docs
Q4 2027            Gate 5  — v1.0, ≥5/7 arenas, 1B flagship
```

Now: Gate 1C — FairBench H100 re-run with `psilogic[cuda]`; refresh wall-time tables in README / PAPER / this file.

Next: ImageNet ResNet-50 + GPT-2 medium budgets; learning-curve figures; W&B tagging.

Later: ablation suite, arXiv submit, ecosystem PRs, v1.0 API freeze, 1B flagship.

## Results (v0.6.0)

Quality from FairBench, Jun 2026, H100. Fusion doesn't change these.

| Arena | Task | AdamW | Lion | ΨLogic | Notes |
|-------|------|-------|------|--------|-------|
| NLP | GPT / TinyStories · 3 seeds | 8.17 PPL | 21.04 | **7.79** | *p*=0.049 vs AdamW |
| ViT | ViT-T / CIFAR-100 · 3 seeds | 0.223 | 0.213 | **0.244** | *p*=0.015 vs AdamW |
| ResNet | R-18 / Tiny ImageNet | 0.219 | 0.205 | **0.222** | vs Adam *p*=0.001; vs AdamW *p*=0.44 (tie) |
| Diffusion | DDPM / CelebA | **0.0199** | 0.0218 | 0.0201 | *p*=0.49 (tie AdamW) |
| Legacy | [OLD_RESULTS.md](OLD_RESULTS.md) | — | — | — | pre-FairBench only |

Wall time, ΨLogic / AdamW. Scorecard numbers are pre-fusion.

| Arena | Pre-fusion H100 (Jun 2026) | Target (fused) | Status |
|-------|----------------------------|----------------|--------|
| NLP | 1.20× | ≤ 1.25× | `[ ]` re-run pending |
| ViT | **1.79×** | **≤ 1.25×** | `[~]` local TinyViTLike **1.96×** on GTX 1650 (Turing); not the gate hardware |
| ResNet | 1.42× | ≤ 1.25× | `[ ]` re-run pending |
| Diffusion | 1.77× | ≤ 1.25× | `[ ]` re-run pending |

Close Gate 1C on Ampere+ / H100 with `psilogic[cuda]`, then start Gate 2. v0.6 turned AGC/GC off on bare `PsiLogic` after NLP follow-ups (they hurt TinyStories GPT-scratch vs AdamW).

## What v1 means

All of these, not a subset:

1. Win or statistically tie (*p* > 0.05) AdamW **and** Lion on ≥ 5 of 7 arenas (Phase 5).
2. Every headline number reproducible with one command + public W&B logs.
3. Listed in HuggingFace `Trainer` docs (merged PR or official recommendation).
4. ≥ 1,000 PyPI downloads/month **or** cited in ≥ 3 independent papers/repos. Check [pypistats](https://pypistats.org/packages/psilogic) / PePy; Scholar alerts for `PsiLogic` / arXiv:2607.16268.
5. Semver v1.0, no breaking changes for 6 months.
6. ≤ 15% step-time overhead vs AdamW on A100, written down.

Until Gate 2 (≈ Mar 2027) I'm not doing: new optimizer variants unrelated to chaos cancellation (full Muon rewrite, second-order), mobile/edge packaging, replacing FairBench with a closed suite, promotion before fused overhead numbers and Gate 2 exist, or public API churn without CHANGELOG + README notes.

## Risks

| Risk | What to do |
|:-----|:-----------|
| No reliable H100 / A100 for Gate 1C / Gate 2 | Rent; write down the exact SKU; delay the gate if needed |
| n=3 underpowered for ties | Report *p*-values; plan n≥5 on flagship runs |
| Nobody replicates | One-command FairBench; Zenodo DOI |
| README / PAPER / ROADMAP / tex disagree | Update them together after every bench ([docs sync](#docs-sync)) |
| Fusion helps Turing less than Ampere | Gate hardware = Ampere+; 1650 microbench is diagnostic only |

## Phase 0 — done (v0.4)

Package split + CI.

- [x] Split monolith → `optimizer.py`, `_chaos.py`, `presets.py`, `param_groups.py`, `convenience.py`
- [x] `py.typed`, Ruff, GitHub Actions CI, GPU test markers
- [x] `psilogic.debug`, CHANGELOG, README structure, FairBench reproduce path
- Gate 0: `pytest` green, `ruff` clean, PyPI v0.4.0 — passed

## Phase 1A / 1B — done

**1A — ViT (beat Lion).** `PsiLogicViT` + `vision_defaults` / FairBench ViT. H100: ΨLogic **0.244** vs Lion **0.213** (*p* < 0.001). Shipped `vit_param_groups`, `PsiLogicViT`, debug helpers, FairBench ViT wiring, `tests/test_vit_preset.py`. Still missing: learning-curve plot epoch 1–15 (Phase 4 / debt).

**1B — GPT / from-scratch LM (beat AdamW).** Bare / `gpt_scratch_defaults` aligned with FairBench NLP. H100: ΨLogic PPL **7.79** vs AdamW **8.17** (*p* = 0.049). Shipped `gpt_param_groups`, auto warmup, `PsiLogicGPT`, chaos warmup tests. Still missing: GPT warmup ablation figure in PAPER.

**v0.6.** Bare `PsiLogic` uses `agc_clip=0.0`, `grad_centralize=False`. Task helpers still opt in. See [CHANGELOG.md](CHANGELOG.md) `[0.6.0]`.

## Phase 1C — performance (v0.5.x) · in progress

Target: **≤1.25× AdamW** step time on ViT-like models with `use_fused_cuda=True` on Ampere+ (A100/H100). Wall times above are pre-fusion.

| ID | Deliverable | Status |
|----|-------------|--------|
| P1 | Triton fused step (`psilogic/_cuda/`) + multi-tensor kernel | `[x]` 0.5.x |
| P2 | `tests/test_step_overhead.py` + `scripts/profile_optimizer.py` | `[x]` |
| P3 | `tests/test_numerical_parity.py` — scalar / foreach / fused | `[x]` 0.5.0 |
| P4 | Re-run FairBench with `psilogic[cuda]` on H100 | `[ ]` pending |
| P5 | Update README + overhead tables | `[~]` docs say pending; numbers not replaced |
| P6 | CI: optional self-hosted GPU job for fused overhead | `[ ]` |

```bash
pip install -e ".[cuda,benchmark]"
cd benchmark
python -m fairbench --arenas vit resnet diffusion --data-root ./data --output-dir results/fused
python ../scripts/profile_optimizer.py
```

Gate 1C: FairBench ViT wall time ΨLogic/AdamW **≤ 1.25×** (mean over 3 seeds), **or** `profile_optimizer.py` median step ratio ≤ 1.25× on Ampere+ hardware.

## Phase 2 — scale (v0.7 – v0.8) · Oct 2026 – Mar 2027

Toy budgets are not enough. Rough compute: CV-1 ≈ 3 seeds × 90 epoch ResNet-50 ImageNet (a few hundred A100-hours, depends on nodes); NLP-2 ≈ 3 seeds × 50k GPT-2 medium steps, same order. Book the cloud spend before starting.

### 2A — vision

| Benchmark | Model | Dataset | Epochs | Seeds | vs | Target |
|-----------|-------|---------|--------|-------|-----|--------|
| **CV-1** | ResNet-50 | ImageNet-1k | 90 | 3 | AdamW, Lion | Top-1 ≥ best baseline |
| **CV-2** | ViT-Base | ImageNet-1k | 300 | 3 | AdamW, Lion | Top-1 ≥ best baseline |
| **CV-3** | ConvNeXt-T | CIFAR-100 | 100 | 5 | AdamW, Lion | Beat FairBench ViT protocol |

- [ ] **`benchmark/imagenet/`** — DDP train script, AMP bf16, cosine LR
- [x] `profile_step_time` + `scripts/profile_optimizer.py`
- [ ] Non-deprecated `torch.amp` in all scale benchmarks
- [x] `tests/test_grad_accum.py`
- [ ] Multi-GPU DDP scaling note (2×GPU)

### 2B — NLP / LM

| Benchmark | Model | Dataset | Steps | Seeds | vs | Target |
|-----------|-------|---------|-------|-------|-----|--------|
| **NLP-1** | BERT-large | GLUE (MNLI, QQP, QNLI, SST-2) | 3 ep each | 3 | AdamW | Avg ≥ AdamW |
| **NLP-2** | GPT-2 medium (345M) | OpenWebText | 50k | 3 | AdamW | PPL ≤ AdamW |
| **NLP-3** | Whisper-small | LibriSpeech fine-tune | 10 ep | 3 | AdamW | WER ≤ AdamW |

- [x] `PsiLogicWhisper`, `whisper_defaults()`, `glue_defaults()`
- [ ] FSDP: assert loss decreases (`tests/test_fsdp.py`)
- [ ] Peak VRAM vs AdamW on GPT-2 medium

Gate 2: win or tie on **CV-1 (ResNet-50 ImageNet)** and **NLP-2 (GPT-2 medium)**.

## Phase 3 — algorithm (v0.9) · Apr–Jun 2027

| Feature | Status |
|---------|--------|
| Auto-γ scheduler (`gamma_auto`) | `[x]` 0.4.0 |
| Per-layer chaos sync DDP | `[x]` 0.4.0 |
| Fused CUDA kernel | `[x]` 0.5.0 |
| `PsiLogic.auto(model)` | `[x]` 0.4.0 |
| State dict versioning | `[x]` 0.4.0 |
| ValueError validation | `[x]` 0.4.0 |
| Muon compatibility (hybrid 2D) | `[ ]` explore |

Ablations in `PAPER.md` / `arxiv/paper.tex` + W&B. FairBench-scale; synthetic-only is not enough.

- [ ] GC on/off
- [ ] AGC on/off
- [ ] adaptive_tau on/off
- [ ] quantum_decay on/off
- [ ] lion_mode on/off
- [x] chaos disabled / mirror test
- [ ] γ schedule: constant vs cosine vs auto
- [ ] GPT `chaos_warmup` ablation (from 1B docs debt)

Gate 3: step overhead ≤ 15% vs AdamW on A100, ablation suite published, auto-config on 3 unseen architectures.

## Phase 4 — paper and docs (v0.95) · Jul–Sep 2027

- [x] LaTeX — [arxiv/paper.tex](arxiv/paper.tex) (source of truth for PDF)
- [x] PDF build — [scripts/build_arxiv_pdf.py](scripts/build_arxiv_pdf.py)
- [ ] Sync tex + [PAPER.md](PAPER.md) with fused-CUDA overhead + v0.6 defaults
- [ ] Submit / update arXiv (cs.LG)
- [ ] Update Zenodo DOI on each major scientific release
- [ ] Reproduce one published AdamW baseline curve; Lion ViT match/beat writeup
- [ ] Convergence sketch with stated assumptions

| Platform | Status |
|----------|--------|
| HuggingFace Trainer helpers | `[x]` |
| Lightning helpers | `[x]` |
| torchtune example YAML | `[x]` |
| Axolotl / LLaMA-Factory PRs | `[ ]` external |
| HF example notebook | `[ ]` |
| README “works with HF” badge | `[ ]` |

- [ ] Public W&B project (benches tagged by version)
- [ ] Writeup after Gate 1C and Gate 2, not before
- [ ] PyPI downloads badge
- [ ] `OPTIMIZER_COMPARISON.md` — include the losses

Gate 4: arXiv live, HF path documented, W&B public, ≥ 500 PyPI downloads/month.

## Phase 5 — v1.0 · Q4 2027

- [ ] Semver **v1.0.0** API freeze
- [ ] mypy strict on `psilogic/`
- [ ] 100% public docstring coverage
- [ ] Sphinx docs on GitHub Pages
- [ ] `torch.optim` compatibility audit
- [x] Deprecation policy in [CONTRIBUTING.md](CONTRIBUTING.md)
- [x] [SECURITY.md](SECURITY.md)
- [ ] LLaMA-style **1B** · 100k steps · OpenWebText vs AdamW vs Lion (4×A100-class). Target: PPL ≤ AdamW at same steps, or target PPL in ≤ 80% steps

| # | Arena | Must beat | Overlap |
|---|-------|-----------|---------|
| 1 | CIFAR-10 / ResNet-18 | AdamW | Partial |
| 2 | CIFAR-100 / ViT-Small | Lion | FairBench ViT `[x]` |
| 3 | ImageNet-1k / ResNet-50 | AdamW | Gate 2 CV-1 |
| 4 | BERT-base / SST-2 | AdamW | HF example |
| 5 | GPT-2 / Wikitext-2 scratch | AdamW | FairBench NLP partial |
| 6 | GPT-2 medium / OpenWebText | AdamW | Gate 2 NLP-2 |
| 7 | nanoGPT / Tiny Shakespeare | AdamW | — |

- [ ] One-command reproduction + optional nightly GPU CI
- [ ] Venue: ICLR / NeurIPS 2028 with the 1B result

Gate 5: ≥ 5/7 green, v1.0 on PyPI, Sphinx live, HF path, 1B published.

## After v1.0 (2028+)

- [ ] First-class HF `Trainer` enum / PyTorch docs mention
- [ ] Chaos dashboard (W&B panel)
- [ ] FSDP global chaos; Shampoo/Muon + chaos hybrids
- [ ] Early-stopping proxy via `slow_EMA`
- [ ] Hardware-partner writeups

## Debt

| Item | Action |
|------|--------|
| FairBench results pre-fusion | Re-run with `psilogic[cuda]`, commit new aggregate + SHA |
| `gamma_auto` / `sync_chaos_ddp` lightly tested | Behavioral + multi-process tests |
| Whisper / GLUE presets | Smoke tests |
| FSDP loss assertion | Extend `tests/test_fsdp.py` |
| GPU CI gap | Self-hosted or nightly GPU workflow |
| arXiv / PAPER out of sync with fusion + v0.6 | Sync before next arXiv bump |
| Learning-curve / warmup ablation figures | Close or mark deferred in PAPER |
| Axolotl / LLaMA-Factory | External PRs |

## Tests

GPU-marked tests (`@pytest.mark.gpu`, `@pytest.mark.multi_gpu`) skip on CPU CI.

| Area | Covered | Gap |
|------|---------|-----|
| Core optimizer | convergence, Lion mode, γ decay, checkpoints | — |
| Presets | ViT, GPT, auto-config | Whisper, GLUE |
| Chaos | warmup, debug | `gamma_auto` behavior |
| Backends | numerical parity, step overhead | GPU overhead not enforced in CI |
| Distributed | DDP, FSDP smoke | `sync_chaos_ddp`, FSDP loss ↓ |
| Integrations | HF + Lightning | — |
| Training patterns | grad accum, AMP, compile | — |
| FairBench | manual | No pytest e2e arenas |

## Versions

| Version | Phase | What | Status |
|---------|-------|------|--------|
| **0.4.0** | 0 | Modular package, CI | Released |
| **0.5.0** | 1A/1B + 1C partial | Presets, Triton fusion, parity | Released |
| **0.6.0** | 1 wrap-up | Safer bare defaults (AGC/GC off), docs | **Released (current)** |
| **0.6.x / 0.5.1-perf** | 1C | Fused FairBench re-run, overhead docs | Planned |
| **0.7.0** | 2A | ResNet-50 ImageNet | Planned |
| **0.8.0** | 2B | GPT-2 medium, GLUE, Whisper benches | Planned |
| **0.9.0** | 3 | Ablations, Muon explore, ≤15% overhead | Planned |
| **0.95.0** | 4 | arXiv sync, HF docs, W&B | Planned |
| **1.0.0** | 5 | API freeze, 5/7 arenas, 1B | Planned |

Shipped gates go in [CHANGELOG.md](CHANGELOG.md) with date + version.

## Docs sync

After a benchmark or default change, update together:

1. `benchmark/results/…` (or note “pending”)
2. [README.md](README.md) tables
3. This file (results + gate table)
4. [PAPER.md](PAPER.md) and `arxiv/paper.tex` if the PDF claims the number
5. [CHANGELOG.md](CHANGELOG.md) if it's user-facing

Check items off as they ship. Don't let the gate table get ahead of the CSVs. `pytest tests/ -v` before a release.
