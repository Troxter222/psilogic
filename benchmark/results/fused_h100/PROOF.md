# ΨLogic Gate 1C — H100 fused proof

Generated: 20260924T174634Z

## Hardware

- GPU: **NVIDIA H100 80GB HBM3** (79.18 GB)
- Torch / CUDA: 2.8.0+cu128 / 12.8
- AMP: bfloat16
- Seeds / steps: [0, 1, 2] / 2000

## Quality (mean over seeds)

| Arena | Metric | AdamW | ΨLogic | Δ vs AdamW |
|-------|--------|------:|-------:|-----------:|
| nlp | perplexity | 8.201 | **8.152** | -0.6% |
| vit | val_acc | 0.2825 | **0.2672** | -5.4% |
| resnet | val_acc | 0.2214 | **0.2208** | -0.3% |

## Speed (wall_time ΨLogic / AdamW)

| Arena | AdamW (s) | ΨLogic (s) | Ratio | Target |
|-------|----------:|-----------:|------:|:------:|
| nlp | 20.2 | 22.2 | **1.10x** | <=1.25x PASS |
| vit | 114.4 | 117.2 | **1.02x** | <=1.25x PASS |
| resnet | 35.9 | 37.2 | **1.04x** | <=1.25x PASS |

Gate 1C (ViT): ratio <= 1.25x **or** `profile_optimizer.py` median step <= 1.25x.

## Microbench (`profile_optimizer.py`)

```
Device: cuda
Parameters: 202,506 (50 tensors)
AdamW median step: 0.603 ms
PsiLogic (foreach) median step: 4.986 ms (8.27x)
PsiLogic (fused) median step: 1.371 ms (2.27x)

```

## Significance file

See `significance.csv` for Welch t-tests / Cohen's d.

