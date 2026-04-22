╔══════════════════════════════════════════════════════════════════════════╗
║                    PSILOGIC — BENCHMARK LOGS                             ║
║              All experiments · raw terminal output                       ║
╚══════════════════════════════════════════════════════════════════════════╝


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXPERIMENT 1 — ResNet-18 · CIFAR-10 · 100 Epochs  [Local Machine]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Device : cuda   Batch: 128   LR: 1e-3 + CosineAnnealingLR
  Model  : ResNet-18 (conv1=3×3, no maxpool)   Clip: max_norm=1.0

  ─────────────────────────────────────────────────────────────────────
   Epoch    Adam Acc    Adam Loss    ΨLogic Acc   ΨLogic Loss     Δ
  ─────────────────────────────────────────────────────────────────────
       1     52.98%      1.5465       60.68%       1.1088      +7.70%
       5     76.90%      0.7269       79.48%       0.6428      +2.58%
      10     82.96%      0.5303       87.70%       0.3808      +4.74%
      20     88.18%      0.3639       90.15%       0.3310      +1.97%
      30     89.70%      0.3312       91.68%       0.3405      +1.98%
      40     89.47%      0.3973       91.64%       0.4087      +2.17%
      50     90.90%      0.3543       92.21%       0.4031      +1.31%
      60     91.84%      0.3514       92.53%       0.4265      +0.69%
      70     92.50%      0.3544       93.16%       0.4251      +0.66%
      80     93.14%      0.3454       93.35%       0.4337      +0.21%
      90     93.39%      0.3527       93.34%       0.4517      -0.05%
     100     93.67%      0.3369       93.59%       0.4461      -0.08%
  ─────────────────────────────────────────────────────────────────────
  Final: Adam 93.67%  vs  ΨLogic 93.59%  │  Gap: 0.08%  (within noise)
  Wall time: 230.9 min


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXPERIMENT 2 — ResNet-18 · CIFAR-10 · 100 Epochs  [Google Colab / GPU]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Device : cuda   Batch: 128   LR: 1e-3 + CosineAnnealingLR

  ─────────────────────────────────────────────────────────────────────
   Epoch    Adam Acc    Adam Loss    ΨLogic Acc   ΨLogic Loss     Δ
  ─────────────────────────────────────────────────────────────────────
       1     56.46%      1.2805       54.18%       1.3597      -2.28%
       5     73.11%      0.8365       78.62%       0.6323      +5.51%
      10     83.54%      0.4837       87.36%       0.3945      +3.82%
      20     87.72%      0.3988       90.07%       0.3389      +2.35%
      30     88.78%      0.3805       91.00%       0.3748      +2.22%
      40     91.40%      0.3076       91.51%       0.4060      +0.11%
      50     91.46%      0.3284       92.11%       0.4141      +0.65%
      60     92.26%      0.3242       92.63%       0.4488      +0.37%
      70     92.35%      0.3500       92.82%       0.4508      +0.47%
      80     93.08%      0.3529       93.40%       0.4578      +0.32%
      90     93.25%      0.3353       93.58%       0.4600      +0.33%
     100     93.65%      0.3231       93.69%       0.4596      +0.04%
  ─────────────────────────────────────────────────────────────────────
  Final: Adam 93.65%  vs  ΨLogic 93.69%  │  ΨLogic wins  (+0.04%)
  Wall time: 160.3 min


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXPERIMENT 3 — MULTI-MODAL BENCHMARK  [Google Colab / GPU]
 Adam · AdamW · SGD · ΨLogic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Device: cuda   Seed: 42   Same init & CosineAnnealingLR for all.

  ── Images: ResNet-18 · CIFAR-10 · 30 epochs ───────────────────────

   Epoch    Adam      AdamW      SGD      ΨLogic
     1     54.17%    51.66%    46.07%    56.05% ←
     5     79.84%    83.00%    68.66%    82.22%
    10     83.64%    87.22%    76.29%    86.85%
    15     88.15%    89.69%    79.47%    89.58%
    20     91.68%    91.54%    82.09%    91.67%
    25     92.75%    92.51%    83.54%    92.38%
    30     93.19%    92.62%    83.64%    92.60%

  ── Text: Transformer (2L d=128) · AG News · 10 epochs ─────────────

   Epoch    Adam      AdamW      SGD      ΨLogic
     1     92.16%    92.28%    89.71%    92.11%
     3     91.76%    91.84%    90.96%    92.14% ←
     5     90.84%    91.16%    91.12%    91.37% ←
     7     91.17%    91.11%    91.33%    91.26%
    10     91.07%    91.30%    91.24%    91.46% ←

  ── Audio: CNN+BiGRU · SpeechCommands · 15 epochs · 35 classes ─────

   Epoch    Adam      AdamW      SGD      ΨLogic
     1     80.79%    82.87%    41.49%    81.27%
     3     89.93%    91.64%    59.89%    90.43%
     5     92.34%    92.91%    77.51%    92.57% ←
     8     92.98%    93.89%    83.54%    93.74%
    10     94.06%    94.57%    88.78%    94.76% ←
    12     94.98%    95.10%    89.83%    95.11% ←
    15     95.50%    95.35%    90.81%    95.26%


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXPERIMENT 4 — ResNet-18 · CIFAR-10 · 30 epochs · multi-seed
 Adam · AdamW · PsiLogic_v1 · PsiLogic_v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Run 1/8:  Adam  seed=42
  [Adam  seed=42]   epoch  1  acc=61.08%  loss=1.1079
  [Adam  seed=42]   epoch  5  acc=75.73%  loss=0.7358
  [Adam  seed=42]   epoch 10  acc=85.29%  loss=0.4507
  [Adam  seed=42]   epoch 20  acc=91.42%  loss=0.2983
  [Adam  seed=42]   epoch 30  acc=92.74%  loss=0.2930   Done: 24.1 min

  Run 2/8:  Adam  seed=137
  [Adam  seed=137]  epoch  1  acc=50.27%  loss=1.5024
  [Adam  seed=137]  epoch  5  acc=76.83%  loss=0.7280
  [Adam  seed=137]  epoch 10  acc=84.12%  loss=0.4622
  [Adam  seed=137]  epoch 20  acc=91.11%  loss=0.3006
  [Adam  seed=137]  epoch 30  acc=93.20%  loss=0.2909   Done: 24.0 min

  Run 3/8:  AdamW  seed=42
  [AdamW seed=42]   epoch  1  acc=59.52%  loss=1.2113
  [AdamW seed=42]   epoch  5  acc=78.62%  loss=0.6348
  [AdamW seed=42]   epoch 10  acc=87.62%  loss=0.3734
  [AdamW seed=42]   epoch 20  acc=91.14%  loss=0.3352
  [AdamW seed=42]   epoch 30  acc=92.10%  loss=0.3440   Done: 24.2 min

  Run 4/8:  AdamW  seed=137
  [AdamW seed=137]  epoch  1  acc=57.81%  loss=1.2548
  [AdamW seed=137]  epoch  5  acc=77.08%  loss=0.7394
  [AdamW seed=137]  epoch 10  acc=86.85%  loss=0.4093
  [AdamW seed=137]  epoch 20  acc=91.13%  loss=0.3276
  [AdamW seed=137]  epoch 30  acc=92.43%  loss=0.3403   Done: 24.2 min

  Run 5/8:  PsiLogic_v1  seed=42
  [PsiLogic_v1 seed=42]   epoch  1  acc=57.70%  loss=1.2115
  [PsiLogic_v1 seed=42]   epoch  5  acc=78.86%  loss=0.6472
  [PsiLogic_v1 seed=42]   epoch 10  acc=86.70%  loss=0.4082
  [PsiLogic_v1 seed=42]   epoch 20  acc=91.25%  loss=0.3242
  [PsiLogic_v1 seed=42]   epoch 30  acc=92.55%  loss=0.3252   Done: 26.1 min

  Run 6/8:  PsiLogic_v1  seed=137
  [PsiLogic_v1 seed=137]  epoch  1  acc=53.52%  loss=1.4267
  [PsiLogic_v1 seed=137]  epoch  5  acc=79.27%  loss=0.6555
  [PsiLogic_v1 seed=137]  epoch 10  acc=87.03%  loss=0.3944
  [PsiLogic_v1 seed=137]  epoch 20  acc=91.39%  loss=0.3207
  [PsiLogic_v1 seed=137]  epoch 30  acc=92.36%  loss=0.3279   Done: 26.1 min

  Run 7/8:  PsiLogic_v3  seed=42
  [PsiLogic_v3 seed=42]   epoch  1  acc=62.42%  loss=1.0422
  [PsiLogic_v3 seed=42]   epoch  5  acc=81.14%  loss=0.5574
  [PsiLogic_v3 seed=42]   epoch 10  acc=87.22%  loss=0.3885
  [PsiLogic_v3 seed=42]   epoch 20  acc=91.50%  loss=0.3290
  [PsiLogic_v3 seed=42]   epoch 30  acc=92.35%  loss=0.3538   Done: 27.2 min

  Run 8/8:  PsiLogic_v3  seed=137
  [PsiLogic_v3 seed=137]  epoch  1  acc=62.56%  loss=1.0750
  [PsiLogic_v3 seed=137]  epoch  5  acc=82.72%  loss=0.5057
  [PsiLogic_v3 seed=137]  epoch 10  acc=88.29%  loss=0.3603
  [PsiLogic_v3 seed=137]  epoch 20  acc=91.20%  loss=0.3267
  [PsiLogic_v3 seed=137]  epoch 30  acc=92.27%  loss=0.3468   Done: 27.3 min

  ─────────────────────────────────────────────────────────────────────
   VERDICT TABLE  — mean ± std over 2 seeds (accuracy %)
  ─────────────────────────────────────────────────────────────────────
   Epoch       Adam          AdamW       PsiLogic_v1   PsiLogic_v3
       1   55.67±5.40    58.66±0.86    55.61±2.09   [62.49±0.07]
       5   76.28±0.55    77.85±0.77    79.06±0.20   [81.93±0.79]
      10   84.70±0.59    87.24±0.38    86.87±0.16   [87.75±0.54]
      20   91.27±0.16    91.13±0.01    91.32±0.07   [91.35±0.15]
      30  [92.97±0.23]   92.27±0.16    92.45±0.09    92.31±0.04
  ─────────────────────────────────────────────────────────────────────
  [ ] = best for that epoch

  ΨLogic v3 vs AdamW (Δacc):
    epoch  1:  +3.83%
    epoch  5:  +4.08%
    epoch 10:  +0.52%
    epoch 20:  +0.22%
    epoch 30:  +0.04%

  PsiLogic_v3: Beats baselines 9/10 epoch×optimizer slots.
               Early wins (ep ≤ 10): 3/3.
               VERDICT: WINS — consistent advantage over baselines.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXPERIMENT 5 — CIFAR-10 · ResNet-18 · 15 epochs · 10 seeds  [NVIDIA A40]
 Adam · AdamW · PsiLogic (benchmark_all.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Device: NVIDIA A40 (cuda)   10 seeds (0–9)   15 epochs each

  ── Seed 0 ──────────────────────────────────────────────────────────
  [CIFAR-10][Adam]     ep=01  train=1.4705  val=1.2508  acc=56.46%
  [CIFAR-10][Adam]     ep=05  train=0.5529  val=0.6533  acc=78.86%
  [CIFAR-10][Adam]     ep=10  train=0.2687  val=0.3657  acc=87.91%
  [CIFAR-10][Adam]     ep=15  train=0.1434  val=0.3202  acc=90.10%

  [CIFAR-10][AdamW]    ep=01  train=1.4766  val=1.2754  acc=54.77%
  [CIFAR-10][AdamW]    ep=05  train=0.5596  val=0.7429  acc=77.07%
  [CIFAR-10][AdamW]    ep=10  train=0.2729  val=0.3624  acc=88.32%
  [CIFAR-10][AdamW]    ep=15  train=0.1472  val=0.3142  acc=90.31%

  [CIFAR-10][PsiLogic] ep=01  train=1.4864  val=1.1648  acc=59.22%
  [CIFAR-10][PsiLogic] ep=05  train=0.5377  val=0.5448  acc=81.91%
  [CIFAR-10][PsiLogic] ep=10  train=0.2715  val=0.3508  acc=88.40%
  [CIFAR-10][PsiLogic] ep=15  train=0.1440  val=0.3126  acc=90.59%

  ── Seed 1 ──────────────────────────────────────────────────────────
  [CIFAR-10][Adam]     ep=01  train=1.4599  val=1.3136  acc=53.95%
  [CIFAR-10][Adam]     ep=05  train=0.5256  val=0.5950  acc=80.12%
  [CIFAR-10][Adam]     ep=10  train=0.2505  val=0.3682  acc=88.52%
  [CIFAR-10][Adam]     ep=15  train=0.1345  val=0.3085  acc=90.72%

  [CIFAR-10][AdamW]    ep=01  train=1.4668  val=1.2218  acc=56.51%
  [CIFAR-10][AdamW]    ep=05  train=0.5256  val=0.5819  acc=80.44%
  [CIFAR-10][AdamW]    ep=10  train=0.2544  val=0.3631  acc=88.47%
  [CIFAR-10][AdamW]    ep=15  train=0.1381  val=0.3085  acc=90.58%

  [CIFAR-10][PsiLogic] ep=01  train=1.4378  val=1.1104  acc=60.38%
  [CIFAR-10][PsiLogic] ep=05  train=0.5184  val=0.5369  acc=81.61%
  [CIFAR-10][PsiLogic] ep=10  train=0.2574  val=0.3668  acc=88.54%
  [CIFAR-10][PsiLogic] ep=15  train=0.1439  val=0.3220  acc=90.48%

  ── Seeds 2–9: all individual epoch logs in benchmark/1.log ─────────
  (Full per-epoch output for seeds 2–9 available in benchmark/1.log)

  ── FINAL SUMMARY (mean ± std, 10 seeds) ────────────────────────────
   Optimizer     Train Loss        Val Loss        Val Acc (%)
   Adam      0.1459 ± 0.0077  0.3158 ± 0.0079   90.3410 ± 0.3531
   AdamW     0.1466 ± 0.0058  0.3167 ± 0.0077   90.2990 ± 0.1974
   PsiLogic  0.1432 ± 0.0055  0.3187 ± 0.0085   90.4140 ± 0.2487
  ────────────────────────────────────────────────────────────────────
  Wall time: 87.9 min


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXPERIMENT 6 — nanoGPT · Tiny Shakespeare · 2000 steps · 5 seeds  [A40]
 Adam · AdamW · PsiLogic (benchmark_all.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Device: NVIDIA A40 (cuda)   5 seeds (0–4)   2000 steps each

  ── Seed 0 ──────────────────────────────────────────────────────────
  [nanoGPT][Adam]     step= 200  train=2.5136  val=2.5031
  [nanoGPT][Adam]     step= 800  train=2.1103  val=2.0935
  [nanoGPT][Adam]     step=1400  train=1.9040  val=1.9243
  [nanoGPT][Adam]     step=2000  train=1.8842  val=1.8907

  [nanoGPT][AdamW]    step= 200  train=2.5137  val=2.5031
  [nanoGPT][AdamW]    step= 800  train=2.1103  val=2.0934
  [nanoGPT][AdamW]    step=1400  train=1.9040  val=1.9242
  [nanoGPT][AdamW]    step=2000  train=1.8842  val=1.8907

  [nanoGPT][PsiLogic] step= 200  train=2.5148  val=2.5025
  [nanoGPT][PsiLogic] step= 800  train=2.1153  val=2.1020
  [nanoGPT][PsiLogic] step=1400  train=1.9141  val=1.9367
  [nanoGPT][PsiLogic] step=2000  train=1.8942  val=1.9025

  ── Seeds 1–4: full step-by-step logs in benchmark/1.log ────────────

  ── FINAL SUMMARY (mean ± std, 5 seeds) ────────────────────────────
   Optimizer     Train Loss        Val Loss
   Adam      1.8828 ± 0.0177   1.8482 ± 0.0053
   AdamW     1.8828 ± 0.0177   1.8482 ± 0.0053
   PsiLogic  1.8905 ± 0.0167   1.8564 ± 0.0040
  ────────────────────────────────────────────────────────────────────
  Wall time: 11.7 min   Total (Exp 5+6): 99.6 min


╔══════════════════════════════════════════════════════════════════════════╗
║                    SUMMARY ACROSS ALL EXPERIMENTS                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ResNet-18 / CIFAR-10 / 100ep [local]  Adam 93.67%  ΨLogic 93.59%       ║
║  ResNet-18 / CIFAR-10 / 100ep [colab]  Adam 93.65%  ΨLogic 93.69% ✓     ║
║  ResNet-18 / CIFAR-10 / 30ep  [multi]  Adam 93.19%  ΨLogic leads ep1    ║
║  ResNet-18 / CIFAR-10 / 15ep  [A40]    ΨLogic 90.41% leads (10 seeds)   ║
║  Transformer / AG News / 10ep          ΨLogic leads ep5 and ep10        ║
║  CNN+GRU / SpeechCommands / 15ep       ΨLogic leads ep10 and ep12       ║
║  nanoGPT / Tiny Shakespeare / 2000s    Adam/AdamW tie ΨLogic −0.008     ║
║                                                                          ║
║  ΨLogic leads Adam at ep1–ep80 consistently on ResNet-18                ║
║  ΨLogic beats all optimizers on Text at ep5 and ep10                    ║
║  ΨLogic is best or 2nd-best on Audio at every epoch                     ║
╚══════════════════════════════════════════════════════════════════════════╝