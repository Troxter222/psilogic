"""
Benchmark suite for testing PsiLogic variants on NLP (Small GPT / TinyStories)
over 5000 steps across seeds {0, 1, 2}.
"""

import argparse
import copy
import math
import os
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Импорт компонентов PsiLogic
import psilogic
from psilogic import PsiLogic, PsiLogicGPT
from psilogic.param_groups import gpt_param_groups

# =====================================================================
# 1. Архитектура Small GPT (FairBench-совместимая)
# =====================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, ctx_len: int = 256):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(ctx_len, ctx_len)).view(1, 1, ctx_len, ctx_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, ctx_len: int = 256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, ctx_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SmallGPT(nn.Module):
    def __init__(self, vocab_size: int = 10000, d_model: int = 384, n_layers: int = 6, n_heads: int = 6, ctx_len: int = 256):
        super().__init__()
        self.ctx_len = ctx_len
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(ctx_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, ctx_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.wte.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# =====================================================================
# 2. Датасет (TinyStories / Синтетический fallback при оффлайн запуске)
# =====================================================================

class SyntheticOrRealTinyStories(Dataset):
    def __init__(self, data_root: str, split: str = "train", ctx_len: int = 256, vocab_size: int = 10000):
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.data = None

        # Проверка наличия скачанных токенов TinyStories
        bin_path = os.path.join(data_root, f"tinystories_{split}.bin")
        if os.path.exists(bin_path):
            import numpy as np
            print(f"[{split}] Загрузка датасета из {bin_path}...")
            raw = np.memmap(bin_path, dtype=np.uint16, mode="r")
            self.data = torch.from_numpy(raw.astype(np.int64))
        else:
            print(f"[{split}] {bin_path} не найден! Генерируем воспроизводимый синтетический поток...")
            g = torch.Generator().manual_seed(42 if split == "train" else 1337)
            self.data = torch.randint(0, vocab_size, (500000,), generator=g, dtype=torch.long)

    def __len__(self):
        return (len(self.data) - self.ctx_len - 1) // self.ctx_len

    def __getitem__(self, idx):
        start = idx * self.ctx_len
        chunk = self.data[start : start + self.ctx_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# =====================================================================
# 3. Варианты PsiLogic и Модификации
# =====================================================================

class PsiLogicMomentumDamp(PsiLogic):
    """
    Вариант: Shock Absorber (Демпфирование импульса m_t вместо весов theta_t).
    Исключает неустранимый сдвиг B_delta на поздних этапах.
    """
    def _apply_unified_decay(self, param, raw_grad, *, lr, wd, gamma_eff, qd_eff, p_ext, max_cancel,
                             slow_t, fast_t, adaptive_tau, chaos_tau, tau_scale, eps, chaos_gain):
        # Применяем только стандартный AdamW weight decay на веса
        if wd > 0:
            param.mul_(1.0 - lr * wd)

    def _adam_or_lion_update(self, param, grad, state, *, lr, beta1, beta2, eps, step, lion):
        # Демпфируем накопленный импульс перед шагом
        if "fast" in state and "slow" in state:
            ratio = state["fast"] / (state["slow"] + eps)
            is_spike = (state["fast"] > 2.0 * state["slow"] + eps).float()
            chaos = torch.tanh(state["slow"]) * (1.0 + 0.5 * torch.tanh(torch.clamp(ratio - 1.0, min=0.0)))
            c_damp = torch.clamp(chaos * 0.05 * is_spike, max=0.5)
            state["m"].mul_(1.0 - c_damp)

        super()._adam_or_lion_update(param, grad, state, lr=lr, beta1=beta1, beta2=beta2, eps=eps, step=step, lion=lion)


def build_optimizer_variant(variant_name: str, model: nn.Module, lr: float, total_steps: int) -> torch.optim.Optimizer:
    """
    Фабрика создания вариантов PsiLogic для тестирования.
    """
    if variant_name == "0_baseline_5k":
        # Базовый PsiLogic с масштабированным шедулером на 5000 шагов
        return PsiLogicGPT(model.parameters(), lr=lr, gamma_T_max=total_steps)

    elif variant_name == "1_gamma_auto":
        # Идея 1: Автоматическое затухание gamma при стабилизации slow EMA
        return PsiLogicGPT(model.parameters(), lr=lr, gamma_T_max=total_steps, gamma_auto=True)

    elif variant_name == "2_embed_shield":
        # Идея 2: Защита эмбеддингов (пониженная gamma=0.001 на wte/lm_head)
        groups = gpt_param_groups(
            model,
            lr=lr,
            embedding_gamma=0.001,
            block_gamma=0.02,
            head_gamma=0.005,
            weight_decay=0.1,
        )
        return PsiLogic(groups, lr=lr, gamma_T_max=total_steps, chaos_warmup=-1)

    elif variant_name == "3_stricter_tau":
        # Идея 3: Более строгий детектор (tau_scale=3.5) против ложных спайков
        return PsiLogicGPT(model.parameters(), lr=lr, gamma_T_max=total_steps, tau_scale=3.5)

    elif variant_name == "4_momentum_friction":
        # Идея 4: Гашение импульса m_t вместо сжатия параметров theta_t
        groups = gpt_param_groups(model, lr=lr, weight_decay=0.1)
        return PsiLogicMomentumDamp(groups, lr=lr, gamma_T_max=total_steps)

    elif variant_name == "5_synergy_combo":
        # Комбинация лучших решений: Shielding + Stricter Tau + Auto-Gamma
        groups = gpt_param_groups(
            model,
            lr=lr,
            embedding_gamma=0.0005,
            block_gamma=0.02,
            head_gamma=0.002,
            weight_decay=0.1,
        )
        return PsiLogic(
            groups,
            lr=lr,
            gamma_T_max=total_steps,
            tau_scale=3.5,
            gamma_auto=True,
            chaos_warmup=-1
        )
    else:
        raise ValueError(f"Неизвестный вариант: {variant_name}")

# =====================================================================
# 4. Цикл обучения и валидации
# =====================================================================

def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float = 1e-6) -> float:
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step > total_steps:
        return min_lr
    ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


def evaluate(model: nn.Module, val_loader: DataLoader, device: str, max_batches: int = 50) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                _, loss = model(x, y)
            total_loss += loss.item()
            count += 1
    val_loss = total_loss / max(1, count)
    ppl = math.exp(min(val_loss, 20.0))  # ограничение от переполнения
    return val_loss, ppl


def run_experiment(variant_name: str, seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    # Фиксация сидов
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Датасеты
    train_ds = SyntheticOrRealTinyStories(args.data_root, "train", ctx_len=args.ctx_len)
    val_ds = SyntheticOrRealTinyStories(args.data_root, "val", ctx_len=args.ctx_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Инициализация модели
    model = SmallGPT(ctx_len=args.ctx_len).to(device)
    optimizer = build_optimizer_variant(variant_name, model, args.lr, args.steps)
    
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    t_start = time.time()
    model.train()
    step = 0
    train_iter = iter(train_loader)

    print(f"\n🚀 Запуск [{variant_name}] | Seed: {seed} | Steps: {args.steps}")

    while step < args.steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        step += 1
        x, y = x.to(device), y.to(device)

        # Cosine LR
        curr_lr = get_cosine_lr(step, args.steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 1000 == 0 or step == args.steps:
            val_loss, ppl = evaluate(model, val_loader, device, max_batches=20)
            print(f"  Step {step:4d}/{args.steps} | Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | LR: {curr_lr:.2e}")
            model.train()

    wall_time = time.time() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    final_val_loss, final_ppl = evaluate(model, val_loader, device, max_batches=100)

    return {
        "variant": variant_name,
        "seed": seed,
        "final_val_loss": final_val_loss,
        "final_perplexity": final_ppl,
        "peak_vram_mb": peak_vram,
        "wall_time_s": wall_time,
    }

# =====================================================================
# 5. Главный запуск и аналитика
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="PsiLogic 5000-step NLP Variants Benchmark")
    parser.add_argument("--data-root", type=str, default="./data", help="Путь к данным TinyStories")
    parser.add_argument("--steps", type=int, default=5000, help="Количество шагов обучения")
    parser.add_argument("--lr", type=float, default=3.1622776601683794e-4, help="Learning rate (как в FairBench)")
    parser.add_argument("--batch-size", type=int, default=64, help="Размер батча")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Шаги LR warmup")
    parser.add_argument("--ctx-len", type=int, default=256, help="Длина контекста")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Список сидов")
    args = parser.parse_args()

    variants = [
        "0_baseline_5k",        # Текущий дефолт на 5000 шагов
        "1_gamma_auto",          # Авто-затухание гаммы
        "2_embed_shield",        # Защита матриц эмбеддингов
        "3_stricter_tau",        # Повышенный порог спайков
        "4_momentum_friction",   # Гашение моментума вместо весов
        "5_synergy_combo",       # Лучшая комбинация всех мер
    ]

    all_results = []

    for var in variants:
        for seed in args.seeds:
            res = run_experiment(var, seed, args)
            all_results.append(res)

    # Вывод агрегированных результатов
    print("\n" + "=" * 80)
    print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ (NLP Small GPT @ {args.steps} steps, Mean ± Std по 3 сидам)")
    print("=" * 80)
    print(f"{'Вариант':<25} | {'Val Perplexity ↓':<18} | {'Val Loss ↓':<16} | {'Time (s)':<10}")
    print("-" * 80)

    for var in variants:
        var_res = [r for r in all_results if r["variant"] == var]
        ppls = [r["final_perplexity"] for r in var_res]
        losses = [r["final_val_loss"] for r in var_res]
        times = [r["wall_time_s"] for r in var_res]

        m_ppl, s_ppl = (sum(ppls)/len(ppls)), (torch.tensor(ppls).std().item() if len(ppls) > 1 else 0.0)
        m_loss, s_loss = (sum(losses)/len(losses)), (torch.tensor(losses).std().item() if len(losses) > 1 else 0.0)
        m_time = sum(times)/len(times)

        print(f"{var:<25} | {m_ppl:6.2f} ± {s_ppl:<8.2f} | {m_loss:6.4f} ± {s_loss:<6.4f} | {m_time:6.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()