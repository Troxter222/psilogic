import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from psilogic import PsiLogic


def get_data():
    torch.manual_seed(123)
    x = torch.randn(2000, 10)
    y = torch.randint(0, 2, (2000,))
    return DataLoader(TensorDataset(x, y), batch_size=64)

def run_gc_agc_ablation():
    print("Running GC & AGC Ablation Study...")

    # We will simulate a small challenging network where spikes occur
    base_model = nn.Sequential(
        nn.Linear(10, 100), nn.GELU(),
        nn.Linear(100, 100), nn.GELU(),
        nn.Linear(100, 2)
    ).cuda()

    crit = nn.CrossEntropyLoss()
    dl = get_data()

    def train_config(name, get_opt_fn):
        torch.manual_seed(42)
        model = copy.deepcopy(base_model)
        opt = get_opt_fn(model)

        losses = []
        for _ in range(5):
            for bx, by in dl:
                bx, by = bx.cuda(), by.cuda()
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
                losses.append(loss.item())

        final_loss = sum(losses[-10:]) / 10.0
        print(f"[{name:<20}] Final Avg Loss: {final_loss:.4f}")
        return final_loss

    # Configurations
    train_config(
        "PsiLogic (Full)",
        lambda m: PsiLogic(m.parameters(), lr=1e-3, chaos_warmup=0)
    )
    train_config(
        "PsiLogic (No GC)",
        lambda m: PsiLogic(m.parameters(), lr=1e-3, chaos_warmup=0, grad_centralize=False)
    )
    train_config(
        "PsiLogic (No AGC)",
        lambda m: PsiLogic(m.parameters(), lr=1e-3, chaos_warmup=0, agc_clip=0.0)
    )
    train_config(
        "AdamW Baseline",
        lambda m: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    )

if __name__ == "__main__":
    run_gc_agc_ablation()
