import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from psilogic import PsiLogic
import copy
import math

def get_data():
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 2)
    return DataLoader(TensorDataset(x, y), batch_size=32)

def run_mirror_ablation():
    torch.manual_seed(42)
    
    # 1. Initialize models to exactly the same weights
    model_psi = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2)).cuda()
    model_adam = copy.deepcopy(model_psi)
    model_base = copy.deepcopy(model_psi)
    
    # 2. Train PsiLogic
    # Disable GC and AGC to ensure a fair comparison with standard AdamW
    opt_psi = PsiLogic(
        model_psi.parameters(), 
        lr=1e-3, 
        chaos_warmup=0, 
        chaos_tau=0.01,
        grad_centralize=False,
        agc_clip=0.0,
        use_foreach=False # Use scalar to easily track EMA state
    )
    crit = nn.MSELoss()
    dl = get_data()
    
    chaos_logs = []
    
    print("Training PsiLogic...")
    psi_losses = []
    for ep in range(10):
        for bx, by in dl:
            bx, by = bx.cuda(), by.cuda()
            opt_psi.zero_grad()
            loss = crit(model_psi(bx), by)
            loss.backward()
            opt_psi.step()
            
            psi_losses.append(loss.item())
            
            # Compute average chaos across all params
            chaos_sum = 0
            count = 0
            for group in opt_psi.param_groups:
                gamma = group["gamma"]
                p_ext = group["p_ext"]
                for p in group["params"]:
                    st = opt_psi.state[p]
                    slow_t = st["slow"].item()
                    fast_t = st["fast"].item()
                    
                    if fast_t > 2.0 * slow_t + 1e-8: # tau_scale default
                        ratio = fast_t / (slow_t + 1e-8)
                        chaos = math.tanh(slow_t) * (1.0 + 0.5 * math.tanh(max(ratio - 1.0, 0.0)))
                        cc = min(chaos * group["lr"] * gamma * p_ext, 0.05) # max_cancel
                        chaos_sum += cc
                    count += 1
            chaos_logs.append(chaos_sum / count)
            
    # 3. Train AdamW Mirror
    print("Training AdamW Mirror...")
    opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=1e-3, weight_decay=1e-4)
    
    adam_losses = []
    step = 0
    for ep in range(10):
        for bx, by in dl:
            bx, by = bx.cuda(), by.cuda()
            
            # Apply dynamic WD schedule
            base_wd = 1e-4
            current_chaos_wd = chaos_logs[step] / opt_adam.param_groups[0]["lr"]
            opt_adam.param_groups[0]["weight_decay"] = base_wd + current_chaos_wd
            
            opt_adam.zero_grad()
            loss = crit(model_adam(bx), by)
            loss.backward()
            opt_adam.step()
            
            adam_losses.append(loss.item())
            step += 1
            
    # 4. Train Standard AdamW Baseline
    print("Training AdamW Baseline...")
    opt_base = torch.optim.AdamW(model_base.parameters(), lr=1e-3, weight_decay=1e-4)
    
    base_losses = []
    for ep in range(10):
        for bx, by in dl:
            bx, by = bx.cuda(), by.cuda()
            opt_base.zero_grad()
            loss = crit(model_base(bx), by)
            loss.backward()
            opt_base.step()
            base_losses.append(loss.item())
            
    print("-" * 40)
    print(f"Final PsiLogic Loss:        {psi_losses[-1]:.4f}")
    print(f"Final AdamW Mirror Loss:    {adam_losses[-1]:.4f}")
    print(f"Final AdamW Baseline Loss:  {base_losses[-1]:.4f}")
    
    # Are the losses nearly identical between PsiLogic and AdamW Mirror?
    diff = abs(psi_losses[-1] - adam_losses[-1])
    print(f"Loss difference (Psi vs Mirror): {diff:.6f}")
    
    if diff < 1e-3:
        print("\nCONCLUSION: PsiLogic is empirically equivalent to an automatically generated AdamW weight-decay schedule.")
    else:
        print("\nCONCLUSION: PsiLogic has effects beyond simple uniform weight-decay scheduling.")

if __name__ == "__main__":
    run_mirror_ablation()
