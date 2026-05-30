import torch
import torch.nn as nn
from psilogic import PsiLogic
import copy

def test_amp_robustness():
    torch.manual_seed(42)
    model_fp32 = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)).cuda()
    model_amp = copy.deepcopy(model_fp32)
    
    # We set chaos_tau=0.01 to ensure chaos fires
    opt_fp32 = PsiLogic(model_fp32.parameters(), lr=1e-3, chaos_warmup=0, chaos_tau=0.01)
    opt_amp = PsiLogic(model_amp.parameters(), lr=1e-3, chaos_warmup=0, chaos_tau=0.01)
    
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    x = torch.randn(4, 10).cuda()
    y = torch.randn(4, 2).cuda()
    
    for _ in range(5):
        # FP32 Step
        opt_fp32.zero_grad()
        loss_fp32 = criterion(model_fp32(x), y)
        loss_fp32.backward()
        opt_fp32.step()
        
        # AMP Step
        opt_amp.zero_grad()
        with torch.cuda.amp.autocast():
            loss_amp = criterion(model_amp(x), y)
        
        scaler.scale(loss_amp).backward()
        # Ensure gradients are unscaled BEFORE optimizer.step() reads them.
        scaler.unscale_(opt_amp)
        opt_amp.step()
        
        # We manually update scaler to mock full loop
        scaler.update()

    # Compare weights.
    for p_fp32, p_amp in zip(model_fp32.parameters(), model_amp.parameters()):
        assert torch.allclose(p_fp32, p_amp, atol=1e-4), \
            "PsiLogic trajectories diverge under AMP due to scale-dependence"

