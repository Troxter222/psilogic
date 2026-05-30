import torch
import torch.nn as nn
from psilogic import PsiLogic
import copy

def test_scale_invariance():
    torch.manual_seed(42)
    model_normal = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)).cuda()
    model_scaled = copy.deepcopy(model_normal)
    
    # Use standard learning rate, enable chaos, turn off AGC to avoid clamping
    opt_normal = PsiLogic(model_normal.parameters(), lr=1e-3, chaos_warmup=0, chaos_tau=0.01, agc_clip=0.0)
    opt_scaled = PsiLogic(model_scaled.parameters(), lr=1e-3, chaos_warmup=0, chaos_tau=0.01, agc_clip=0.0)
    
    criterion = nn.MSELoss()
    
    x = torch.randn(4, 10).cuda()
    y = torch.randn(4, 2).cuda()
    
    SCALE = 100.0
    
    # We want to see how `chaos_contrib` behaves under scaled gradients.
    
    for _ in range(5):
        # Normal Step
        opt_normal.zero_grad()
        loss_normal = criterion(model_normal(x), y)
        loss_normal.backward()
        opt_normal.step()
        
        # Scaled Step
        opt_scaled.zero_grad()
        loss_scaled = criterion(model_scaled(x), y) * SCALE
        loss_scaled.backward()
        opt_scaled.step()
    
    p_normal = list(model_normal.parameters())[0]
    p_scaled = list(model_scaled.parameters())[0]
    
    state_normal = opt_normal.state[p_normal]
    state_scaled = opt_scaled.state[p_scaled]
    
    print(f"Normal slow: {state_normal['slow'].item():.6f}")
    print(f"Scaled slow: {state_scaled['slow'].item():.6f}")
    
    # Assert that the normalized "slow" states match perfectly
    assert torch.allclose(state_normal["slow"], state_scaled["slow"], atol=1e-4), \
        "Slow EMA should be scale-invariant"
    
    # Assert that the normalized "fast" states match perfectly
    assert torch.allclose(state_normal["fast"], state_scaled["fast"], atol=1e-4), \
        "Fast EMA should be scale-invariant"

