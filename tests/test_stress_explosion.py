import torch
import torch.nn as nn
from psilogic import PsiLogic

def test_gradient_explosion():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2)).cuda()
    
    # We set a normal max_cancel bound. 
    # Even if chaos detects an extreme spike, it should not shrink weights more than 5% per step.
    opt = PsiLogic(model.parameters(), lr=1e-3, chaos_warmup=0, chaos_tau=0.01, max_cancel=0.05)
    crit = nn.CrossEntropyLoss()
    
    x = torch.randn(4, 10).cuda()
    y = torch.randint(0, 2, (4,)).cuda()
    
    # Train normal
    for _ in range(5):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        
    pre_spike_norm = list(model.parameters())[0].norm().item()
    
    # Simulate Spike
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    
    # Artificially explode gradients by 1000x
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(1000.0)
            
    opt.step()
    
    post_spike_norm = list(model.parameters())[0].norm().item()
    
    # Check that weights did not collapse to 0 or explode to NaN/inf
    assert not torch.isnan(list(model.parameters())[0]).any(), "Weights became NaN after explosion!"
    assert not torch.isinf(list(model.parameters())[0]).any(), "Weights became Inf after explosion!"
    
    # Check shrinkage
    shrinkage = pre_spike_norm / post_spike_norm
    print(f"Norm before spike: {pre_spike_norm:.4f}")
    print(f"Norm after spike:  {post_spike_norm:.4f}")
    print(f"Shrinkage factor:  {shrinkage:.4f}")
    
    # It should shrink by roughly up to max_cancel (1.0 / 0.95 = 1.05) but definitely not collapse
    assert shrinkage < 1.10, f"Weights collapsed too much! Shrinkage: {shrinkage}"

