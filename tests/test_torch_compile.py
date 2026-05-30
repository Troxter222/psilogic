import torch
import torch.nn as nn
from psilogic import PsiLogic

def test_torch_compile_fullgraph():
    # Only test if on a supported platform/version
    if not hasattr(torch, "compile"):
        return
        
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )
    if torch.cuda.is_available():
        model = model.cuda()
        
    optimizer = PsiLogic(model.parameters(), lr=1e-3, use_foreach=True)
    criterion = nn.MSELoss()
    
    def train_step(x, y):
        # We only compile the optimizer step for now to bypass zero_grad tracing issues
        loss = criterion(model(x), y)
        loss.backward()
        return loss

    def opt_step():
        optimizer.step()

    compiled_opt_step = torch.compile(opt_step, fullgraph=True)
    
    x = torch.randn(4, 10)
    y = torch.randn(4, 2)
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()
        
    try:
        # Run forward and backward to get gradients
        optimizer.zero_grad()
        train_step(x, y)
        compiled_opt_step()
    except Exception as e:
        import traceback
        traceback.print_exc()
        assert False, f"torch.compile(fullgraph=True) failed: {e}"
        
    # Run a few more steps
    for _ in range(3):
        optimizer.zero_grad()
        train_step(x, y)
        compiled_opt_step()

