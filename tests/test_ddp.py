import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from psilogic import PsiLogic

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size) # Gloo is CPU safe for testing

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(42 + rank)
    
    # create model and move it to CPU for testing DDP safely without Multi-GPU
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2))
    ddp_model = nn.parallel.DistributedDataParallel(model)

    loss_fn = nn.MSELoss()
    optimizer = PsiLogic(ddp_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=False)

    for i in range(3):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(4, 10))
        labels = torch.randn(4, 2)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    cleanup()

def test_ddp():
    world_size = 2
    mp.spawn(demo_basic,
             args=(world_size,),
             nprocs=world_size,
             join=True)

