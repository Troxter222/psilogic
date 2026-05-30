import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from psilogic import PsiLogic

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_fsdp(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(42 + rank)
    torch.cuda.set_device(rank)
    
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 2)).cuda(rank)
    fsdp_model = FSDP(model, device_id=rank)
        
    loss_fn = nn.MSELoss()
    optimizer = PsiLogic(fsdp_model.parameters(), lr=1e-3, chaos_warmup=0, use_foreach=True)

    for i in range(3):
        optimizer.zero_grad()
        outputs = fsdp_model(torch.randn(4, 10).cuda(rank))
        labels = torch.randn(4, 2).cuda(rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    cleanup()

def test_fsdp():
    # Only run if we have at least 2 GPUs
    if torch.cuda.device_count() >= 2:
        world_size = 2
        mp.spawn(demo_fsdp,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        print("Skipping FSDP test because < 2 GPUs available.")

