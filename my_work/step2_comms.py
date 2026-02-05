import os

import torch
import torch.distributed as dist


def init_distributed():
    """
    Initializes the distributed process group.
    Reads state directly from environment variables set by torchrun.
    """
    # 1. Read Environment Variables (set by torchrun)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # 2. Set Device
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu") 
    # 3. Initialize Group
    
    if torch.cuda.is_available():
        dist.init_process_group('nccl',rank,world_size) 
    else:
        dist.init_process_group(backend='gloo',rank=rank,world_size=world_size)

    return rank, world_size, device


class PipelineComms:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.prev_rank = self.rank - 1 if self.rank != 0 else None
        # self.last = self.rank==self.world_size - 1
        self.next_rank = self.rank + 1 if self.rank != self.world_size - 1 else None

    def send_forward(self, tensor):
        """Send activation to the next GPU."""
        dist.send(tensor, self.next_rank)

    def recv_forward(self, shape, device, dtype=torch.float32):
        """Receive activation from the previous GPU."""
        tensor = torch.zeros(shape,device = device,dtype=dtype)
        dist.recv(tensor,self.prev_rank)
        return tensor
        

    def send_backward(self, tensor):
        """Send gradients back to the previous GPU."""
        dist.send(tensor,self.prev_rank)

    def recv_backward(self, shape, device, dtype=torch.float32):
        """Receive gradients from the next GPU."""
        tensor = torch.zeros(shape,device = device,dtype=dtype)
        dist.recv(tensor,self.next_rank)
        return tensor
