from step2_comms import init_distributed, PipelineComms
import torch


def ping_pong():
    """
    Send a tensor from device rank 0 to device rank 1 and print to verify.
    """
    rank, world_size, device = init_distributed()
    
    print(rank,world_size, device)
    communication = PipelineComms(rank,world_size)
    if rank == 0:
        torch.distributed.barrier()
        tensor = torch.rand(3)
        communication.send_forward(tensor)
    if rank == 1:
        torch.distributed.barrier()
        tensor = communication.recv_forward(3,device)
        print(f"recieved gensor:{tensor}")

if __name__ == "__main__":
    ping_pong()
