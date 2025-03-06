# Basic training loop insired by https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# This script trains locally on a single GPU.
# This version has a correction to ensure that it is training on the GPU; previously I was not correctly calling model.to(device), inputs.to(device), and labels.to(device).

from cycling_utils import TimestampedTimer
import time
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

timer = TimestampedTimer("Imported TimestampedTimer")

from dataset import ChessPuzzleDataset
import torch
from torch.utils.data import DataLoader, random_split
from model import Model

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def jaccard_similarity(tensor1, tensor2):
    """
    Compute Jaccard similarity between two binary tensors.
    
    Args:
        tensor1: Binary PyTorch tensor
        tensor2: Binary PyTorch tensor of the same shape as tensor1
    
    Returns:
        Jaccard similarity score (float between 0 and 1)
    """
    intersection = torch.logical_and(tensor1, tensor2).sum()
    union = torch.logical_or(tensor1, tensor2).sum()
    
    return intersection.float() / (union.float() + 1e-8)  # add small epsilon to avoid division by zero

# Initialize distributed process group
def init_distributed():
    # Set default environment variables for single GPU training
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    # Initialize distributed training
    local_rank = init_distributed()
    device = torch.device(f'cuda:{local_rank}')
    print(f"Using device: {device} with local_rank: {local_rank}")
    torch.autograd.set_detect_anomaly(True)

    # Create a timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', f'run_{timestamp}')
    # Only create writer on main process
    writer = SummaryWriter(log_dir) if local_rank == 0 else None
    if local_rank == 0:
        timer.report(f"Created TensorBoard writer at {log_dir}")

    model = Model()
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    dataset = ChessPuzzleDataset('lichess_db_puzzle.csv')

    # Split the dataset into train and test sets
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)
    if local_rank == 0:
        timer.report(f"Initialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test board evaluations.")

    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    # Use distributed samplers with DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=4, sampler=test_sampler)
    if local_rank == 0:
        timer.report("Prepared dataloaders")

    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Track global step for TensorBoard
    global_step = 0

    for epoch in range(10):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        for i, data in enumerate(train_dataloader, 0):
            running_loss = 0.0
            running_jaccard_index = 0.0
            
            inputs = data['board']
            inputs = inputs.unsqueeze(1).to(device)
            labels = data['themes'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Apply sigmoid to get probabilities for Jaccard calculation
            output_probs = torch.sigmoid(outputs)
            jaccard_loss = jaccard_similarity(output_probs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item()
            running_jaccard_index = jaccard_loss.item()
            
            if local_rank == 0:  # Only log on main process
                print(f"epoch: {epoch} loss: {running_loss:.8f} jaccard index: {running_jaccard_index:.8f} steps: {i+1}")
                
                if writer is not None:
                    writer.add_scalar('Loss/train', running_loss, global_step)
                    writer.add_scalar('Jaccard/train', running_jaccard_index, global_step)
                
                if i % 100000 == 0:
                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    checkpoint_filename = f"checkpoint_{timestamp}_{i}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),  # Save the inner model's state
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, checkpoint_filename)
                    
                    if writer is not None:
                        writer.add_text('Checkpoint', f'Saved checkpoint: {checkpoint_filename}', global_step)
            
            global_step += 1
    
    # Clean up
    if writer is not None:
        writer.close()
        timer.report("Closed TensorBoard writer")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()