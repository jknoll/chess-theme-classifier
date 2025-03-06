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
from metrics import jaccard_similarity

timer = TimestampedTimer("Imported TimestampedTimer")

from dataset import ChessPuzzleDataset
import torch
from torch.utils.data import DataLoader, random_split
from model import Model

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

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

    # Define the loss function and optimizer before loading checkpoint
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize starting epoch and global step
    start_epoch = 0
    global_step = 0

    # Check for checkpoint file
    checkpoint_path = "checkpoint_resume.pth"
    if os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint.get('global_step', 0)  # Get global_step if exists, else 0
        
        # Calculate if we completed the previous epoch
        dataset = ChessPuzzleDataset('lichess_db_puzzle.csv')
        train_size = int(0.8 * len(dataset))  # Same split as in training
        steps_per_epoch = train_size // 4  # batch_size=4
        completed_epochs = global_step // steps_per_epoch
        
        start_epoch = checkpoint['epoch']
        if global_step >= (start_epoch + 1) * steps_per_epoch:
            # We completed this epoch, start the next one
            start_epoch += 1
        
        if local_rank == 0:
            print(f"Resumed from checkpoint:")
            print(f"  Epoch: {checkpoint['epoch']}")
            print(f"  Global Step: {global_step}")
            print(f"  Loss: {checkpoint['loss']:.8f}")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Completed epochs: {completed_epochs}")
            print(f"  Resuming from epoch: {start_epoch}")

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

    # Track global step for TensorBoard
    global_step = 0

    for epoch in range(start_epoch, 10):
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
            jaccard_loss = jaccard_similarity(output_probs, labels, threshold=0.5)

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
                
                if i % 100000 == 0 and i > 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    checkpoint_filename = f"checkpoint_{timestamp}_{i}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),  # Save the inner model's state
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'global_step': global_step,
                        'jaccard_loss': jaccard_loss,
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