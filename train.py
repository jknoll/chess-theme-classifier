# Basic training loop insired by https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# This script trains locally on a single GPU.
# This version has a correction to ensure that it is training on the GPU; previously I was not correctly calling model.to(device), inputs.to(device), and labels.to(device).

from cycling_utils import TimestampedTimer
import time
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from metrics import jaccard_similarity
import wandb

timer = TimestampedTimer("Imported TimestampedTimer")

from dataset import ChessPuzzleDataset
import torch
from torch.utils.data import DataLoader, random_split
from model import Model

# Create logs and checkpoints directories if they don't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Initialize distributed process group
def init_distributed(distributed=None):
    """
    Initialize distributed process group based on environment variables or parameter.
    
    Args:
        distributed: If explicitly provided as True/False, will override detection.
                    If None, will auto-detect based on environment variables.
    
    Returns:
        local_rank: The local rank of the process (0 if running in non-distributed mode)
        is_distributed: Boolean indicating if running in distributed mode
    """
    # Check if we're running with torchrun based on environment variables
    running_distributed = (
        'LOCAL_RANK' in os.environ and 
        'RANK' in os.environ and 
        'WORLD_SIZE' in os.environ and 
        int(os.environ.get('WORLD_SIZE', '1')) > 1
    )
    
    # Override detection if explicitly specified
    if distributed is not None:
        running_distributed = distributed

    # Initialize for single-process mode
    if not running_distributed:
        local_rank = 0
        return local_rank, False
    
    # Running in distributed mode - set up process group
    # Set default environment variables for distributed training if not already set
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
    
    # Check for GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires GPUs, but none were found.")
    
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, True

def parse_args():
    parser = argparse.ArgumentParser(description='Train chess puzzle classifier')
    parser.add_argument('--checkpoint_steps', type=int, default=50000,
                        help='Number of steps between checkpoints (default: 50000)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with a smaller dataset')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='chess-theme-classifier',
                        help='Weights & Biases project name')
    parser.add_argument('--name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--distributed', action='store_true',
                        help='Force distributed training mode (overrides auto-detection)')
    parser.add_argument('--local', action='store_true',
                        help='Force local training mode (overrides auto-detection)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Determine distributed mode based on command line args
    distributed_override = None
    if args.distributed and args.local:
        print("Warning: Both --distributed and --local flags set. Using --distributed.")
        distributed_override = True
    elif args.distributed:
        distributed_override = True
    elif args.local:
        distributed_override = False
    
    # Initialize distributed or local training
    local_rank, is_distributed = init_distributed(distributed_override)
    
    # Set device
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device} with local_rank: {local_rank} (Running in {'distributed' if is_distributed else 'local'} mode)")
    torch.autograd.set_detect_anomaly(True)

    # Create a timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('logs', f'run_{timestamp}')
    
    # Initialize wandb on the main process if enabled
    # The WANDB_API_KEY environment variable should be set before running
    # or use `wandb login` command beforehand
    if local_rank == 0 and args.wandb:
        run_name = args.name if args.name else f"run_{timestamp}"
        wandb.init(
            project=args.project,
            name=run_name,
            config={
                "optimizer": "Lamb",
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "architecture": "CNN with attention and residual blocks",
                "dataset": "lichess_db_puzzle.csv" if not args.test_mode else "lichess_db_puzzle_test.csv",
                "epochs": 3 if args.test_mode else 10,
                "batch_size": 4,
                "test_mode": args.test_mode,
            }
        )
        print(f"Initialized Weights & Biases for project {args.project}, run {run_name}")
    
    # Only create writer on main process
    writer = SummaryWriter(log_dir) if local_rank == 0 else None
    if local_rank == 0:
        timer.report(f"Created TensorBoard writer at {log_dir}")

    # Initialize starting epoch and global step
    start_epoch = 0
    global_step = 0

    # Use a smaller dataset if in test mode
    if args.test_mode:
        dataset = ChessPuzzleDataset('lichess_db_puzzle_test.csv')
        if local_rank == 0:
            print("Running in test mode with smaller dataset (1000 samples)")
    else:
        dataset = ChessPuzzleDataset('lichess_db_puzzle.csv')
    
    # Get the number of labels from the dataset
    num_labels = len(dataset.all_labels)
    if local_rank == 0:
        print(f"Number of unique labels (themes + opening tags): {num_labels}")
    
    # Create model with the correct number of labels
    # Original hacakthon-winner-3 model config with default nlayers=2.
    # model = Model(num_labels=num_labels)
    model = Model(num_labels=num_labels, nlayers=5, embed_dim=64, inner_dim=320, attention_dim=64, use_1x1conv=True, dropout=0.5)
    model = model.to(device)
    
    # Wrap model in DDP if running in distributed mode
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    # In local mode, we can use DataParallel if multiple GPUs are available
    elif torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    
    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Use Lamb optimizer with settings from chess-hackathon
    # Instead of Adam as was used previously
    lr = 0.001
    weight_decay = 0.01
    warmup_steps = 1000
    
    if local_rank == 0:
        print(f"Using Lamb optimizer with lr={lr}, weight_decay={weight_decay}, warmup_steps={warmup_steps}")
    
    from model import Lamb, get_lr_with_warmup
    optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Get current learning rate (for logging)
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    # Check for checkpoint file
    checkpoint_path = "checkpoints/checkpoint_resume.pth"
    if os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Load state dict properly depending on model type (DDP, DataParallel, or regular)
        if is_distributed or isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint.get('global_step', 0)  # Get global_step if exists, else 0
        
        # Calculate if we completed the previous epoch
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
            print(f"  Jaccard Loss: {checkpoint['jaccard_loss']:.8f}")
            print(f"  Learning Rate: {checkpoint.get('learning_rate', get_lr(optimizer)):.8f}")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Completed epochs: {completed_epochs}")
            print(f"  Resuming from epoch: {start_epoch}")

    # Split the dataset into train and test sets
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)
    if local_rank == 0:
        timer.report(f"Initialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test board evaluations.")

    # Create samplers based on distributed mode
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        
        # Use distributed samplers with DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=4, sampler=test_sampler)
    else:
        # Use standard samplers for local training
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        # Define dummy sampler for local mode so we can call set_epoch without errors
        train_sampler = type('DummySampler', (), {'set_epoch': lambda self, epoch: None})()
    if local_rank == 0:
        timer.report("Prepared dataloaders")

    # Use fewer epochs in test mode
    max_epochs = 3 if args.test_mode else 10
    for epoch in range(start_epoch, max_epochs):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        for i, data in enumerate(train_dataloader, 0):
            running_loss = 0.0
            running_jaccard_index = 0.0
            
            inputs = data['board']
            inputs = inputs.unsqueeze(1).to(device)
            labels = data['themes'].to(device)

            optimizer.zero_grad()

            # Apply learning rate warmup
            # Calculate total steps since the start of training
            total_step = global_step
            # Apply learning rate scheduler function
            lr_factor = min(total_step / warmup_steps, 1.0)
            current_lr = lr_factor * lr
            # Update learning rate
            for g in optimizer.param_groups:
                g['lr'] = current_lr

            # Use debug mode for the first batch of the first epoch
            debug_mode = (epoch == start_epoch and i == 0)
            outputs = model(inputs, debug=debug_mode)
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
                current_lr = get_lr(optimizer)
                print(f"epoch: {epoch}/{max_epochs-1} step: {i+1} lr: {current_lr:.8f} loss: {running_loss:.8f} jaccard index: {running_jaccard_index:.8f}")
                
                if writer is not None:
                    writer.add_scalar('Loss/train', running_loss, global_step)
                    writer.add_scalar('Jaccard/train', running_jaccard_index, global_step)
                    writer.add_scalar('Learning_rate', current_lr, global_step)
                
                # Log to wandb if enabled
                if args.wandb:
                    wandb.log({
                        "loss": running_loss,
                        "jaccard_index": running_jaccard_index,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "global_step": global_step
                    })
                
                if i % args.checkpoint_steps == 0 and i > 0:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    checkpoint_filename = f"checkpoints/checkpoint_{timestamp}_{i}.pth"
                    # Save the model state dict properly depending on model type
                    if is_distributed or isinstance(model, torch.nn.DataParallel):
                        model_state = model.module.state_dict()
                    else:
                        model_state = model.state_dict()
                        
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'global_step': global_step,
                        'jaccard_loss': jaccard_loss,
                        'learning_rate': current_lr,
                    }, checkpoint_filename)
                    
                    # Also save a copy as the latest checkpoint for easy resuming
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'global_step': global_step,
                        'jaccard_loss': jaccard_loss,
                        'learning_rate': current_lr,
                    }, "checkpoints/checkpoint_resume.pth")
                    
                    if writer is not None:
                        writer.add_text('Checkpoint', f'Saved checkpoint: {checkpoint_filename}', global_step)
                        print(f"Checkpoint saved at step {i} in {checkpoint_filename}")
            
            global_step += 1
    
    # Clean up
    if writer is not None:
        writer.close()
        timer.report("Closed TensorBoard writer")
    
    # Close wandb run if it was used
    if local_rank == 0 and args.wandb:
        wandb.finish()
        print("Closed Weights & Biases run")
    
    # Clean up distributed process group if we were using it
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()