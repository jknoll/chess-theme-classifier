 # Basic training loop insired by https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# This script trains locally on a single GPU.
# This version has a correction to ensure that it is training on the GPU; previously I was not correctly calling model.to(device), inputs.to(device), and labels.to(device).

# Set OpenMP thread limit to avoid "Thread creation failed" error
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads

from cycling_utils import TimestampedTimer
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import subprocess
import webbrowser
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from metrics import jaccard_similarity, precision_recall_f1, get_classification_report
import wandb
import torch

timer = TimestampedTimer("Imported TimestampedTimer")

from dataset import ChessPuzzleDataset
import torch
from torch.utils.data import DataLoader, random_split
from model import Model
from class_weights import compute_label_weights

# Automatically launch tensorboard on port 6006 when training is started.
def launch_tensorboard(logdir, port=6006):
    # Only launch from rank 0 process
    if os.environ.get("LOCAL_RANK", "0") == "0":
        try:
            subprocess.Popen(
                ["tensorboard", f"--logdir={logdir}", f"--port={port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            webbrowser.open(f"http://localhost:{port}")
            print(f"TensorBoard launched at http://localhost:{port}, --logdir={logdir}")
        except Exception as e:
            print(f"Could not launch TensorBoard: {e}")

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
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Use weighted BCE loss for class imbalance')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Run in single-GPU mode (replaces train_locally_single_gpu.py)')
    parser.add_argument('--optimizer', type=str, default='lamb',
                        help='Optimizer to use: lamb or adam (default: lamb)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (default: 3 for test mode, 10 for full)')
    parser.add_argument('--low_memory', action='store_true',
                        help='Use lower memory settings for dataset processing')
    parser.add_argument('--dataset-id', type=str, default=None,
                        help='Dataset ID for ISC training (overrides environment variable)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Determine distributed mode based on command line args
    distributed_override = None
    if args.single_gpu:
        # Single GPU mode overrides other distributed settings
        print("Running in single GPU mode (equivalent to train_locally_single_gpu.py)")
        distributed_override = False
    elif args.distributed and args.local:
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
    
    # Add master flag similar to ISC training
    args.is_master = local_rank == 0
    
    if args.is_master:
        print(f"Using device: {device} with local_rank: {local_rank} (Running in {'distributed' if is_distributed else 'local'} mode)")
    torch.autograd.set_detect_anomaly(True)

    # Create a timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Check for ISC environment
    isc_mode = "LOSSY_ARTIFACT_PATH" in os.environ
    
    # Set log directory based on environment (local or ISC)
    if isc_mode and args.is_master:
        print(f"Running in ISC mode, using LOSSY_ARTIFACT_PATH for logging")
        log_dir = os.environ["LOSSY_ARTIFACT_PATH"]
    else:
        log_dir = os.path.join('logs', f'run_{timestamp}')
    
    # Set up dataset file choice first so we can use it in wandb config
    # Set dataset based on mode and environment
    # Get dataset ID from command line arg or environment variable
    dataset_id = args.dataset_id
    
    # If not provided as arg, check environment variable
    if not dataset_id and "DATASET_IDS" in os.environ:
        try:
            # Get the first dataset ID from the environment (comma-separated list)
            dataset_id = os.environ["DATASET_IDS"].split(",")[0]
            if args.is_master:
                print(f"Found dataset ID in environment: {dataset_id}")
        except Exception as e:
            if args.is_master:
                print(f"Error parsing dataset ID: {e}")
                
    # Select the appropriate dataset file
    if args.test_mode:
        csv_filename = 'lichess_db_puzzle_test.csv'
    elif args.single_gpu:
        csv_filename = 'lichess_db_puzzle_test.csv'
    else:
        csv_filename = 'lichess_db_puzzle.csv'
    
    # Determine the full path to the CSV file
    if isc_mode and dataset_id:
        # Use the ISC dataset path
        csv_file = f'/data/{dataset_id}/{csv_filename}'
        if args.is_master:
            print(f"Using ISC dataset path: {csv_file}")
    else:
        # Use the local path
        csv_file = csv_filename
    
    if args.is_master:
        if args.test_mode:
            print(f"Running in TEST MODE with smaller dataset")
        elif args.single_gpu:
            print(f"Running in SINGLE GPU MODE with test dataset")
        else:
            print(f"Running with FULL DATASET. This may require significant memory and processing time.")
            
    # Initialize wandb on the master process if enabled
    # The WANDB_API_KEY environment variable should be set before running
    # or use `wandb login` command beforehand
    if args.is_master and args.wandb:
        run_name = args.name if args.name else f"run_{timestamp}"
        wandb.init(
            project=args.project,
            name=run_name,
            config={
                "optimizer": args.optimizer,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "architecture": "CNN with attention and residual blocks",
                "dataset": csv_file,
                "epochs": max_epochs,
                "batch_size": batch_size,
                "test_mode": args.test_mode,
                "single_gpu_mode": args.single_gpu,
                "weighted_loss": args.weighted_loss,
                "class_conditional_augmentation": True,
            }
        )
        print(f"Initialized Weights & Biases for project {args.project}, run {run_name}")
    
    # Only create writer on master process
    writer = SummaryWriter(log_dir) if args.is_master else None
    if args.is_master:
        timer.report(f"Created TensorBoard writer at {log_dir}")
        # Only launch tensorboard in non-ISC mode
        if not isc_mode:
            launch_tensorboard('.') # Launch tensorboard from the parent logs directory to enable comparison of runs.

    # Initialize starting epoch and global step
    start_epoch = 0
    global_step = 0

    # csv_file is already set earlier for wandb config
    # Always use low_memory mode for class conditional augmentation to avoid memory issues
    low_memory = True
    
    if args.is_master:
        print(f"Creating dataset from {csv_file} with class_conditional_augmentation=True and low_memory={low_memory}")
        print(f"⚠️ Processing may take several minutes, especially for the full dataset")
    
    # Thread limit is already set at the top of the file
    # No need to set it again here
    
    # Create dataset with memory-saving options
    dataset = ChessPuzzleDataset(csv_file, class_conditional_augmentation=True, low_memory=low_memory)
    
    # Get the number of labels from the dataset
    num_labels = len(dataset.all_labels)
    if args.is_master:
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
    if args.is_master:
        print(f"\n{'='*60}")
        print(f"LOSS FUNCTION: {'Weighted BCE Loss' if args.weighted_loss else 'Standard BCE Loss (no class weights)'}")
        print(f"{'='*60}\n")
    
    if args.weighted_loss:
        if args.is_master:
            print("Computing class weights for weighted BCE loss...")
        # Compute label weights
        pos_weights = compute_label_weights(dataset)
        pos_weights = pos_weights.to(device)
        
        if args.is_master:
            print("Using weighted BCE loss for class imbalance mitigation")
        
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    else:
        if args.is_master:
            print("Using standard BCE loss without class weights")
            print("⚠️  WARNING: Not using class weights may result in poor performance on imbalanced datasets")
            print("⚠️  Consider using --weighted_loss for better results on rare chess themes")
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # Optimizer settings
    lr = 0.001
    weight_decay = 0.01
    warmup_steps = 1000
    
    # Choose optimizer based on args
    from model import Lamb, get_lr_with_warmup
    if args.optimizer.lower() == 'adam':
        if args.is_master:
            print(f"Using Adam optimizer with lr={lr}, weight_decay={weight_decay}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # Default to Lamb
        if args.is_master:
            print(f"Using Lamb optimizer with lr={lr}, weight_decay={weight_decay}, warmup_steps={warmup_steps}")
        optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Get current learning rate (for logging)
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    # Check for checkpoint file, but skip if in test mode
    # Check for ISC environment - use OUTPUT_PATH if available for checkpoint loading
    if isc_mode and "OUTPUT_PATH" in os.environ:
        checkpoint_dir = os.environ["OUTPUT_PATH"]
    else:
        checkpoint_dir = "checkpoints"
        
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_resume.pth")
    if not args.test_mode and os.path.exists(checkpoint_path):
        if args.is_master:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Support both formats - native format and ISC format
        if 'model_state_dict' in checkpoint:
            # Original format
            if is_distributed or isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model' in checkpoint:
            # ISC format from StrongResearch
            if args.is_master:
                print("Loading checkpoint in ISC format")
            if is_distributed or isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
        else:
            if args.is_master:
                print("Warning: Checkpoint doesn't contain recognized model state format")
        # Load optimizer state, supporting both formats
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            
        # Get global step, supporting both formats
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        elif 'train_sampler' in checkpoint and hasattr(checkpoint['train_sampler'], 'progress'):
            # ISC format uses the sampler progress as the global step
            global_step = checkpoint['train_sampler'].progress
        else:
            global_step = 0
        
        # Calculate if we completed the previous epoch
        train_size = int(0.8 * len(dataset))  # Same split as in training
        steps_per_epoch = train_size // 4  # batch_size=4
        completed_epochs = global_step // steps_per_epoch
        
        # Get epoch, supporting both formats
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        elif 'train_sampler' in checkpoint and hasattr(checkpoint['train_sampler'], 'epoch'):
            # ISC format uses the sampler epoch
            start_epoch = checkpoint['train_sampler'].epoch
        else:
            # Estimate from global step
            start_epoch = global_step // steps_per_epoch
            
        # Check if we completed the current epoch
        if global_step >= (start_epoch + 1) * steps_per_epoch:
            # We completed this epoch, start the next one
            start_epoch += 1
        
        if args.is_master:
            print(f"Resumed from checkpoint:")
            print(f"  Epoch: {start_epoch}")
            print(f"  Global Step: {global_step}")
            
            # Show loss and metrics if available
            if 'loss' in checkpoint:
                print(f"  Loss: {checkpoint['loss']:.8f}")
            if 'jaccard_loss' in checkpoint:
                print(f"  Jaccard Loss: {checkpoint['jaccard_loss']:.8f}")
                
            # Get learning rate
            lr_value = checkpoint.get('learning_rate', get_lr(optimizer))
            print(f"  Learning Rate: {lr_value:.8f}")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Completed epochs: {completed_epochs}")
            print(f"  Resuming from epoch: {start_epoch}")
    elif args.test_mode and args.is_master:
        print("Test mode enabled, skipping checkpoint loading")

    # Split the dataset into train and test sets
    if args.is_master:
        print("Splitting dataset into train and test sets...")
        
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)
    
    if args.is_master:
        timer.report(f"Initialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test board evaluations.")

    # Get batch size from args
    batch_size = args.batch_size
    
    # Create samplers based on distributed mode
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        
        # Use distributed samplers with DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    else:
        # Use standard samplers for local training
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Define dummy sampler for local mode so we can call set_epoch without errors
        train_sampler = type('DummySampler', (), {'set_epoch': lambda self, epoch: None})()
        
    if args.is_master:
        print(f"Using batch size: {batch_size}")
        timer.report("Prepared dataloaders")

    # Set number of epochs based on args
    if args.epochs is not None:
        max_epochs = args.epochs
    else:
        max_epochs = 3 if args.test_mode else 10
    
    if args.is_master:
        print(f"Training for {max_epochs} epochs")
        
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

            # Use debug mode for the first batch of the first epoch or first 5 batches
            debug_mode = (epoch == start_epoch and (i == 0 or i < 5))
            
            # Print detailed info for the very first batch
            detailed_debug = (epoch == start_epoch and i == 0)
            
            if detailed_debug:
                print("\n--------- Detailed Debug Information ---------")
                print(f"Inputs shape: {inputs.shape}")
                print(f"Data keys available: {list(data.keys())}")
                print(f"Labels datatype: {labels.dtype}")
                print(f"Using device: {device}")
                print(f"Model type: {type(model).__name__}")
                if hasattr(model, 'module'):
                    print(f"Underlying model: {type(model.module).__name__}")
                
            outputs = model(inputs, debug=detailed_debug)
            
            # Debug shape issues
            if debug_mode:
                print(f"\nBatch {i}: Model output shape: {outputs.shape}")
                print(f"Batch {i}: Labels shape: {labels.shape}")
                print(f"Batch {i}: Batch size: {batch_size}")
                print(f"Batch {i}: Number of labels: {num_labels}")
            
            # Handle shape issues - ensure outputs match labels exactly
            if outputs.shape != labels.shape:
                # For debugging
                if debug_mode:
                    print(f"Shape mismatch: outputs {outputs.shape} vs labels {labels.shape}")
                
                # Get correct batch size and feature dimensions
                batch_size = labels.size(0)
                feature_dim = labels.size(1)
                
                if outputs.dim() == 1:
                    # 1D tensor needs to be reshaped to 2D
                    outputs = outputs.view(batch_size, feature_dim)
                elif outputs.dim() == 2:
                    # If dimensions don't match, force reshape
                    outputs = outputs.view(batch_size, feature_dim)
                
                if debug_mode:
                    print(f"Reshaped outputs to: {outputs.shape}")
            
            loss = criterion(outputs, labels)
            
            # Apply sigmoid to get probabilities for metrics calculation
            output_probs = torch.sigmoid(outputs)
            
            # Output probs should already have the correct shape since we fixed outputs
            # But just to be safe, verify it matches the labels shape
            if output_probs.shape != labels.shape:
                if debug_mode:
                    print(f"Warning: Need to reshape output_probs from {output_probs.shape} to {labels.shape}")
                # Get correct batch size and feature dimensions
                batch_size = labels.size(0)
                feature_dim = labels.size(1)
                output_probs = output_probs.view(batch_size, feature_dim)
                
            jaccard_loss = jaccard_similarity(output_probs, labels, threshold=0.5)
            
            # Calculate precision, recall, F1 metrics (every 100 steps to avoid overhead)
            calculate_detailed_metrics = (i % 100 == 0)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item()
            running_jaccard_index = jaccard_loss.item()
            
            if args.is_master:  # Only log on master process
                current_lr = get_lr(optimizer)
                print(f"epoch: {epoch}/{max_epochs-1} step: {i+1} lr: {current_lr:.8f} loss: {running_loss:.8f} jaccard index: {running_jaccard_index:.8f}")
                
                # Calculate precision, recall, F1 periodically (to reduce computational overhead)
                metrics_dict = {}
                if calculate_detailed_metrics:
                    # Calculate precision, recall, F1 with different averaging methods
                    # Turn on verbose mode for the first calculation in each epoch
                    first_in_epoch = (i == 0 and epoch == start_epoch)
                    
                    # First batch of each epoch uses verbose mode to show progress
                    precision_micro, recall_micro, f1_micro = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='micro', verbose=first_in_epoch
                    )
                    precision_macro, recall_macro, f1_macro = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='macro', verbose=first_in_epoch
                    )
                    precision_weighted, recall_weighted, f1_weighted = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='weighted', verbose=first_in_epoch
                    )
                    
                    metrics_dict = {
                        "precision_micro": precision_micro,
                        "recall_micro": recall_micro,
                        "f1_micro": f1_micro,
                        "precision_macro": precision_macro,
                        "recall_macro": recall_macro,
                        "f1_macro": f1_macro,
                        "precision_weighted": precision_weighted,
                        "recall_weighted": recall_weighted,
                        "f1_weighted": f1_weighted
                    }
                    
                    # Generate and log full classification report every 1000 steps
                    if i % 1000 == 0:
                        # Get label names for the report
                        label_names = dataset.all_labels
                        report = get_classification_report(
                            output_probs, labels, threshold=0.5, labels=label_names,
                            verbose=(i == 0 and epoch == start_epoch)  # Verbose on first report
                        )
                        print("\nClassification Report:")
                        print(report)
                        if writer is not None:
                            writer.add_text('Classification Report', report, global_step)
                
                if writer is not None:
                    writer.add_scalar('Loss/train', running_loss, global_step)
                    writer.add_scalar('Jaccard/train', running_jaccard_index, global_step)
                    writer.add_scalar('Learning_rate', current_lr, global_step)
                    
                    # Log additional metrics if calculated
                    for metric_name, metric_value in metrics_dict.items():
                        writer.add_scalar(f'Metrics/{metric_name}', metric_value, global_step)
                
                # Log to wandb if enabled
                if args.wandb:
                    log_dict = {
                        "loss": running_loss,
                        "jaccard_index": running_jaccard_index,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "global_step": global_step
                    }
                    # Add metrics if calculated
                    log_dict.update(metrics_dict)
                    wandb.log(log_dict)
                
                if args.is_master and i % args.checkpoint_steps == 0 and i > 0:
                    # Check for ISC environment - use OUTPUT_PATH if available
                    if isc_mode and "OUTPUT_PATH" in os.environ:
                        checkpoint_dir = os.environ["OUTPUT_PATH"]
                    else:
                        checkpoint_dir = "checkpoints"
                        
                    # Create directory if it doesn't exist
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}_{i}.pth")
                    resume_filename = os.path.join(checkpoint_dir, "checkpoint_resume.pth")
                    
                    # Save the model state dict properly depending on model type
                    if is_distributed or isinstance(model, torch.nn.DataParallel):
                        model_state = model.module.state_dict()
                    else:
                        model_state = model.state_dict()
                        
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'global_step': global_step,
                        'jaccard_loss': jaccard_loss,
                        'learning_rate': current_lr,
                    }
                    
                    # Save both the timestamped checkpoint and the resume checkpoint
                    torch.save(checkpoint_data, checkpoint_filename)
                    torch.save(checkpoint_data, resume_filename)
                    
                    if writer is not None:
                        writer.add_text('Checkpoint', f'Saved checkpoint: {checkpoint_filename}', global_step)
                        print(f"Checkpoint saved at step {i} in {checkpoint_filename}")
                        print(f"Resume checkpoint saved at {resume_filename}")
            
            global_step += 1
    
    # Clean up
    if writer is not None:
        writer.close()
        if args.is_master:
            timer.report("Closed TensorBoard writer")
    
    # Close wandb run if it was used
    if args.is_master and args.wandb:
        wandb.finish()
        print("Closed Weights & Biases run")
    
    # Print completion message
    if args.is_master:
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        
        # Show different output paths based on environment
        if isc_mode and "OUTPUT_PATH" in os.environ:
            print(f"Checkpoints saved to: {os.environ['OUTPUT_PATH']}")
            if "LOSSY_ARTIFACT_PATH" in os.environ:
                print(f"Logs saved to: {os.environ['LOSSY_ARTIFACT_PATH']}")
        else:
            print(f"Checkpoints saved to: {os.path.join(os.getcwd(), 'checkpoints')}")
            print(f"Logs saved to: {log_dir}")
            
        print(f"{'='*60}\n")
    
    # Clean up distributed process group if we were using it
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()