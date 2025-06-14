# This script is the result of attempting to deduplicate the code in train.py and train-isc.py. It doesn't work very well;
# training locally with train-depuplication-attempt.py results in very few log lines for training, and key metrics like lr and jaccard index are zeroes.
# I'm not sure why this is the case, but I'm not going to spend more time on it right now.
# 
# 
# Basic training loop insired by https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# This script trains locally on a single GPU.
# This version has a correction to ensure that it is training on the GPU; previously I was not correctly calling model.to(device), inputs.to(device), and labels.to(device).

# Set OpenMP thread limit to avoid "Thread creation failed" error
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads

from cycling_utils import TimestampedTimer
from cycling_utils import InterruptableDistributedSampler, MetricsTracker, AtomicDirectory, atomic_torch_save
import time
import argparse
import yaml
import math
from pathlib import Path
import socket
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
    # Original arguments
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
    
    # New ISC-specific arguments
    parser.add_argument('--model-config', type=str, default='model_config.yaml',
                        help='Path to model configuration file')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save checkpoints (defaults to OUTPUT_PATH in ISC mode)')
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path to checkpoint file to resume from')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--wd', type=float, default=0.01, 
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--ws', type=int, default=1000,
                        help='Warmup steps (default: 1000)')
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--save-steps', type=int, default=50,
                        help='Number of steps between saves in ISC mode (default: 50)')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Detect ISC environment
    # Make sure all required env vars are present before declaring ISC mode
    required_isc_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    isc_mode = all(var in os.environ for var in required_isc_vars)
    
    # Handle distributed setup differently based on ISC vs. local environment
    if isc_mode:
        try:
            # ISC environment - direct initialization without auto-detection
            dist.init_process_group("nccl")  # ISC environment already has required env vars
            rank = int(os.environ["RANK"])  # Global rank in cluster
            args.world_size = int(os.environ["WORLD_SIZE"])  # Total GPUs
            args.device_id = int(os.environ["LOCAL_RANK"])  # Local rank on this node
            args.is_master = rank == 0  # Master process flag
            
            # Set device for this process
            torch.cuda.set_device(args.device_id)
            device = torch.device(f'cuda:{args.device_id}')
            
            # Set distributed flag
            is_distributed = True
            local_rank = args.device_id
            
            if args.is_master:
                hostname = socket.gethostname()
                print(f"Running in ISC mode on host: {hostname}")
                print(f"Distributed setup: Rank {rank}, World Size {args.world_size}, Local Device {args.device_id}")
        except Exception as e:
            # If anything fails, fall back to local mode
            print(f"Warning: Failed to initialize ISC mode: {e}")
            print("Falling back to local mode")
            isc_mode = False
            # Continue to local mode setup below
    
    if not isc_mode:
        # Local environment - use auto-detection logic
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
        
        # Add device_id and world_size for compatibility with ISC code
        args.device_id = local_rank
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if args.is_master:
        print(f"Using device: {device} with local_rank: {local_rank} (Running in {'distributed' if is_distributed else 'local'} mode)")
    torch.autograd.set_detect_anomaly(True)
    
    # Setup checkpoint directories for ISC or local mode
    if isc_mode:
        # In ISC mode, use environment variables for paths
        if not args.save_dir and "OUTPUT_PATH" in os.environ:
            args.save_dir = os.environ["OUTPUT_PATH"]
        
        # Create saver using AtomicDirectory for ISC compatibility
        if args.save_dir:
            saver = AtomicDirectory(output_directory=args.save_dir, is_master=args.is_master)
            if args.is_master:
                print(f"Using ISC checkpoint directory: {args.save_dir}")
        else:
            args.save_dir = "checkpoints"  # Default if not specified
            saver = AtomicDirectory(output_directory=args.save_dir, is_master=args.is_master)
            if args.is_master:
                print(f"No OUTPUT_PATH found, using default: {args.save_dir}")
    else:
        # In local mode, ensure checkpoints directory exists
        args.save_dir = args.save_dir or "checkpoints"
        os.makedirs(args.save_dir, exist_ok=True)
        if args.is_master:
            print(f"Using local checkpoint directory: {args.save_dir}")
            
        # Create a dummy saver object for local mode
        # We can't use AtomicDirectory in local mode because it requires environment variables
        # that are only present in ISC mode
        class DummySaver:
            def __init__(self, output_directory, is_master):
                self.output_directory = output_directory
                self.is_master = is_master
                self.symlink_name = "latest"
                
            def prepare_checkpoint_directory(self):
                # In local mode, just return the checkpoint directory
                os.makedirs(self.output_directory, exist_ok=True)
                return self.output_directory
                
            def symlink_latest(self, checkpoint_directory):
                # In local mode, just create a symlink to the latest checkpoint
                if self.is_master:
                    try:
                        latest_link = os.path.join(self.output_directory, self.symlink_name)
                        if os.path.exists(latest_link) or os.path.islink(latest_link):
                            os.remove(latest_link)
                        os.symlink(os.path.basename(checkpoint_directory), latest_link)
                    except Exception as e:
                        print(f"Warning: Could not create symlink: {e}")
        
        saver = DummySaver(output_directory=args.save_dir, is_master=args.is_master)

    # Create a timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Initialize timer if we haven't already
    if 'timer' not in locals():
        timer = TimestampedTimer("Initialized timer")
        
    # Check for ISC environment's logging path
    log_dir_from_env = "LOSSY_ARTIFACT_PATH" in os.environ
    
    # Set log directory based on environment (local or ISC)
    if isc_mode and log_dir_from_env and args.is_master:
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
                
    # Select the appropriate dataset file; _test.csv is only 100 lines/positions.
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
        # Use the local path in the dataset directory
        csv_file = os.path.join('dataset', csv_filename)
    
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
    
    # Check if the CSV file exists
    csv_exists = os.path.exists(csv_file)
    
    # If CSV doesn't exist, use cache files from processed_lichess_puzzle_files directly
    if not csv_exists:
        if args.is_master:
            print(f"⚠️ CSV file {csv_file} not found. Looking for pre-processed cache files...")
        
        # Get the base filename without path
        csv_basename = os.path.basename(csv_file)
        
        # Determine the appropriate processed directory based on whether we're in ISC mode or not
        if isc_mode and dataset_id:
            # In ISC mode, check if we have cache files in the ISC data directory
            processed_dir = f'/data/{dataset_id}'
            if args.is_master:
                print(f"Running in ISC mode, checking for cache files in: {processed_dir}")
        else:
            # Standard case - use local processed_lichess_puzzle_files directory
            processed_dir = 'processed_lichess_puzzle_files'
            
        # Update csv_file to point to the processed directory
        # This will make the dataset loader look for cache files in the correct directory
        new_csv_file = os.path.join(processed_dir, csv_basename)
        
        if os.path.exists(processed_dir):
            # Separate truly essential files from nice-to-have files
            # Essential files without which training cannot proceed at all
            truly_essential_files = [
                f"{csv_basename}.themes.json",
                f"{csv_basename}.openings.json"
            ]
            
            # Files needed specifically for class conditional augmentation
            conditional_aug_files = [
                f"{csv_basename}.cooccurrence.json",
                f"{csv_basename}.tensors.pt_conditional",
                f"{csv_basename}.tensors.pt_conditional.augmented_indices.json"
            ]
            
            # Files needed for weighted loss (but can train without them)
            weighted_loss_files = [
                f"{csv_basename}.class_weights.pt"
            ]
            
            # Optional cache files - we'll just note if they're missing
            optional_files = [
                f"{csv_basename}.tensors.pt_reflected",
                f"{csv_basename}.tensors.pt"
            ]
            
            # For simplicity in the current mode, create the essential files list
            essential_files = truly_essential_files.copy()
            
            # Add conditional augmentation files if we're using that feature
            if True:  # Class conditional augmentation is enabled by default in line 410
                essential_files.extend(conditional_aug_files)
            
            # Add weighted loss files if that feature is enabled
            if args.weighted_loss:
                essential_files.extend(weighted_loss_files)
            
            # Check for essential files
            essential_files_exist = all(os.path.exists(os.path.join(processed_dir, f)) for f in essential_files)
            
            if essential_files_exist:
                if args.is_master:
                    print(f"✅ Found all essential cache files in {processed_dir}")
                    
                    # Check optional files for debugging
                    all_monitored_files = truly_essential_files + conditional_aug_files + weighted_loss_files + optional_files
                    missing_optional = [f for f in all_monitored_files if f not in essential_files and not os.path.exists(os.path.join(processed_dir, f))]
                    if missing_optional:
                        print(f"ℹ️ Some non-essential cache files are missing (this is okay):")
                        for f in missing_optional:
                            print(f"  - {f}")
                    
                    print(f"⚠️ Training will proceed using pre-processed cache files without CSV")
                
                # Update the csv_file path to use the processed directory
                csv_file = new_csv_file
                if args.is_master:
                    print(f"👉 Using cache files from: {processed_dir}")
                    print(f"👉 Setting csv_file path to: {csv_file}")
            else:
                missing_files = [f for f in essential_files if not os.path.exists(os.path.join(processed_dir, f))]
                
                # Check if only non-truly-essential files are missing
                missing_truly_essential = [f for f in truly_essential_files if not os.path.exists(os.path.join(processed_dir, f))]
                
                if not missing_truly_essential:
                    # We're missing some feature-specific files but have the truly essential ones
                    # We can still proceed but with warnings
                    if args.is_master:
                        print(f"⚠️ Some feature-specific cache files are missing from {processed_dir}:")
                        for f in missing_files:
                            print(f"  - {f}")
                        
                        # Warn about disabled features
                        if any(f in missing_files for f in conditional_aug_files):
                            print(f"⚠️ Class conditional augmentation may be limited due to missing files")
                        
                        if any(f in missing_files for f in weighted_loss_files) and args.weighted_loss:
                            print(f"⚠️ Weighted loss feature may be limited due to missing class weight file")
                        
                        print(f"⚠️ Training will proceed with limited features using available cache files")
                    
                    # Update the csv_file path to use the processed directory despite missing some files
                    csv_file = new_csv_file
                    if args.is_master:
                        print(f"👉 Using available cache files from: {processed_dir}")
                        print(f"👉 Setting csv_file path to: {csv_file}")
                    essential_files_exist = True  # Allow training to proceed
                else:
                    if args.is_master:
                        print(f"❌ Some truly essential cache files are missing from {processed_dir}:")
                        for f in missing_truly_essential:
                            print(f"  - {f}")
                        
                        # If in ISC mode and files missing, try the local processed directory as fallback
                        if isc_mode and 'processed_lichess_puzzle_files' not in processed_dir:
                            fallback_dir = 'processed_lichess_puzzle_files'
                            if os.path.exists(fallback_dir):
                                print(f"Checking fallback directory: {fallback_dir}")
                                # Check for essential files in fallback
                                fallback_essential_exist = all(os.path.exists(os.path.join(fallback_dir, f)) for f in truly_essential_files)
                                if fallback_essential_exist:
                                    print(f"✅ Found truly essential cache files in fallback directory {fallback_dir}")
                                    csv_file = os.path.join(fallback_dir, csv_basename)
                                    print(f"👉 Using cache files from: {fallback_dir}")
                                    print(f"👉 Setting csv_file path to: {csv_file}")
                                    # Success! We found the essential files in the fallback directory
                                    essential_files_exist = True  # Mark as successful
                
                # Only raise error if we didn't find truly essential files in primary or fallback location
                if not essential_files_exist and missing_truly_essential:
                    raise FileNotFoundError(
                        f"CSV file {csv_file} not found and truly essential cache files not available in {processed_dir}. "
                        "Please provide either the original CSV file or at minimum the themes.json and openings.json files."
                    )
        else:
            if args.is_master:
                print(f"❌ Directory {processed_dir} not found")
                
                # If in ISC mode and primary directory not found, try local directory as fallback
                fallback_success = False
                if isc_mode and 'processed_lichess_puzzle_files' not in processed_dir:
                    fallback_dir = 'processed_lichess_puzzle_files'
                    if os.path.exists(fallback_dir):
                        print(f"Checking fallback directory: {fallback_dir}")
                        # Define truly essential files to check in fallback
                        truly_essential_files = [
                            f"{csv_basename}.themes.json",
                            f"{csv_basename}.openings.json"
                        ]
                        
                        # Check for truly essential files first - these are non-negotiable
                        fallback_truly_essential_exist = all(os.path.exists(os.path.join(fallback_dir, f)) for f in truly_essential_files)
                        
                        if fallback_truly_essential_exist:
                            # Now check for feature-specific files
                            conditional_aug_files = [
                                f"{csv_basename}.cooccurrence.json",
                                f"{csv_basename}.tensors.pt_conditional",
                                f"{csv_basename}.tensors.pt_conditional.augmented_indices.json"
                            ]
                            
                            weighted_loss_files = [
                                f"{csv_basename}.class_weights.pt"
                            ]
                            
                            # Check which feature-specific files exist
                            missing_conditional = [f for f in conditional_aug_files if not os.path.exists(os.path.join(fallback_dir, f))]
                            missing_weighted = [f for f in weighted_loss_files if not os.path.exists(os.path.join(fallback_dir, f))]
                            
                            # We can proceed with fallback if we have truly essential files
                            print(f"✅ Found truly essential cache files in fallback directory {fallback_dir}")
                            
                            # Warn about any missing feature-specific files
                            if missing_conditional:
                                print(f"⚠️ Some class conditional augmentation files are missing:")
                                for f in missing_conditional:
                                    print(f"  - {f}")
                                print(f"⚠️ Class conditional augmentation may be limited")
                                
                            if missing_weighted and args.weighted_loss:
                                print(f"⚠️ Class weight file is missing but weighted loss is enabled:")
                                for f in missing_weighted:
                                    print(f"  - {f}")
                                print(f"⚠️ Weighted loss feature may use default weights")
                            
                            csv_file = os.path.join(fallback_dir, csv_basename)
                            print(f"👉 Using cache files from: {fallback_dir}")
                            print(f"👉 Setting csv_file path to: {csv_file}")
                            fallback_success = True  # Mark that we found at least the truly essential files
            
            # Only raise the error if we didn't find files in the fallback location
            if not fallback_success:
                raise FileNotFoundError(
                    f"CSV file {csv_file} not found and {processed_dir} directory not found. "
                    "Please provide either the original CSV file or the pre-processed cache files."
                )
    
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
        try:
            # Compute label weights
            pos_weights = compute_label_weights(dataset)
            pos_weights = pos_weights.to(device)
            
            if args.is_master:
                print("Using weighted BCE loss for class imbalance mitigation")
            
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        except Exception as e:
            if args.is_master:
                print(f"⚠️ Error computing class weights: {e}")
                print("⚠️ Falling back to standard BCE loss")
                print("⚠️ This may result in reduced performance for rare themes")
            criterion = torch.nn.BCEWithLogitsLoss()
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
    # Handle checkpoint loading based on mode (ISC or local)
    checkpoint_path = None
    
    if isc_mode:
        # In ISC mode, we use a different approach to find the latest checkpoint
        if args.load_path and os.path.isfile(args.load_path):
            # Explicitly specified checkpoint
            checkpoint_path = args.load_path
            if args.is_master:
                print(f"Using explicitly specified checkpoint: {checkpoint_path}")
        else:
            # Try to find latest checkpoint using ISC symlink convention
            local_resume_path = os.path.join(args.save_dir, saver.symlink_name)
            if os.path.islink(local_resume_path):
                checkpoint = os.path.join(os.readlink(local_resume_path), "checkpoint.pt")
                if os.path.isfile(checkpoint):
                    checkpoint_path = checkpoint
                    if args.is_master:
                        print(f"Found latest ISC checkpoint via symlink: {checkpoint_path}")
    else:
        # In local mode, use the standard approach
        if args.load_path and os.path.isfile(args.load_path):
            # Explicitly specified checkpoint
            checkpoint_path = args.load_path
            if args.is_master:
                print(f"Using explicitly specified checkpoint: {checkpoint_path}")
        else:
            # Use default resume checkpoint path
            default_checkpoint_path = os.path.join(args.save_dir, "checkpoint_resume.pth")
            if os.path.exists(default_checkpoint_path):
                checkpoint_path = default_checkpoint_path
                if args.is_master:
                    print(f"Found default resume checkpoint: {checkpoint_path}")
    
    # Load checkpoint if found and not in test mode
    if checkpoint_path and not args.test_mode:
        if args.is_master:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint with appropriate device mapping
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
        
        # Handle sampler state for InterruptableDistributedSampler
        if isc_mode and 'train_sampler' in checkpoint and isinstance(train_dataloader.sampler, InterruptableDistributedSampler):
            try:
                train_dataloader.sampler.load_state_dict(checkpoint['train_sampler'])
                if 'test_sampler' in checkpoint:
                    test_dataloader.sampler.load_state_dict(checkpoint['test_sampler'])
                if args.is_master:
                    print(f"Loaded sampler state from ISC checkpoint")
            except Exception as e:
                if args.is_master:
                    print(f"Warning: Could not load sampler state: {e}")
        
        # Load metrics and timer if available (ISC format)
        if 'metrics' in checkpoint and isc_mode:
            metrics = checkpoint['metrics']
            if args.is_master:
                print(f"Loaded metrics state from checkpoint")
        
        if 'timer' in checkpoint and isc_mode:
            timer = checkpoint['timer']
            timer.start_time = time.time()  # Reset timer start time
            if args.is_master:
                print(f"Loaded timer from checkpoint")
            
        # Get global step, supporting both formats
        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        elif 'train_sampler' in checkpoint and hasattr(checkpoint['train_sampler'], 'progress'):
            # ISC format uses the sampler progress as the global step
            global_step = checkpoint['train_sampler'].progress
        else:
            global_step = 0
        
        # Get epoch, supporting both formats
        if isc_mode and isinstance(train_dataloader.sampler, InterruptableDistributedSampler):
            # In ISC mode with InterruptableDistributedSampler, use sampler's epoch
            start_epoch = train_dataloader.sampler.epoch
        elif 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        elif 'train_sampler' in checkpoint and hasattr(checkpoint['train_sampler'], 'epoch'):
            # ISC format uses the sampler epoch
            start_epoch = checkpoint['train_sampler'].epoch
        else:
            # Estimate from global step
            train_size = int(0.8 * len(dataset))  # Same split as in training
            steps_per_epoch = train_size // batch_size
            start_epoch = global_step // steps_per_epoch
        
        # Check if we completed the current epoch in non-ISC mode
        if not isc_mode:
            train_size = int(0.8 * len(dataset))
            steps_per_epoch = train_size // batch_size
            completed_epochs = global_step // steps_per_epoch
            
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
            
            # Log additional metrics if available
            for metric_name in ['precision_micro', 'recall_micro', 'f1_micro', 'f1_macro']:
                if metric_name in checkpoint:
                    print(f"  {metric_name}: {checkpoint[metric_name]:.8f}")
            
            # Get learning rate
            lr_value = checkpoint.get('learning_rate', get_lr(optimizer))
            print(f"  Learning Rate: {lr_value:.8f}")
            
            # In non-ISC mode, show additional info
            if not isc_mode:
                print(f"  Steps per epoch: {steps_per_epoch}")
                print(f"  Completed epochs: {completed_epochs}")
            print(f"  Resuming from epoch: {start_epoch}")
    elif args.test_mode and args.is_master:
        print("Test mode enabled, skipping checkpoint loading")
        start_epoch = 0
        global_step = 0
    else:
        # No checkpoint found or test mode
        start_epoch = 0
        global_step = 0
        if args.is_master and not args.test_mode:
            print("No checkpoint found, starting from scratch")

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
    if isc_mode:
        # In ISC mode, always use InterruptableDistributedSampler
        train_sampler = InterruptableDistributedSampler(train_dataset)
        test_sampler = InterruptableDistributedSampler(test_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
        
        if args.is_master:
            print(f"Using InterruptableDistributedSampler for ISC mode")
    elif is_distributed:
        # In non-ISC distributed mode, use standard DistributedSampler
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
        # Initialize metrics dictionary if in ISC mode
        if isc_mode:
            metrics = {"train": MetricsTracker(), "test": MetricsTracker()}
        
        # Set up train loop based on sampler type
        if isc_mode and isinstance(train_dataloader.sampler, InterruptableDistributedSampler):
            # Use ISC-style in_epoch context manager
            epoch_context = train_dataloader.sampler.in_epoch(epoch)
        else:
            # Use regular epoch setting for other samplers
            # Only call set_epoch if the sampler has that method (to avoid AttributeError)
            if hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)
            # Create dummy context manager
            epoch_context = type('DummyContext', (), {'__enter__': lambda self: None, '__exit__': lambda self, *args: None})()
            
        # Compute steps per epoch for gradient accumulation
        train_batches_per_epoch = len(train_dataloader)
        train_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum)
            
        if args.is_master:
            print(f"Training epoch {epoch} with {train_batches_per_epoch} batches, {train_steps_per_epoch} steps (grad_accum={args.grad_accum})")
            
        # Set up gradient accumulation
        optimizer.zero_grad()
        
        # Use context manager for ISC mode
        with epoch_context:
            for i, data in enumerate(train_dataloader, 0):
                running_loss = 0.0
                running_jaccard_index = 0.0
                
                inputs = data['board']
                inputs = inputs.unsqueeze(1).to(device)
                labels = data['themes'].to(device)
                
                # Check if we need to accumulate gradients or update weights
                is_accum_step = ((i + 1) % args.grad_accum == 0) or (i + 1 == train_batches_per_epoch)
                
                # Apply learning rate warmup
                # Calculate total steps since the start of training
                total_step = global_step
                # Apply learning rate scheduler function
                warmup_steps = args.ws
                lr = args.lr
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
                    print(f"Gradient accumulation steps: {args.grad_accum}")
                
                    
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
                    actual_batch_size = labels.size(0)
                    feature_dim = labels.size(1)
                    
                    if outputs.dim() == 1:
                        # 1D tensor needs to be reshaped to 2D
                        outputs = outputs.view(actual_batch_size, feature_dim)
                    elif outputs.dim() == 2:
                        # If dimensions don't match, force reshape
                        outputs = outputs.view(actual_batch_size, feature_dim)
                    
                    if debug_mode:
                        print(f"Reshaped outputs to: {outputs.shape}")
                
                # Scale the loss based on gradient accumulation
                if isc_mode:
                    # For ISC mode, follow the train-isc.py approach with sum reduction
                    loss = criterion(outputs, labels) / args.grad_accum
                else:
                    # For regular mode, just use regular loss
                    loss = criterion(outputs, labels)
                
                # Apply sigmoid to get probabilities for metrics calculation
                output_probs = torch.sigmoid(outputs)
                
                # Output probs should already have the correct shape since we fixed outputs
                # But just to be safe, verify it matches the labels shape
                if output_probs.shape != labels.shape:
                    if debug_mode:
                        print(f"Warning: Need to reshape output_probs from {output_probs.shape} to {labels.shape}")
                    # Get correct batch size and feature dimensions
                    actual_batch_size = labels.size(0)
                    feature_dim = labels.size(1)
                    output_probs = output_probs.view(actual_batch_size, feature_dim)
                    
                jaccard_loss = jaccard_similarity(output_probs, labels, threshold=0.5)
                
                # Calculate precision, recall, F1 metrics (every 100 steps to avoid overhead)
                calculate_detailed_metrics = (i % 100 == 0)
                
                # Backward pass with gradient accumulation
                loss.backward()
                
                # Update sampler progress in ISC mode
                if isc_mode and isinstance(train_dataloader.sampler, InterruptableDistributedSampler):
                    train_dataloader.sampler.advance(len(inputs))
                
                # Update metrics tracking for ISC mode
                if isc_mode and 'metrics' in locals():
                    metrics["train"].update({
                        "examples_seen": len(inputs),
                        "accum_loss": loss.item() * args.grad_accum,  # undo loss scale for reporting
                        "jaccard": jaccard_loss.item()
                    })
                
                # Only update weights when we've accumulated enough gradients or at the end
                if is_accum_step:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # print statistics
                running_loss = loss.item() * (args.grad_accum if isc_mode else 1)  # Scale loss for reporting
                running_jaccard_index = jaccard_loss.item()
            
                if args.is_master:  # Only log on master process; should this also have an (if globa_step % log_ever_n_steps == 0) clause?
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
                            '''
                            print("\nClassification Report:")
                            print(report)
                            '''
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
                    
                    # Determine if this is a save step
                    save_checkpoint = args.is_master and ((i % args.checkpoint_steps == 0 and i > 0) or
                                                    (i % args.save_steps == 0 and i > 0 and isc_mode))
                    
                    if save_checkpoint:
                        # Prepare for checkpoint saving
                        if isc_mode:
                            # In ISC mode, use the saver to prepare a new checkpoint directory
                            checkpoint_directory = saver.prepare_checkpoint_directory()
                            if args.is_master:
                                print(f"Preparing checkpoint directory: {checkpoint_directory}")
                        else:
                            # In local mode, use standard checkpoint directory
                            checkpoint_dir = args.save_dir
                            # Create directory if it doesn't exist
                            os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        # Get current learning rate
                        current_lr = get_lr(optimizer)
                        
                        # Save the model state dict properly depending on model type
                        if is_distributed or isinstance(model, torch.nn.DataParallel):
                            model_state = model.module.state_dict()
                        else:
                            model_state = model.state_dict()
                        
                        if isc_mode and isinstance(train_dataloader.sampler, InterruptableDistributedSampler):
                            # In ISC mode, save in ISC format with sampler state
                            checkpoint_data = {
                                "model": model_state,
                                "optimizer": optimizer.state_dict(),
                                "train_sampler": train_dataloader.sampler.state_dict(),
                                "test_sampler": test_dataloader.sampler.state_dict(),
                                "metrics": metrics if 'metrics' in locals() else None,
                                "timer": timer,
                                "loss": loss.item(),
                                "jaccard_loss": jaccard_loss.item(),
                                "learning_rate": current_lr,
                                "epoch": epoch,
                                "global_step": global_step
                            }
                            
                            # Add any detailed metrics if available
                            if 'metrics_dict' in locals() and metrics_dict:
                                for k, v in metrics_dict.items():
                                    checkpoint_data[k] = v
                            
                            # Use atomic save in ISC mode
                            atomic_torch_save(
                                checkpoint_data,
                                os.path.join(checkpoint_directory, "checkpoint.pt"),
                            )
                            saver.symlink_latest(checkpoint_directory)
                            
                            if args.is_master:
                                print(f"ISC checkpoint saved at step {i} in {checkpoint_directory}/checkpoint.pt")
                        else:
                            # In local mode, save in both timestamped and resume formats
                            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                            checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}_{i}.pth")
                            resume_filename = os.path.join(checkpoint_dir, "checkpoint_resume.pth")
                            
                            checkpoint_data = {
                                'epoch': epoch,
                                'model_state_dict': model_state,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss.item() if hasattr(loss, 'item') else loss,
                                'global_step': global_step,
                                'jaccard_loss': jaccard_loss.item(),
                                'learning_rate': current_lr,
                            }
                            
                            # Add sampler state if available (for compatibility)
                            if hasattr(train_dataloader.sampler, 'state_dict'):
                                checkpoint_data['train_sampler'] = train_dataloader.sampler.state_dict()
                            
                            # Add any detailed metrics if available
                            if 'metrics_dict' in locals() and metrics_dict:
                                for k, v in metrics_dict.items():
                                    checkpoint_data[k] = v
                            
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