 # Basic training loop insired by https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# This script trains locally on a single GPU.
# This version has a correction to ensure that it is training on the GPU; previously I was not correctly calling model.to(device), inputs.to(device), and labels.to(device).

# Set OpenMP thread limit to avoid "Thread creation failed" error
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads

from cycling_utils import TimestampedTimer, AtomicDirectory, atomic_torch_save, InterruptableDistributedSampler
import time
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import subprocess
import webbrowser
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from metrics import jaccard_similarity, precision_recall_f1, get_classification_report
import wandb
import torch
import yaml
import socket

timer = TimestampedTimer("Imported TimestampedTimer")

from dataset import ChessPuzzleDataset
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



def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Train chess puzzle classifier')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with a smaller dataset')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='chess-theme-classifier', help='Weights & Biases project name')
    parser.add_argument('--name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--distributed', action='store_true', help='Force distributed training mode (overrides auto-detection)')
    parser.add_argument('--local', action='store_true', help='Force local training mode (overrides auto-detection)')
    parser.add_argument('--weighted_loss', action='store_true', help='Use weighted BCE loss for class imbalance')
    parser.add_argument('--single_gpu', action='store_true', help='Run in single-GPU mode (replaces train_locally_single_gpu.py)')
    parser.add_argument('--optimizer', type=str, default='lamb', help='Optimizer to use: lamb or adam (default: lamb)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train (default: 3 for test mode, 10 for full)')
    parser.add_argument('--low_memory', action='store_true', help='Use lower memory settings for dataset processing')
    parser.add_argument('--full_dataset', action='store_true', help='Use full dataset tensor cache instead of class conditional augmentation')
    parser.add_argument('--full_class_conditional', action='store_true', help='Use full class-conditional augmented dataset (Milestone 2)')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of training steps per epoch (useful for testing with large datasets)')
    parser.add_argument('--dataset-id', type=str, default=None, help='Dataset ID for ISC training (overrides environment variable)')
    parser.add_argument("--grad-accum", help="gradient accumulation steps", type=int, default=6)       
    parser.add_argument("--save-steps", help="saving interval steps", type=int, default=1000)           
    parser.add_argument("--model-config", help="model config path", type=Path, default="/root/chess-theme-classifier/model_config.yaml")          
    parser.add_argument("--log-on-all-ranks", action='store_true', help="Enable logging on all ranks (default: False)")
    return parser.parse_args(args)

def main():
    # Parse command line arguments
    args = parse_args()

    # Check for ISC environment
    isc_mode = "LOSSY_ARTIFACT_PATH" in os.environ

    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.device_id = int(os.environ["LOCAL_RANK"])
    args.is_master = rank == 0
    torch.cuda.set_device(args.device_id)
    device = torch.device(f'cuda:{args.device_id}')
    is_distributed = True

    if args.device_id == 0: # If we are the 0th GPU on the local node (not related to master rank), print the hostname and args
        hostname = socket.gethostname()
        print("Hostname:", hostname)
        print(f"TrainConfig: {args}")
        print(f"World size: {args.world_size}")
        print(f"Device ID: {args.device_id}")
        print(f"Is master: {args.is_master}")
        print(f"Is distributed: {is_distributed}")
        print(f"Is single GPU: {args.single_gpu}")
        print(f"Is test mode: {args.test_mode}")
        print(f"Is weighted loss: {args.weighted_loss}")
    
    if args.is_master:
        print(f"Using device: {device} with local_rank: {args.device_id} (Running in {'distributed' if is_distributed else 'local'} mode)")
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
        csv_file = os.path.join('processed_lichess_puzzle_files', csv_filename)
        print(f"Using local dataset path: {csv_file}")
    
    if args.is_master:
        if args.test_mode:
            print(f"Running in TEST MODE with smaller dataset")
        elif args.single_gpu:
            print(f"Running in SINGLE GPU MODE with test dataset")
        else:
            print(f"Running with FULL DATASET. This may require significant memory and processing time.")
            
    # Get batch size from args
    batch_size = args.batch_size
    if args.is_master:
        print(f"Using batch size: {batch_size}")

    # Set number of epochs based on args
    if args.epochs is not None:
        max_epochs = args.epochs
    else:
        max_epochs = 3 if args.test_mode else 30

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
                "full_dataset": args.full_dataset,
                "full_class_conditional": args.full_class_conditional,
                "class_conditional_augmentation": args.full_class_conditional or (not args.full_dataset),
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
        mode_desc = "full dataset cache" if args.full_dataset else "class conditional augmentation"
        print(f"Creating dataset from {csv_file} with {mode_desc} and low_memory={low_memory}")
        if args.full_dataset:
            print(f"✅ Using pre-built tensor cache for fast loading")
        else:
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
                    print(f"Found all essential cache files in {processed_dir}")
                    
                    # Check optional files for debugging
                    all_monitored_files = truly_essential_files + conditional_aug_files + weighted_loss_files + optional_files
                    missing_optional = [f for f in all_monitored_files if f not in essential_files and not os.path.exists(os.path.join(processed_dir, f))]
                    if missing_optional:
                        print(f"Some non-essential cache files are missing (this is okay):")
                        for f in missing_optional:
                            print(f"  - {f}")
                    
                    print(f"Training will proceed using pre-processed cache files without CSV")
                
                # Update the csv_file path to use the processed directory
                csv_file = new_csv_file
                if args.is_master:
                    print(f"Using cache files from: {processed_dir}")
                    print(f"Setting csv_file path to: {csv_file}")
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
    # Use full dataset, full class-conditional, or regular class conditional augmentation
    if args.full_dataset:
        if args.is_master:
            print(f"Using full dataset tensor cache (4.9M samples)")
        dataset = ChessPuzzleDataset(csv_file, class_conditional_augmentation=False, low_memory=low_memory, use_cache=True)
    elif args.full_class_conditional:
        if args.is_master:
            print(f"Using full class-conditional augmented dataset (>4.9M samples)")
        dataset = ChessPuzzleDataset(csv_file, class_conditional_augmentation=True, low_memory=low_memory, use_cache=True, full_class_conditional=True)
    else:
        if args.is_master:
            print(f"Using class conditional augmentation (partial dataset)")
        dataset = ChessPuzzleDataset(csv_file, class_conditional_augmentation=True, low_memory=low_memory)
    
    # Get the number of labels from the dataset
    num_labels = len(dataset.all_labels)
    if args.is_master:
        print(f"Dataset loaded successfully:")
        print(f" Dataset size: {len(dataset):,} samples")
        print(f" Number of unique labels (themes + opening tags): {num_labels}")
        if args.full_dataset:
            mode = 'Full dataset cache'
        elif args.full_class_conditional:
            mode = 'Full class-conditional augmented dataset'
        else:
            mode = 'Class conditional augmentation (partial)'
        print(f"   Mode: {mode}")
    
    # Create model with the correct number of labels
    # Original hacakthon-winner-3 model config with default nlayers=2.
    # model = Model(num_labels=num_labels)

    # Load model config or use default values
    model_config = {}
    if os.path.exists(args.model_config):
        model_config = yaml.safe_load(open(args.model_config))
        
        # We'll use the full model architecture even with smaller datasets
        if "num_labels" in model_config:
            if model_config["num_labels"] < num_labels:
                # If config has fewer labels than dataset, that's a problem
                if args.is_master:
                    print(f"Warning: num_labels in model_config ({model_config['num_labels']}) is less than dataset ({num_labels})")
                    print(f"Increasing num_labels to match dataset: {num_labels}")
                model_config["num_labels"] = num_labels
            elif model_config["num_labels"] > num_labels:
                # Using larger model (from full dataset) with smaller dataset (test mode)
                if args.is_master:
                    print(f"Using full model architecture with {model_config['num_labels']} output labels")
                    print(f"Current dataset only has {num_labels} labels (likely using test dataset)")
                    print(f"Extra outputs will be ignored during training")
        else:
            # No num_labels in config, use the dataset's value
            model_config["num_labels"] = num_labels
            
        if args.is_master:
            print(f"ModelConfig: {model_config}")
    else:
        # No config file, use the dataset's value
        model_config = {
            "num_labels": num_labels, 
            "nlayers": 5, 
            "embed_dim": 64, 
            "inner_dim": 320, 
            "attention_dim": 64, 
            "use_1x1conv": True, 
            "dropout": 0.5
        }
        if args.is_master:
            print(f"Using default ModelConfig: {model_config}")
    
    model = Model(**model_config)
    model = model.to(device)
    
    # Wrap model in DDP if running in distributed mode
    if is_distributed:
        model = DDP(model, device_ids=[args.device_id])
        print(f"checkpoint-debug: Using DDP with device_ids: {args.device_id}")
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

    #########################################################
    # Dataset and Dataloader setup
    #########################################################

    # Split the dataset into train and test sets
    if args.is_master:
        print("Splitting dataset into train and test sets...")
        
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.85, 0.15], generator=random_generator)
    
    if args.is_master:
        timer.report(f"Initialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test board evaluations.")
    
    # Create samplers based on distributed mode
    if is_distributed:
        train_sampler = InterruptableDistributedSampler(train_dataset)
        test_sampler = InterruptableDistributedSampler(test_dataset)
        
        # Use distributed samplers with DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    else:
        # Use standard samplers for local training
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Define dummy sampler for local mode so we can call set_epoch without errors
        train_sampler = type('DummySampler', (), {'set_epoch': lambda self, epoch: None})()

    #########################################################
    # Checkpoint loading
    #########################################################
    output_directory = os.environ.get("CHECKPOINT_ARTIFACT_PATH", "checkpoints")
    saver = AtomicDirectory(output_directory=output_directory, is_master=args.is_master)

    # set the checkpoint_path if there is one to resume from
    checkpoint_path = None
    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)
        checkpoint_path = os.path.join(latest_checkpoint_path, "checkpoint.pt")

    if checkpoint_path:
        timer.report(f"checkpoint-debug: Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        print(f"checkpoint-debug: Loaded checkpoint from {checkpoint_path}")
        print(f"checkpoint-debug: Starting from epoch {start_epoch} and global step {global_step}")
        
    if args.is_master:
        print(f"Using batch size: {batch_size}")
        timer.report("Prepared dataloaders")

    # Set number of epochs based on args
    if args.epochs is not None:
        max_epochs = args.epochs
    else:
        max_epochs = 3 if args.test_mode else 30
    
    if args.is_master:
        print(f"Training for {max_epochs} epochs")
        
    for epoch in range(start_epoch, max_epochs):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        test_dataloader.sampler.set_epoch(epoch)
        
        for i, data in enumerate(train_dataloader, 0):
            # Check if we should stop early for testing with large datasets
            if args.max_steps is not None and i >= args.max_steps:
                if args.is_master:
                    print(f"Stopping epoch {epoch} early after {args.max_steps} steps for testing")
                break
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
            
            # Print detailed info for the very first batch
            detailed_debug = False #(epoch == start_epoch and i == 0)
            
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
    
            
            # Handle shape issues when using full model with smaller dataset
            if outputs.shape[1] > labels.shape[1]:
                # We're using a model with more output labels than in the current dataset
                if debug_mode:
                    print(f"Model has more outputs ({outputs.shape[1]}) than dataset labels ({labels.shape[1]})")
                    print(f"Using only the first {labels.shape[1]} outputs for loss calculation")
                
                # Only use the first part of the outputs that match the dataset's labels
                outputs_for_loss = outputs[:, :labels.shape[1]]
                
                if debug_mode:
                    print(f"Using outputs subset with shape: {outputs_for_loss.shape}")
            elif outputs.shape != labels.shape:
                # Some other shape mismatch that needs fixing
                if debug_mode:
                    print(f"Shape mismatch: outputs {outputs.shape} vs labels {labels.shape}")
                
                # Get correct batch size and feature dimensions
                batch_size = labels.size(0)
                feature_dim = labels.size(1)
                
                if outputs.dim() == 1:
                    # 1D tensor needs to be reshaped to 2D
                    outputs_for_loss = outputs.view(batch_size, feature_dim)
                elif outputs.dim() == 2:
                    # If dimensions don't match, force reshape
                    outputs_for_loss = outputs.view(batch_size, feature_dim)
                
                if debug_mode:
                    print(f"Reshaped outputs to: {outputs_for_loss.shape}")
            else:
                # Shapes match exactly
                outputs_for_loss = outputs
            
            # Use the adjusted outputs for loss calculation
            loss = criterion(outputs_for_loss, labels)
            
            # Apply sigmoid to get probabilities for metrics calculation
            # Use the same subset of outputs used for loss calculation
            output_probs = torch.sigmoid(outputs_for_loss)
            
            # Verify output_probs matches the labels shape
            if output_probs.shape != labels.shape:
                if debug_mode:
                    print(f"Warning: Need to reshape output_probs from {output_probs.shape} to {labels.shape}")
                # Get correct batch size and feature dimensions
                batch_size = labels.size(0)
                feature_dim = labels.size(1)
                output_probs = output_probs.view(batch_size, feature_dim)
                
            # Calculate Jaccard similarity with detailed debugging on first batch of first epoch
            # Force verbose mode for the first few batches regardless of dataset mode
            debug_jaccard = False # detailed_debug or (i < 3 and epoch == start_epoch)
            
            # Add additional diagnostics for the first batch
            if False: # i == 0 and epoch == start_epoch:
                # Print statistics about the predictions and targets
                print("\n----- JACCARD SIMILARITY DIAGNOSTICS -----")
                print(f"Output probabilities - min: {output_probs.min().item():.4f}, max: {output_probs.max().item():.4f}, mean: {output_probs.mean().item():.4f}")
                print(f"Target labels - min: {labels.min().item():.4f}, max: {labels.max().item():.4f}, mean: {labels.mean().item():.4f}")
                
                # Check how many positive values are in labels
                positive_labels = labels.sum().item()
                total_labels = labels.numel()
                print(f"Positive labels: {positive_labels} / {total_labels} = {positive_labels/total_labels:.6f}")
                
                # Print a few sample values
                num_samples = min(5, output_probs.size(0))
                num_features = min(10, output_probs.size(1))
                print(f"Sample output probabilities (first {num_samples} samples, first {num_features} features):")
                for s in range(num_samples):
                    print(f"  Sample {s}: {output_probs[s, :num_features].cpu().tolist()}")
                    
                print(f"Sample labels (first {num_samples} samples, first {num_features} features):")
                for s in range(num_samples):
                    print(f"  Sample {s}: {labels[s, :num_features].cpu().tolist()}")
                print("-----------------------------------------\n")
            
            jaccard_loss = jaccard_similarity(output_probs, labels, threshold=0.5, 
                                             adaptive_threshold=True, verbose=debug_jaccard)
            
            # Calculate precision, recall, F1 metrics (every 100 steps to avoid overhead)
            
            #########################################################
            # Needed for tensorboard classification performance (F1, Precision, Recall) graphs
            #########################################################
            calculate_detailed_metrics = (i % 100 == 0)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item()
            running_jaccard_index = jaccard_loss.item()
            
            # NOTE: log_on_all_ranks is currently required for checkpoint saving to work correctly. We need to detangle logging and checkpoint saving.
            if args.is_master or args.log_on_all_ranks:  # Only log on master process
                current_lr = get_lr(optimizer)
                print(f"epoch: {epoch}/{max_epochs-1} step: {global_step+1} lr: {current_lr:.8f} loss: {running_loss:.8f} jaccard index: {running_jaccard_index:.8f}")
                
                # Calculate precision, recall, F1 periodically (to reduce computational overhead)
                metrics_dict = {}
                if calculate_detailed_metrics:
                    # Calculate precision, recall, F1 with different averaging methods
                    
                    precision_micro, recall_micro, f1_micro = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='micro', verbose=False, adaptive_threshold=True
                    )
                    precision_macro, recall_macro, f1_macro = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='macro', verbose=False, adaptive_threshold=True
                    )
                    precision_weighted, recall_weighted, f1_weighted = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='weighted', verbose=False, adaptive_threshold=True
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
                    # print(f"epoch: {epoch} step: {i} metrics_dict: {metrics_dict}")
                
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
                
                #########################################################
                # Checkpoint saving; all ranks must participate in preparing the checkpoint directory, only master rank saves the checkpoint, then all ranks must finalize.
                # This is currently behind an is_master flag, which is overridden by log_on_all_ranks. I should refactor to move this out of the presumed logging code,
                # which is incorrectly also handling checkpoint saving.
                #########################################################

                # Checkpoint saving - following train_chessVision.py reference implementation exactly
                # Use batch-based triggers like the reference for proper distributed synchronization  
                batch = i  # Using step counter as batch equivalent for now
                is_save_batch = (batch + 1) % args.save_steps == 0 # TODO: add is_last_batch to save when last batch of epoch
                print(f"checkpoint-debug: is_save_batch: {is_save_batch}")

                if is_save_batch:

                    print(f"checkpoint-debug: calling prepare_checkpoint_directory")
                    
                    # prepare the checkpoint directory - called by ALL ranks
                    checkpoint_directory = saver.prepare_checkpoint_directory()
                    print(f"checkpoint-debug: checkpoint_directory: {checkpoint_directory}")
                    print(f"checkpoint-debug: prepare_checkpoint_directory done")

                    # saving files to the checkpoint_directory - called by MASTER rank only
                    if args.is_master:
                        print(f"checkpoint-debug: calling atomic_torch_save")
                        atomic_torch_save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "train_sampler": train_dataloader.sampler.state_dict(),
                                "test_sampler": test_dataloader.sampler.state_dict(),
                                "global_step": global_step,
                            },
                            os.path.join(checkpoint_directory, "checkpoint.pt"),
                        )
                        print(f"checkpoint-debug: atomic_torch_save done")    
                    print(f"checkpoint-debug: calling saver.symlink_latest")

                    # finalizing checkpoint with symlink - called by ALL ranks
                    saver.symlink_latest(checkpoint_directory)
                    print(f"checkpoint-debug: saver.symlink_latest done")
            
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
        if isc_mode and "CHECKPOINT_ARTIFACT_PATH" in os.environ:
            print(f"Checkpoints saved to: {os.environ['CHECKPOINT_ARTIFACT_PATH']}")
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
