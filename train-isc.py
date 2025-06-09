import sys
# Continue with imports for backward compatibility
from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import os
import socket
import yaml
import time
import math
import sys

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    AtomicDirectory,
    atomic_torch_save,
)

from dataset import ChessPuzzleDataset
from model import Model, Lamb
from metrics import jaccard_similarity, precision_recall_f1, get_classification_report

timer.report("Completed imports")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", help="model config path", type=Path, default="/root/chess-theme-classifier/model_config.yaml")
    parser.add_argument("--save-dir", help="save checkpoint path", type=Path, default=os.environ.get("OUTPUT_PATH"))
    parser.add_argument("--load-path", help="path to checkpoint.pt file to resume from", type=Path, default="/root/chess-theme-classifier/recover/checkpoint.pt")
    parser.add_argument("--dataset-id", help="dataset ID for constructing dataset path", type=str)
    parser.add_argument("--bs", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--wd", help="weight decay", type=float, default=0.01)
    parser.add_argument("--ws", help="learning rate warm up steps", type=int, default=1000)
    parser.add_argument("--grad-accum", help="gradient accumulation steps", type=int, default=6)
    parser.add_argument("--save-steps", help="saving interval steps", type=int, default=50)
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with a smaller dataset")
    return parser

def main(args, timer):
    dist.init_process_group("nccl")  # Expects RANK set in environment variable
    rank = int(os.environ["RANK"])  # Rank of this GPU in cluster
    args.world_size = int(os.environ["WORLD_SIZE"]) # Total number of GPUs in the cluster
    args.device_id = int(os.environ["LOCAL_RANK"])  # Rank on local node
    args.is_master = rank == 0  # Master node for saving / reporting
    torch.cuda.set_device(args.device_id)  # Enables calling 'cuda'
    torch.autograd.set_detect_anomaly(True) 

    if args.device_id == 0:
        hostname = socket.gethostname()
        print("Hostname:", hostname)
        print(f"TrainConfig: {args}")
    timer.report("Setup for distributed training")

    saver = AtomicDirectory(output_directory=args.save_dir, is_master=args.is_master)
    timer.report("Validated checkpoint path")

    # Use a smaller dataset if in test mode
    if args.test_mode:
        csv_basename = 'lichess_db_puzzle_test.csv'
    else:
        csv_basename = 'lichess_db_puzzle.csv'
    
    # Determine where to look for the dataset or cache files
    isc_mode = bool(args.dataset_id)
    
    if isc_mode:
        # Try to find dataset or cache files in ISC data directory
        processed_dir = f'/data/{args.dataset_id}'
        dataset_path = f'{processed_dir}/{csv_basename}'
    else:
        # Try local dataset directory
        processed_dir = 'dataset'
        dataset_path = f'{processed_dir}/{csv_basename}'
        
        # Also check processed_lichess_puzzle_files as fallback
        fallback_dir = 'processed_lichess_puzzle_files'
    
    # Check if CSV exists
    csv_exists = os.path.exists(dataset_path)
    
    # If CSV doesn't exist, check for cache files
    if not csv_exists:
        if args.device_id == 0:
            print(f"⚠️ CSV file {dataset_path} not found. Checking for cache files...")
        
        # Define critical cache files needed for operation
        essential_cache_files = [
            f"{csv_basename}.themes.json",
            f"{csv_basename}.openings.json"
        ]
        
        # Files specifically needed for class conditional augmentation
        conditional_aug_files = [
            f"{csv_basename}.tensors.pt_conditional",
            f"{csv_basename}.tensors.pt_conditional.augmented_indices.json",
            f"{csv_basename}.cooccurrence.json"
        ]
        
        # Check if essential files exist in the current directory
        essential_files_exist = all(
            os.path.exists(os.path.join(processed_dir, f)) for f in essential_cache_files
        )
        
        conditional_files_exist = all(
            os.path.exists(os.path.join(processed_dir, f)) for f in conditional_aug_files
        )
        
        # If not using ISC mode, check fallback directory
        if not isc_mode and not essential_files_exist:
            if args.device_id == 0:
                print(f"Checking fallback directory: {fallback_dir}")
            
            fallback_essential_exist = os.path.exists(fallback_dir) and all(
                os.path.exists(os.path.join(fallback_dir, f)) for f in essential_cache_files
            )
            
            fallback_conditional_exist = os.path.exists(fallback_dir) and all(
                os.path.exists(os.path.join(fallback_dir, f)) for f in conditional_aug_files
            )
            
            if fallback_essential_exist:
                if args.device_id == 0:
                    print(f"✅ Found essential files in fallback directory {fallback_dir}")
                processed_dir = fallback_dir
                dataset_path = os.path.join(fallback_dir, csv_basename)
                essential_files_exist = True
                conditional_files_exist = fallback_conditional_exist
        
        # If we have essential files but not conditional files
        if essential_files_exist and not conditional_files_exist:
            if args.device_id == 0:
                print(f"⚠️ Class conditional augmentation files missing, but essential files exist")
                print(f"⚠️ Will attempt to use simpler augmentation or no augmentation")
                
        # Proceed if we have at least the essential files
        if essential_files_exist:
            if args.device_id == 0:
                print(f"✅ Using cache files from: {processed_dir}")
                print(f"✅ Setting dataset path to: {dataset_path}")
        else:
            if args.device_id == 0:
                print(f"❌ Essential cache files not found in any location")
                missing_files = [f for f in essential_cache_files 
                                if not os.path.exists(os.path.join(processed_dir, f))]
                print(f"Missing files: {', '.join(missing_files)}")
            raise FileNotFoundError(
                f"CSV file {dataset_path} not found and essential cache files not available. "
                "Please provide either the CSV file or the pre-processed cache files."
            )
    
    # Initialize dataset with appropriate settings
    try:
        # First try with class conditional augmentation (preferred)
        dataset = ChessPuzzleDataset(dataset_path, class_conditional_augmentation=True, low_memory=True)
        if args.device_id == 0:
            print("Using dataset with class conditional augmentation")
    except Exception as e:
        # If that fails, try without class conditional augmentation
        if args.device_id == 0:
            print(f"⚠️ Error with class conditional augmentation: {e}")
            print(f"⚠️ Trying without class conditional augmentation")
        try:
            dataset = ChessPuzzleDataset(dataset_path, class_conditional_augmentation=False, low_memory=True)
            if args.device_id == 0:
                print("Using dataset without class conditional augmentation")
        except Exception as e2:
            # Last resort: try with minimal features
            if args.device_id == 0:
                print(f"⚠️ Error initializing dataset: {e2}")
                print(f"⚠️ Attempting with minimal feature set")
            dataset = ChessPuzzleDataset(dataset_path, class_conditional_augmentation=False, 
                                         low_memory=True, augment_with_reflections=False)
    
    if args.device_id == 0:
        if args.test_mode:
            print(f"Running in test mode with smaller dataset")
        else:
            print(f"Using full dataset")
    
    # Get the number of labels from the dataset
    num_labels = len(dataset.all_labels)
    if args.device_id == 0:
        print(f"Number of unique labels (themes + opening tags): {num_labels}")

    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)
    timer.report(f"Initialized datasets with {len(train_dataset):,} training and {len(test_dataset):,} test samples.")

    train_sampler = InterruptableDistributedSampler(train_dataset)
    test_sampler = InterruptableDistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, sampler=test_sampler)
    timer.report("Prepared dataloaders")

    # Load model config or use default values
    model_config = {}
    if os.path.exists(args.model_config):
        model_config = yaml.safe_load(open(args.model_config))
        
        # We'll use the full model architecture even with smaller datasets
        if "num_labels" in model_config:
            if model_config["num_labels"] < num_labels:
                # If config has fewer labels than dataset, that's a problem
                if args.device_id == 0:
                    print(f"⚠️  Warning: num_labels in model_config ({model_config['num_labels']}) is less than dataset ({num_labels})")
                    print(f"⚠️  Increasing num_labels to match dataset: {num_labels}")
                model_config["num_labels"] = num_labels
            elif model_config["num_labels"] > num_labels:
                # Using larger model (from full dataset) with smaller dataset (test mode)
                if args.device_id == 0:
                    print(f"ℹ️  Using full model architecture with {model_config['num_labels']} output labels")
                    print(f"ℹ️  Current dataset only has {num_labels} labels (likely using test dataset)")
                    print(f"ℹ️  Extra outputs will be ignored during training")
        else:
            # No num_labels in config, use the dataset's value
            model_config["num_labels"] = num_labels
            
        if args.device_id == 0:
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
        if args.device_id == 0:
            print(f"Using default ModelConfig: {model_config}")
    
    model = Model(**model_config)
    model = model.to(args.device_id)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    timer.report(f"Initialized model with {params:,} params, moved to device")

    model = DDP(model, device_ids=[args.device_id])
    timer.report("Prepared model for distributed training")

    # Try to use weighted loss if class weight file exists
    try:
        from class_weights import compute_label_weights
        
        if args.device_id == 0:
            print("Attempting to use weighted BCE loss for class imbalance...")
            
        # Try to compute or load class weights
        pos_weights = compute_label_weights(dataset)
        pos_weights = pos_weights.to(args.device_id)
        
        # If we succeed, use weighted loss
        if args.device_id == 0:
            print("✅ Using weighted BCE loss for class imbalance mitigation")
        
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="sum")
    except Exception as e:
        # If weights can't be computed or loaded, fall back to standard BCE loss
        if args.device_id == 0:
            print(f"⚠️ Could not load or compute class weights: {e}")
            print("⚠️ Using standard BCE loss (no class weights)")
        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}

    if args.is_master:
        tensorboard_log_dir=os.environ["LOSSY_ARTIFACT_PATH"]
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f"Tensorboard logs saved to {tensorboard_log_dir}")

    checkpoint_path = None
    local_resume_path = os.path.join(args.save_dir, saver.symlink_name)
    if os.path.islink(local_resume_path):
        checkpoint = os.path.join(os.readlink(local_resume_path), "checkpoint.pt")
        if os.path.isfile(checkpoint):
            checkpoint_path = checkpoint
    elif args.load_path:
        if os.path.isfile(args.load_path):
            checkpoint_path = args.load_path
    if checkpoint_path:
        if args.is_master:
            timer.report(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer = checkpoint["timer"]
        timer.start_time = time.time()
        timer.report("Retrieved saved checkpoint")

    # Use fewer epochs in test mode
    max_epochs = 3 if args.test_mode else 10
    for epoch in range(train_dataloader.sampler.epoch, max_epochs):
        with train_dataloader.sampler.in_epoch(epoch):
            timer.report(f"Training epoch {epoch}")
            train_batches_per_epoch = len(train_dataloader)
            train_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum)
            optimizer.zero_grad()
            model.train()

            for batch_data in train_dataloader:
                # Unpack batch data and move to device
                boards = batch_data['board'].unsqueeze(1)  # Add channel dimension
                labels = batch_data['themes']
                boards, labels = boards.to(args.device_id), labels.to(args.device_id)

                # Determine the current step
                batch = train_dataloader.sampler.progress // train_dataloader.batch_size
                is_save_batch = (batch + 1) % args.save_steps == 0
                is_accum_batch = (batch + 1) % args.grad_accum == 0
                is_last_batch = (batch + 1) == train_batches_per_epoch

                # Prepare checkpoint directory
                if (is_save_batch or is_last_batch) and args.is_master:
                    checkpoint_directory = saver.prepare_checkpoint_directory()

                logits = model(boards)
                
                # Handle shape issues when using full model with smaller dataset
                if logits.shape[1] > labels.shape[1]:
                    # We're using a model with more output labels than in the current dataset
                    if batch == 0 and args.device_id == 0:
                        print(f"Model has more outputs ({logits.shape[1]}) than dataset labels ({labels.shape[1]})")
                        print(f"Using only the first {labels.shape[1]} outputs for loss calculation")
                    
                    # Only use the first part of the outputs that match the dataset's labels
                    logits_for_loss = logits[:, :labels.shape[1]]
                else:
                    # Shapes match exactly
                    logits_for_loss = logits
                
                loss = loss_fn(logits_for_loss, labels) / args.grad_accum

                loss.backward()
                train_dataloader.sampler.advance(len(boards))

                # Calculate jaccard similarity for this batch - use the same subset of outputs
                output_probs = torch.sigmoid(logits_for_loss)
                jaccard = jaccard_similarity(output_probs, labels, threshold=0.5)

                # Calculate detailed metrics every 100 steps to avoid overhead
                calculate_detailed_metrics = (batch % 100 == 0)
                metrics_dict = {}
                
                if calculate_detailed_metrics:
                    # Calculate precision, recall, F1 with different averaging methods
                    # Turn on verbose mode for the first calculation in each epoch
                    first_in_epoch = (batch == 0 and epoch == train_dataloader.sampler.epoch)
                    
                    # First batch of each epoch uses verbose mode to show progress
                    precision_micro, recall_micro, f1_micro = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='micro', verbose=first_in_epoch if args.is_master else False
                    )
                    precision_macro, recall_macro, f1_macro = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='macro', verbose=first_in_epoch if args.is_master else False
                    )
                    precision_weighted, recall_weighted, f1_weighted = precision_recall_f1(
                        output_probs, labels, threshold=0.5, average='weighted', verbose=first_in_epoch if args.is_master else False
                    )
                    
                    # Create metrics dictionary exactly as in train.py
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
                    if batch % 1000 == 0:
                        # Get label names for the report
                        label_names = dataset.all_labels
                        report = get_classification_report(
                            output_probs, labels, threshold=0.5, labels=label_names,
                            verbose=first_in_epoch  # Verbose on first report
                        )
                        print("\nClassification Report:")
                        print(report)
                        if args.is_master:
                            writer.add_text('Classification Report', report, batch + epoch * train_batches_per_epoch)

                # Update basic metrics
                metrics["train"].update({
                    "examples_seen": len(boards),
                    "accum_loss": loss.item() * args.grad_accum,  # undo loss scale
                    "jaccard": jaccard.item()
                })
                
                # Store detailed metrics properly for distributed reduction
                if calculate_detailed_metrics:
                    # Update metrics dictionary with detailed metrics for proper reduction
                    metrics_update = {}
                    for k, v in metrics_dict.items():
                        metrics_update[k] = v
                    
                    # Use update method to ensure proper handling in distributed settings
                    metrics["train"].update(metrics_update)

                if is_accum_batch or is_last_batch:
                    optimizer.step()
                    optimizer.zero_grad()
                    step = batch // args.grad_accum
                    
                    # learning rate warmup
                    lr_factor = min((epoch * train_steps_per_epoch + step) / args.ws, 1)
                    next_lr = lr_factor * args.lr
                    for g in optimizer.param_groups:
                        g['lr'] = next_lr
                    
                    metrics["train"].reduce()
                    rpt = metrics["train"].local
                    avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                    rpt_jaccard = 100 * rpt["jaccard"] / ((batch % args.grad_accum + 1) * args.world_size)
                    
                    # Add detailed metrics to report if available
                    metrics_report = ""
                    if "f1_micro" in rpt:
                        metrics_report = f" F1-micro: {rpt['f1_micro']:.3f}, F1-macro: {rpt['f1_macro']:.3f}"
                    
                    report = f"""\
Epoch [{epoch:,}] Step [{step:,} / {train_steps_per_epoch:,}] Batch [{batch:,} / {train_batches_per_epoch:,}] Lr: [{lr_factor * args.lr:,.3}], \
Avg Loss [{avg_loss:,.3f}], Jaccard: [{rpt_jaccard:,.3f}%]{metrics_report}, Examples: {rpt['examples_seen']:,.0f}"""
                    timer.report(report)
                    metrics["train"].reset_local()

                    if args.is_master:
                        total_progress = batch + epoch * train_batches_per_epoch
                        # Match train.py logging format exactly
                        writer.add_scalar("Loss/train", avg_loss, total_progress)
                        writer.add_scalar("Jaccard/train", rpt_jaccard, total_progress)
                        writer.add_scalar("Learning_rate", next_lr, total_progress)
                        
                        # Also keep original metrics for backward compatibility
                        writer.add_scalar("train/learn_rate", next_lr, total_progress)
                        writer.add_scalar("train/loss", avg_loss, total_progress)
                        writer.add_scalar("train/batch_jaccard", rpt_jaccard, total_progress)
                        
                        # Log all available metrics, prioritizing values from reduced metrics
                        # These are the metrics we want to track
                        metric_names = ["precision_micro", "recall_micro", "f1_micro", 
                                       "precision_macro", "recall_macro", "f1_macro",
                                       "precision_weighted", "recall_weighted", "f1_weighted"]
                        
                        # First log metrics from the reduced metrics dictionary (these are properly synced across nodes)
                        for metric_name in metric_names:
                            if metric_name in rpt:
                                # Found in reduced metrics - use this value
                                metric_value = rpt[metric_name]
                                writer.add_scalar(f'Metrics/{metric_name}', metric_value, total_progress)
                            elif metric_name in metrics_dict:
                                # Not in reduced metrics but in local metrics_dict - use as fallback
                                metric_value = metrics_dict[metric_name]
                                writer.add_scalar(f'Metrics/{metric_name}', metric_value, total_progress)
                                
                        # Log any additional metrics not in our standard list
                        for metric_name, metric_value in metrics_dict.items():
                            if metric_name not in metric_names:
                                writer.add_scalar(f'Metrics/{metric_name}', metric_value, total_progress)

                # Saving
                if (is_save_batch or is_last_batch) and args.is_master:
                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    # Save checkpoint with more detailed information
                    checkpoint_data = {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_sampler": train_dataloader.sampler.state_dict(),
                        "test_sampler": test_dataloader.sampler.state_dict(),
                        "metrics": metrics,
                        "timer": timer,
                        "loss": loss.item(),
                        "jaccard_loss": jaccard.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "global_step": batch + epoch * train_batches_per_epoch
                    }
                    
                    # Add any detailed metrics if available
                    if metrics_dict:
                        for k, v in metrics_dict.items():
                            checkpoint_data[k] = v
                    
                    # Save the checkpoint
                    atomic_torch_save(
                        checkpoint_data,
                        os.path.join(checkpoint_directory, "checkpoint.pt"),
                    )
                    saver.symlink_latest(checkpoint_directory)

        with test_dataloader.sampler.in_epoch(epoch):
            timer.report(f"Testing epoch {epoch}")
            test_batches_per_epoch = len(test_dataloader)
            model.eval()

            with torch.no_grad():
                for batch_data in test_dataloader:
                    # Unpack batch data and move to device
                    boards = batch_data['board'].unsqueeze(1)  # Add channel dimension
                    labels = batch_data['themes']
                    boards, labels = boards.to(args.device_id), labels.to(args.device_id)

                    # Determine the current step
                    batch = test_dataloader.sampler.progress // test_dataloader.batch_size
                    is_save_batch = (batch + 1) % args.save_steps == 0
                    is_last_batch = (batch + 1) == test_batches_per_epoch

                    # Prepare checkpoint directory
                    if (is_save_batch or is_last_batch) and args.is_master:
                        checkpoint_directory = saver.prepare_checkpoint_directory()

                    logits = model(boards)
                    
                    # Handle shape issues when using full model with smaller dataset
                    if logits.shape[1] > labels.shape[1]:
                        # We're using a model with more output labels than in the current dataset
                        if batch == 0 and is_last_batch and args.device_id == 0:
                            print(f"Test phase: Model has more outputs ({logits.shape[1]}) than dataset labels ({labels.shape[1]})")
                            print(f"Using only the first {labels.shape[1]} outputs for loss calculation")
                        
                        # Only use the first part of the outputs that match the dataset's labels
                        logits_for_loss = logits[:, :labels.shape[1]]
                    else:
                        # Shapes match exactly
                        logits_for_loss = logits
                    
                    loss = loss_fn(logits_for_loss, labels)
                    test_dataloader.sampler.advance(len(boards))

                    # Calculate jaccard similarity for this batch - use the same subset of outputs
                    output_probs = torch.sigmoid(logits_for_loss)
                    jaccard = jaccard_similarity(output_probs, labels, threshold=0.5)
                    
                    # Calculate detailed metrics on the last batch only to avoid overhead
                    metrics_dict = {}
                    if is_last_batch:
                        # Calculate precision, recall, F1 with different averaging methods
                        precision_micro, recall_micro, f1_micro = precision_recall_f1(
                            output_probs, labels, threshold=0.5, average='micro', verbose=args.is_master
                        )
                        precision_macro, recall_macro, f1_macro = precision_recall_f1(
                            output_probs, labels, threshold=0.5, average='macro', verbose=args.is_master
                        )
                        precision_weighted, recall_weighted, f1_weighted = precision_recall_f1(
                            output_probs, labels, threshold=0.5, average='weighted', verbose=args.is_master
                        )
                        
                        # Create metrics dictionary exactly as in train.py
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
                        
                        # Generate classification report for the test set
                        label_names = dataset.all_labels
                        report = get_classification_report(
                            output_probs, labels, threshold=0.5, labels=label_names
                        )
                        print("\nTest Classification Report:")
                        print(report)
                        writer.add_text('Test Classification Report', report, epoch)

                    metrics["test"].update({
                        "examples_seen": len(boards),
                        "accum_loss": loss.item(),
                        "jaccard": jaccard.item()
                    })
                    
                    # Store detailed metrics properly for distributed reduction in test phase
                    if metrics_dict:
                        # Update metrics dictionary with detailed metrics for proper reduction
                        metrics_update = {}
                        for k, v in metrics_dict.items():
                            metrics_update[k] = v
                        
                        # Use update method to ensure proper handling in distributed settings
                        metrics["test"].update(metrics_update)
                    
                    # Reporting
                    if is_last_batch:
                        metrics["test"].reduce()
                        rpt = metrics["test"].local
                        avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                        rpt_jaccard = 100 * rpt["jaccard"] / (test_batches_per_epoch * args.world_size)
                        
                        # Add detailed metrics to report if available
                        metrics_report = ""
                        if "f1_micro" in rpt:
                            metrics_report = f", F1-micro: {rpt['f1_micro']:.3f}, F1-macro: {rpt['f1_macro']:.3f}"
                        
                        report = f"Epoch [{epoch}] Evaluation, Avg Loss [{avg_loss:,.3f}], Jaccard [{rpt_jaccard:,.3f}%]{metrics_report}"
                        timer.report(report)
                        metrics["test"].reset_local()

                        if args.is_master:
                            # Match train.py logging format exactly
                            writer.add_scalar("Loss/test", avg_loss, epoch)
                            writer.add_scalar("Jaccard/test", rpt_jaccard, epoch)
                            
                            # Also keep original metrics for backward compatibility
                            writer.add_scalar("test/loss", avg_loss, epoch)
                            writer.add_scalar("test/batch_jaccard", rpt_jaccard, epoch)
                            
                            # Log all available metrics for test phase, prioritizing values from reduced metrics
                            # These are the metrics we want to track
                            metric_names = ["precision_micro", "recall_micro", "f1_micro", 
                                           "precision_macro", "recall_macro", "f1_macro",
                                           "precision_weighted", "recall_weighted", "f1_weighted"]
                            
                            # First log metrics from the reduced metrics dictionary (these are properly synced across nodes)
                            for metric_name in metric_names:
                                if metric_name in rpt:
                                    # Found in reduced metrics - use this value
                                    metric_value = rpt[metric_name]
                                    writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                                    writer.add_scalar(f'Metrics/test_{metric_name}', metric_value, epoch)
                                elif metric_name in metrics_dict:
                                    # Not in reduced metrics but in local metrics_dict - use as fallback
                                    metric_value = metrics_dict[metric_name]
                                    writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                                    writer.add_scalar(f'Metrics/test_{metric_name}', metric_value, epoch)
                                    
                            # Log any additional metrics not in our standard list
                            for metric_name, metric_value in metrics_dict.items():
                                if metric_name not in metric_names:
                                    writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                                    writer.add_scalar(f'Metrics/test_{metric_name}', metric_value, epoch)
                    
                    # Saving
                    if (is_save_batch or is_last_batch) and args.is_master:
                        timer.report(f"Saving after test batch [{batch} / {test_batches_per_epoch}]")
                        
                        # Get current learning rate
                        current_lr = optimizer.param_groups[0]['lr']
                        
                        # Save checkpoint with more detailed information
                        checkpoint_data = {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "metrics": metrics,
                            "timer": timer,
                            "loss": loss.item(),
                            "jaccard_loss": jaccard.item(),
                            "learning_rate": current_lr,
                            "epoch": epoch,
                            "global_step": batch + epoch * test_batches_per_epoch,
                            "is_test": True  # Mark as test checkpoint
                        }
                        
                        # Add any detailed metrics if available
                        if metrics_dict:
                            for k, v in metrics_dict.items():
                                checkpoint_data[k] = v
                        
                        # Save the checkpoint
                        atomic_torch_save(
                            checkpoint_data,
                            os.path.join(checkpoint_directory, "checkpoint.pt"),
                        )
                        saver.symlink_latest(checkpoint_directory)


timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)