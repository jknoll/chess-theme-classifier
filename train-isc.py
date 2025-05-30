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
from metrics import jaccard_similarity

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
        if args.dataset_id:
            dataset_path = f'/data/{args.dataset_id}/lichess_db_puzzle_test.csv'
        else:
            dataset_path = 'lichess_db_puzzle_test.csv'
        dataset = ChessPuzzleDataset(dataset_path)
        if args.device_id == 0:
            print(f"Running in test mode with smaller dataset: {dataset_path}")
    else:
        if args.dataset_id:
            dataset_path = f'/data/{args.dataset_id}/lichess_db_puzzle.csv'
        else:
            dataset_path = 'lichess_db_puzzle.csv'
        dataset = ChessPuzzleDataset(dataset_path)
        if args.device_id == 0:
            print(f"Using dataset: {dataset_path}")
    
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
        if args.device_id == 0:
            print(f"ModelConfig: {model_config}")
    else:
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

    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}

    if args.is_master:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tb"))

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
                loss = loss_fn(logits, labels) / args.grad_accum

                loss.backward()
                train_dataloader.sampler.advance(len(boards))

                # Calculate jaccard similarity for this batch
                output_probs = torch.sigmoid(logits)
                jaccard = jaccard_similarity(output_probs, labels, threshold=0.5)

                metrics["train"].update({
                    "examples_seen": len(boards),
                    "accum_loss": loss.item() * args.grad_accum,  # undo loss scale
                    "jaccard": jaccard.item()
                })

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
                    report = f"""\
Epoch [{epoch:,}] Step [{step:,} / {train_steps_per_epoch:,}] Batch [{batch:,} / {train_batches_per_epoch:,}] Lr: [{lr_factor * args.lr:,.3}], \
Avg Loss [{avg_loss:,.3f}], Jaccard: [{rpt_jaccard:,.3f}%], Examples: {rpt['examples_seen']:,.0f}"""
                    timer.report(report)
                    metrics["train"].reset_local()

                    if args.is_master:
                        total_progress = batch + epoch * train_batches_per_epoch
                        writer.add_scalar("train/learn_rate", next_lr, total_progress)
                        writer.add_scalar("train/loss", avg_loss, total_progress)
                        writer.add_scalar("train/batch_jaccard", rpt_jaccard, total_progress)

                # Saving
                if (is_save_batch or is_last_batch) and args.is_master:
                    # Save checkpoint
                    atomic_torch_save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "metrics": metrics,
                            "timer": timer
                        },
                        os.path.join(checkpoint_directory, "checkpoint.pt"),
                    )
                    saver.atomic_symlink(checkpoint_directory)

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
                    loss = loss_fn(logits, labels)
                    test_dataloader.sampler.advance(len(boards))

                    # Calculate jaccard similarity for this batch
                    output_probs = torch.sigmoid(logits)
                    jaccard = jaccard_similarity(output_probs, labels, threshold=0.5)

                    metrics["test"].update({
                        "examples_seen": len(boards),
                        "accum_loss": loss.item(),
                        "jaccard": jaccard.item()
                    })
                    
                    # Reporting
                    if is_last_batch:
                        metrics["test"].reduce()
                        rpt = metrics["test"].local
                        avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                        rpt_jaccard = 100 * rpt["jaccard"] / (test_batches_per_epoch * args.world_size)
                        report = f"Epoch [{epoch}] Evaluation, Avg Loss [{avg_loss:,.3f}], Jaccard [{rpt_jaccard:,.3f}%]"
                        timer.report(report)
                        metrics["test"].reset_local()

                        if args.is_master:
                            writer.add_scalar("test/loss", avg_loss, epoch)
                            writer.add_scalar("test/batch_jaccard", rpt_jaccard, epoch)
                    
                    # Saving
                    if (is_save_batch or is_last_batch) and args.is_master:
                        timer.report(f"Saving after test batch [{batch} / {test_batches_per_epoch}]")
                        # Save checkpoint
                        atomic_torch_save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "train_sampler": train_dataloader.sampler.state_dict(),
                                "test_sampler": test_dataloader.sampler.state_dict(),
                                "metrics": metrics,
                                "timer": timer
                            },
                            os.path.join(checkpoint_directory, "checkpoint.pt"),
                        )
                        saver.atomic_symlink(checkpoint_directory)


timer.report("Defined functions")
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)