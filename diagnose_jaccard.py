#!/usr/bin/env python3
import os
import torch
import argparse
import sys
from dataset import ChessPuzzleDataset
from torch.utils.data import DataLoader, random_split
from model import Model
from metrics import jaccard_similarity
import yaml

# Skip most of the training setup, just load model and data to diagnose Jaccard

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Diagnose Jaccard similarity issues')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with a smaller dataset')
    parser.add_argument('--model-config', help="model config path", type=str, default="/root/chess-theme-classifier/model_config.yaml")          
    args = parser.parse_args()
    
    # Select the appropriate dataset file
    if args.test_mode:
        csv_filename = 'lichess_db_puzzle_test.csv'
        # Use the local path in the dataset directory
        csv_file = os.path.join('dataset', csv_filename)
    else:
        # Use processed directory for full dataset 
        csv_filename = 'lichess_db_puzzle.csv'
        csv_file = os.path.join('processed_lichess_puzzle_files', csv_filename)
    
    print(f"Using dataset: {csv_file}")
    
    # Load dataset
    dataset = ChessPuzzleDataset(csv_file, class_conditional_augmentation=True, low_memory=True)
    
    # Get the number of labels from the dataset
    num_labels = len(dataset.all_labels)
    print(f"Number of unique labels (themes + opening tags): {num_labels}")
    
    # Load model config or use default values
    model_config = {}
    if os.path.exists(args.model_config):
        with open(args.model_config, 'r') as f:
            model_config = yaml.safe_load(f)
        model_config["num_labels"] = num_labels
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
    
    # Use CPU to make diagnostics easier
    device = torch.device('cpu')
    
    # Create model
    model = Model(**model_config)
    model = model.to(device)
    
    # Split the dataset 
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)
    
    # Create dataloader with small batch size
    batch_size = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Get just one batch for diagnostics
    for data in train_dataloader:
        inputs = data['board']
        inputs = inputs.unsqueeze(1).to(device)
        labels = data['themes'].to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
        
        # Apply sigmoid to get probabilities
        output_probs = torch.sigmoid(outputs)
        
        # Print detailed information about the batch
        print("\n----- BATCH STATISTICS -----")
        print(f"Batch size: {inputs.size(0)}")
        print(f"Number of labels: {labels.size(1)}")
        print(f"Output probabilities: min={output_probs.min().item():.6f}, max={output_probs.max().item():.6f}, mean={output_probs.mean().item():.6f}")
        print(f"Labels: min={labels.min().item():.6f}, max={labels.max().item():.6f}, mean={labels.mean().item():.6f}")
        
        # Count positive labels
        positive_labels = labels.sum().item()
        total_labels = labels.numel()
        print(f"Positive labels: {positive_labels}/{total_labels} = {positive_labels/total_labels:.6f}")
        
        # Calculate thresholds that would be used
        print("\n----- ADAPTIVE THRESHOLD CALCULATION -----")
        pred_mean = output_probs.mean(dim=0)
        pred_std = output_probs.std(dim=0)
        
        # Show range of means and standard deviations
        print(f"Prediction means: min={pred_mean.min().item():.6f}, max={pred_mean.max().item():.6f}, avg={pred_mean.mean().item():.6f}")
        print(f"Prediction std devs: min={pred_std.min().item():.6f}, max={pred_std.max().item():.6f}, avg={pred_std.mean().item():.6f}")
        
        # Calculate thresholds using three different formulas
        class_thresholds1 = torch.clamp(pred_mean - 0.5 * pred_std, min=0.05, max=0.5)
        class_thresholds2 = torch.clamp(pred_mean - 0.8 * pred_std, min=0.05, max=0.5)
        class_thresholds3 = torch.clamp(pred_mean - 1.0 * pred_std, min=0.01, max=0.5)
        
        print(f"Thresholds (0.5 * std): min={class_thresholds1.min().item():.6f}, max={class_thresholds1.max().item():.6f}, avg={class_thresholds1.mean().item():.6f}")
        print(f"Thresholds (0.8 * std): min={class_thresholds2.min().item():.6f}, max={class_thresholds2.max().item():.6f}, avg={class_thresholds2.mean().item():.6f}")
        print(f"Thresholds (1.0 * std): min={class_thresholds3.min().item():.6f}, max={class_thresholds3.max().item():.6f}, avg={class_thresholds3.mean().item():.6f}")
        
        # Apply each threshold and count resulting positive predictions
        pred_binary1 = torch.zeros_like(output_probs, dtype=torch.bool)
        pred_binary2 = torch.zeros_like(output_probs, dtype=torch.bool)
        pred_binary3 = torch.zeros_like(output_probs, dtype=torch.bool)
        
        for i in range(output_probs.shape[1]):
            pred_binary1[:, i] = (output_probs[:, i] > class_thresholds1[i])
            pred_binary2[:, i] = (output_probs[:, i] > class_thresholds2[i])
            pred_binary3[:, i] = (output_probs[:, i] > class_thresholds3[i])
        
        # Count positive predictions for each threshold
        pos_preds1 = pred_binary1.sum().item()
        pos_preds2 = pred_binary2.sum().item()
        pos_preds3 = pred_binary3.sum().item()
        
        print(f"Positive predictions (0.5 * std): {pos_preds1}/{total_labels} = {pos_preds1/total_labels:.6f}")
        print(f"Positive predictions (0.8 * std): {pos_preds2}/{total_labels} = {pos_preds2/total_labels:.6f}")
        print(f"Positive predictions (1.0 * std): {pos_preds3}/{total_labels} = {pos_preds3/total_labels:.6f}")
        
        # Calculate intersection and union for each threshold
        print("\n----- JACCARD SIMILARITY CALCULATION -----")
        
        # Manual calculation first
        intersection1 = torch.logical_and(pred_binary1, labels).sum().item()
        union1 = torch.logical_or(pred_binary1, labels).sum().item()
        jaccard1_manual = intersection1 / (union1 + 1e-8)
        
        intersection2 = torch.logical_and(pred_binary2, labels).sum().item()
        union2 = torch.logical_or(pred_binary2, labels).sum().item()
        jaccard2_manual = intersection2 / (union2 + 1e-8)
        
        intersection3 = torch.logical_and(pred_binary3, labels).sum().item()
        union3 = torch.logical_or(pred_binary3, labels).sum().item()
        jaccard3_manual = intersection3 / (union3 + 1e-8)
        
        # Now use our updated jaccard_similarity function
        print("\n----- USING UPDATED JACCARD FUNCTION -----")
        jaccard1_func = jaccard_similarity(output_probs, labels, threshold=0.5, adaptive_threshold=True, verbose=True).item()
        
        print(f"\n----- MANUAL CALCULATION RESULTS -----")
        print(f"Jaccard (0.5 * std): intersection={intersection1}, union={union1}, similarity={jaccard1_manual:.6f}")
        print(f"Jaccard (0.8 * std): intersection={intersection2}, union={union2}, similarity={jaccard2_manual:.6f}")
        print(f"Jaccard (1.0 * std): intersection={intersection3}, union={union3}, similarity={jaccard3_manual:.6f}")
        
        print(f"\n----- FIXED FUNCTION RESULT -----")
        print(f"Fixed Jaccard similarity: {jaccard1_func:.6f}")
        
        # Check if any predictions match any targets
        any_matches1 = (intersection1 > 0)
        any_matches2 = (intersection2 > 0)
        any_matches3 = (intersection3 > 0)
        
        print(f"Any matches (0.5 * std): {any_matches1}")
        print(f"Any matches (0.8 * std): {any_matches2}")
        print(f"Any matches (1.0 * std): {any_matches3}")
        
        # See if we can find any threshold that produces matches
        if not any_matches3:
            print("\n----- EXTREME THRESHOLD TESTING -----")
            # Try an extremely low threshold to see if any predictions match targets
            extreme_threshold = 0.001
            pred_binary_extreme = (output_probs > extreme_threshold)
            intersection_extreme = torch.logical_and(pred_binary_extreme, labels).sum().item()
            union_extreme = torch.logical_or(pred_binary_extreme, labels).sum().item()
            jaccard_extreme = intersection_extreme / (union_extreme + 1e-8)
            
            print(f"Extreme low threshold ({extreme_threshold}): intersection={intersection_extreme}, union={union_extreme}, similarity={jaccard_extreme:.6f}")
            
            # Count predictions above extreme threshold
            pos_preds_extreme = pred_binary_extreme.sum().item()
            print(f"Positive predictions (extreme): {pos_preds_extreme}/{total_labels} = {pos_preds_extreme/total_labels:.6f}")
        
        # Check prediction and label distribution in more detail
        print("\n----- DISTRIBUTION ANALYSIS -----")
        # Count number of predictions in different ranges
        ranges = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for low, high in ranges:
            count = ((output_probs >= low) & (output_probs < high)).sum().item()
            percent = count / output_probs.numel() * 100
            print(f"Predictions in [{low:.1f}, {high:.1f}): {count} ({percent:.2f}%)")
        
        # Find the highest predicted probabilities and their corresponding labels
        print("\n----- TOP PREDICTIONS VS ACTUAL -----")
        # For each sample, find the top 5 predictions
        for i in range(min(2, batch_size)):
            probs = output_probs[i]
            label = labels[i]
            
            # Get indices of top 5 predictions
            _, top_indices = torch.topk(probs, min(5, probs.size(0)))
            
            print(f"\nSample {i} - Top predictions:")
            for idx in top_indices:
                is_true = "TRUE" if label[idx] > 0 else "false"
                print(f"  Label {idx}: prob={probs[idx].item():.6f} ({is_true})")
            
            # Find any true labels that weren't in top predictions
            true_indices = torch.where(label > 0)[0]
            print(f"  True labels: {true_indices.tolist()}")
            
            # For each true label, show its prediction
            for idx in true_indices:
                print(f"  True label {idx}: predicted prob={probs[idx].item():.6f}")
        
        # Only process one batch for diagnostics
        break

if __name__ == "__main__":
    main()