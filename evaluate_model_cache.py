import torch
import numpy as np
import argparse
import time
import os
from dataset import ChessPuzzleDataset
from model import Model
from metrics import compute_multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Test chess puzzle classifier using cached data')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to test (default: 20)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold for classification (default: None, uses adaptive thresholding)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_resume.pth",
                        help='Checkpoint file to use for testing')
    parser.add_argument('--max_themes', type=int, default=50,
                        help='Maximum number of themes to include in the confusion matrix (default: 50)')
    parser.add_argument('--dataset_path', type=str, default="dataset/lichess_db_puzzle.csv",
                        help='Path to the dataset file (default: dataset/lichess_db_puzzle.csv)')
    return parser.parse_args()

def calculate_adaptive_threshold(outputs, targets):
    """
    Calculate optimal threshold based on F1 score
    
    Args:
        outputs: Tensor of predicted probabilities
        targets: Tensor of ground truth labels
    
    Returns:
        Best threshold value
    """
    from sklearn.metrics import f1_score
    
    # Flatten for binary classification metrics and convert to numpy
    flat_outputs = outputs.reshape(-1).cpu().numpy()
    flat_targets = targets.reshape(-1).cpu().numpy()
    
    # Print statistics
    print("\nAdaptive Threshold Calculation:")
    print(f"Using {len(outputs)} samples")
    print(f"Positive targets: {flat_targets.sum()}/{len(flat_targets)} ({flat_targets.sum()/len(flat_targets)*100:.2f}%)")
    print(f"Output stats - min: {flat_outputs.min():.4f}, max: {flat_outputs.max():.4f}, mean: {flat_outputs.mean():.4f}")
    
    # Test a range of thresholds
    thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
    
    best_f1 = 0
    best_threshold = 0.3  # Default
    
    print("\nFinding optimal threshold:")
    print(f"{'Threshold':>10} {'F1 Score':>10} {'Precision':>10} {'Recall':>10} {'Positives':>10}")
    print("-" * 55)
    
    for threshold in thresholds:
        # Convert to binary predictions
        binary_preds = (flat_outputs > threshold).astype(int)
        
        # Calculate metrics
        f1 = f1_score(flat_targets, binary_preds, average='macro', zero_division=0)
        precision = (binary_preds * flat_targets).sum() / binary_preds.sum() if binary_preds.sum() > 0 else 0
        recall = (binary_preds * flat_targets).sum() / flat_targets.sum() if flat_targets.sum() > 0 else 0
        positives_pct = binary_preds.sum() / len(binary_preds) * 100
        
        print(f"{threshold:10.4f} {f1:10.4f} {precision:10.4f} {recall:10.4f} {positives_pct:9.2f}%")
        
        # Update best threshold
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nOptimal threshold: {best_threshold:.4f} (F1 score: {best_f1:.4f})")
    return best_threshold

def create_confusion_matrix(dataset, label_names, theme_counts, model, device, prediction_threshold, num_samples, max_themes):
    """Create confusion matrix from model predictions using all themes"""
    
    # Use all themes but limit by max_themes for visualization
    # Sort themes by frequency to prioritize common themes
    sorted_themes = sorted([(name, count) for name, count in theme_counts.items()], 
                          key=lambda x: x[1], reverse=True)
    
    # Filter to only include actual themes (not openings)
    theme_only_counts = {}
    for name, count in sorted_themes:
        if dataset.is_theme(name):
            theme_only_counts[name] = count
    
    # Limit to max_themes for visualization
    active_themes = list(theme_only_counts.keys())[:max_themes]
    print(f"Creating confusion matrix for top {len(active_themes)} themes (out of {len(theme_only_counts)} total themes)")
    
    # Create theme to index mapping
    theme_to_idx = {theme: idx for idx, theme in enumerate(active_themes)}
    
    # Initialize confusion matrix
    matrix = np.zeros((len(active_themes), len(active_themes)))
    
    # Process each sample
    for i in tqdm(range(num_samples), desc="Processing samples"):
        sample = dataset[i]
        target = sample['themes']
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
            out = model(input_tensor)
            predictions = torch.sigmoid(out).squeeze().cpu()
            
        # Ensure predictions and targets have the same size
        min_size = min(len(predictions), len(target))
        predictions = predictions[:min_size]
        
        # Get actual themes from the target tensor
        actual_theme_names = []
        for j, is_present in enumerate(target):
            if is_present == 1 and j < len(label_names):
                theme_name = label_names[j]
                if dataset.is_theme(theme_name) and theme_name in theme_to_idx:
                    actual_theme_names.append(theme_name)
        
        # Get predicted themes using the threshold
        pred_binary = (predictions > prediction_threshold).float()
        predicted_theme_names = []
        for j, is_predicted in enumerate(pred_binary):
            if is_predicted == 1 and j < len(label_names):
                theme_name = label_names[j]
                if dataset.is_theme(theme_name) and theme_name in theme_to_idx:
                    predicted_theme_names.append(theme_name)
        
        # For debugging, print the actual and predicted labels for the first few samples
        if i < 3:
            print(f"\nSample {i+1}:")
            print(f"FEN: {sample['fen'] if 'fen' in sample else 'N/A'}")
            print(f"Actual themes: {', '.join(actual_theme_names) if actual_theme_names else 'None'}")
            print(f"Predicted themes: {', '.join(predicted_theme_names) if predicted_theme_names else 'None'}")
            
            # Print top predicted values
            theme_probs = []
            for j, prob in enumerate(predictions):
                if j < len(label_names):
                    theme_name = label_names[j]
                    if dataset.is_theme(theme_name):
                        theme_probs.append((theme_name, prob.item()))
            
            # Print top 10 theme predictions
            print("\nTop theme predictions:")
            for theme_name, prob in sorted(theme_probs, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {theme_name}: {prob:.4f} > {prediction_threshold} = {prob > prediction_threshold}")
        
        # Update confusion matrix
        for actual in actual_theme_names:
            if actual in theme_to_idx:
                for predicted in predicted_theme_names:
                    if predicted in theme_to_idx:
                        matrix[theme_to_idx[actual], theme_to_idx[predicted]] += 1
    
    # Create visualization of the confusion matrix
    # Adjust figure size based on number of themes
    if len(active_themes) > 30:
        figsize = (24, 20)
        fontsize = 8
    elif len(active_themes) > 20:
        figsize = (20, 16)
        fontsize = 9
    else:
        figsize = (16, 12)
        fontsize = 10
    
    plt.figure(figsize=figsize)
    
    # Normalize matrix by rows (actual occurrences)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_sums, 
                                out=np.zeros_like(matrix), 
                                where=row_sums > 0)
    
    # Create heatmap
    sns.heatmap(normalized_matrix, 
                xticklabels=active_themes,
                yticklabels=active_themes,
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                annot=False,  # Too many themes for annotations
                square=True)
    
    plt.title(f'Theme Co-occurrence Matrix (threshold={prediction_threshold})\n(Row: Actual, Column: Predicted)')
    plt.xlabel('Predicted Theme')
    plt.ylabel('Actual Theme')
    
    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # Create directory for output
    os.makedirs('analysis/matrices', exist_ok=True)
    
    # Save the visualization
    output_file = f'analysis/matrices/theme_matrix_all_{prediction_threshold}_{num_samples}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the theme list
    theme_file = f'analysis/matrices/theme_list_{prediction_threshold}_{num_samples}.txt'
    with open(theme_file, 'w') as f:
        f.write(f"Theme co-occurrence matrix with threshold={prediction_threshold}, samples={num_samples}\n")
        f.write(f"Number of themes: {len(active_themes)}\n\n")
        f.write("Themes included in the matrix:\n")
        for i, theme in enumerate(active_themes, 1):
            f.write(f"{i}. {theme} (count: {theme_only_counts[theme]})\n")
    
    print(f"Confusion matrix saved to {output_file}")
    print(f"Theme list saved to {theme_file}")
    return matrix

def evaluate_performance(dataset, label_names, model, device, threshold, num_samples):
    """Evaluate model performance metrics"""
    
    # Initialize confusion matrix accumulators
    confusion_stats = {
        'true_positive': np.zeros(len(label_names)),
        'false_positive': np.zeros(len(label_names)),
        'false_negative': np.zeros(len(label_names)),
        'true_negative': np.zeros(len(label_names))
    }
    
    # Track performance for themes vs openings
    theme_indices = [i for i, name in enumerate(label_names) if dataset.is_theme(name)]
    opening_indices = [i for i, name in enumerate(label_names) if dataset.is_opening_tag(name)]
    
    # Process samples
    all_actual_themes = []
    all_predicted_themes = []
    
    for i in tqdm(range(num_samples), desc="Calculating metrics"):
        sample = dataset[i]
        target = sample['themes']
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
            out = model(input_tensor)
            predictions = torch.sigmoid(out).squeeze().cpu()
        
        # Ensure predictions and targets have the same size
        min_size = min(len(predictions), len(target))
        predictions_subset = predictions[:min_size]
        target_subset = target[:min_size]
        
        # Calculate metrics for this sample
        stats = compute_multilabel_confusion_matrix(predictions_subset, target_subset, threshold=threshold)
        
        # Update totals
        for key in confusion_stats:
            confusion_stats[key] += stats[key]
        
        # Collect actual and predicted themes for reporting
        pred_binary = (predictions > threshold).float()
        for j in range(len(target)):
            if j < len(label_names):
                theme_name = label_names[j]
                if dataset.is_theme(theme_name):
                    if target[j] == 1:
                        all_actual_themes.append(theme_name)
                    if pred_binary[j] == 1:
                        all_predicted_themes.append(theme_name)
    
    # Print overall metrics
    print("\n=== Overall Performance Statistics ===")
    
    # Calculate key metrics
    tp_sum = confusion_stats['true_positive'].sum()
    fp_sum = confusion_stats['false_positive'].sum()
    fn_sum = confusion_stats['false_negative'].sum()
    tn_sum = confusion_stats['true_negative'].sum()
    
    # Calculate overall metrics
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positives: {tp_sum}")
    print(f"False Positives: {fp_sum}")
    print(f"False Negatives: {fn_sum}")
    print(f"True Negatives: {tn_sum}")
    
    # Calculate theme-only metrics
    theme_tp = sum(confusion_stats['true_positive'][i] for i in theme_indices)
    theme_fp = sum(confusion_stats['false_positive'][i] for i in theme_indices)
    theme_fn = sum(confusion_stats['false_negative'][i] for i in theme_indices)
    theme_tn = sum(confusion_stats['true_negative'][i] for i in theme_indices)
    
    theme_precision = theme_tp / (theme_tp + theme_fp) if (theme_tp + theme_fp) > 0 else 0
    theme_recall = theme_tp / (theme_tp + theme_fn) if (theme_tp + theme_fn) > 0 else 0
    theme_f1 = 2 * (theme_precision * theme_recall) / (theme_precision + theme_recall) if (theme_precision + theme_recall) > 0 else 0
    
    print("\n=== Theme-only Performance ===")
    print(f"Theme Precision: {theme_precision:.4f}")
    print(f"Theme Recall: {theme_recall:.4f}")
    print(f"Theme F1 Score: {theme_f1:.4f}")
    
    # Calculate opening-only metrics
    opening_tp = sum(confusion_stats['true_positive'][i] for i in opening_indices)
    opening_fp = sum(confusion_stats['false_positive'][i] for i in opening_indices)
    opening_fn = sum(confusion_stats['false_negative'][i] for i in opening_indices)
    opening_tn = sum(confusion_stats['true_negative'][i] for i in opening_indices)
    
    opening_precision = opening_tp / (opening_tp + opening_fp) if (opening_tp + opening_fp) > 0 else 0
    opening_recall = opening_tp / (opening_tp + opening_fn) if (opening_tp + opening_fn) > 0 else 0
    opening_f1 = 2 * (opening_precision * opening_recall) / (opening_precision + opening_recall) if (opening_precision + opening_recall) > 0 else 0
    
    print("\n=== Opening-only Performance ===")
    print(f"Opening Precision: {opening_precision:.4f}")
    print(f"Opening Recall: {opening_recall:.4f}")
    print(f"Opening F1 Score: {opening_f1:.4f}")
    
    # Print top theme metrics
    print("\n=== Top Theme Performance ===")
    print(f"{'Theme':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)
    
    # Get theme metrics
    theme_metrics = []
    for i in theme_indices:
        if i < len(label_names):
            name = label_names[i]
            tp = confusion_stats['true_positive'][i]
            fp = confusion_stats['false_positive'][i]
            fn = confusion_stats['false_negative'][i]
            tn = confusion_stats['true_negative'][i]
            
            # Skip themes with no occurrences
            if tp + fn == 0:
                continue
                
            # Calculate metrics
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            theme_metrics.append((name, tp, fp, fn, tn, p, r, f))
    
    # Sort by F1 score and print top 20
    for name, tp, fp, fn, tn, p, r, f in sorted(theme_metrics, key=lambda x: x[7], reverse=True)[:20]:
        print(f"{name:<20} {tp:5.0f} {fp:5.0f} {fn:5.0f} {tn:5.0f} {p:10.4f} {r:10.4f} {f:10.4f}")
    
    # Return collected statistics
    return confusion_stats

def run_test():
    """Main test function that evaluates the model using cached data"""
    
    # Create dataset using the cached files
    print("Loading dataset...")
    dataset = ChessPuzzleDataset(dataset_path)
    
    # Adjust num_samples if larger than dataset
    global num_samples, prediction_threshold, max_themes
    num_samples = min(num_samples, len(dataset))
    print(f"Testing with {num_samples} samples")
    
    # Get label names and count theme occurrences
    label_names = dataset.get_theme_names()
    print(f"Dataset has {len(label_names)} labels")
    
    # Count theme occurrences for matrix filtering
    theme_counts = {}
    for i in tqdm(range(min(1000, len(dataset))), desc="Counting theme occurrences"):
        sample = dataset[i]
        target = sample['themes']
        
        for j, is_present in enumerate(target):
            if is_present == 1 and j < len(label_names):
                theme_name = label_names[j]
                if theme_name in theme_counts:
                    theme_counts[theme_name] += 1
                else:
                    theme_counts[theme_name] = 1
    
    # Load model configuration
    model_config = {
        "num_labels": len(label_names),
        "nlayers": 8,
        "embed_dim": 64,
        "inner_dim": 320, 
        "attention_dim": 64,
        "use_1x1conv": True,
        "dropout": 0.5
    }

    # Load from YAML if available
    config_path = 'model_config.yaml'
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Update config from YAML
            for key, value in yaml_config.items():
                if key in model_config:
                    model_config[key] = value
    
    # Create model
    print("Initializing model...")
    model = Model(**model_config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join("checkpoints", checkpoint_file)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = checkpoint_file
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # If using adaptive thresholding, collect outputs and targets first
    if prediction_threshold is None:
        print("\nCalculating adaptive threshold...")
        
        # Collect all predictions and targets
        all_outputs = []
        all_targets = []
        
        # Process sample subset to calculate threshold
        threshold_samples = min(100, num_samples)  # Use at most 100 samples for threshold calculation
        for i in tqdm(range(threshold_samples), desc="Collecting predictions"):
            sample = dataset[i]
            target = sample['themes']
            
            # Get model prediction
            with torch.no_grad():
                input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
                out = model(input_tensor)
                predictions = torch.sigmoid(out).squeeze().cpu()
            
            # Ensure predictions and targets have the same size
            min_size = min(len(predictions), len(target))
            all_outputs.append(predictions[:min_size].unsqueeze(0))
            all_targets.append(target[:min_size].unsqueeze(0))
        
        # Combine all outputs and targets
        combined_outputs = torch.cat(all_outputs, dim=0)
        combined_targets = torch.cat(all_targets, dim=0)
        
        # Calculate adaptive threshold
        prediction_threshold = calculate_adaptive_threshold(combined_outputs, combined_targets)
    
    print(f"\nUsing threshold: {prediction_threshold}")
    
    # Create confusion matrix
    matrix = create_confusion_matrix(
        dataset, label_names, theme_counts, model, device, 
        prediction_threshold, num_samples, max_themes)
    
    # Calculate metrics
    confusion_stats = evaluate_performance(
        dataset, label_names, model, device, prediction_threshold, num_samples)
    
    print("\nEvaluation complete!")
    return matrix, confusion_stats

if __name__ == "__main__":
    # Get command line arguments
    args = parse_args()
    num_samples = args.num_samples
    prediction_threshold = args.threshold
    checkpoint_file = args.checkpoint
    max_themes = args.max_themes
    dataset_path = args.dataset_path
    
    if prediction_threshold is None:
        print(f"Testing with {num_samples} samples using adaptive thresholding")
    else:
        print(f"Testing with {num_samples} samples and threshold {prediction_threshold}")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Run test
        matrix, confusion_stats = run_test()
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test execution: {str(e)}")
        import traceback
        traceback.print_exc()