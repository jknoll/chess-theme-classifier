import torch
import numpy as np
import argparse
import time
import os
import pandas as pd
from tabulate import tabulate
from dataset import ChessPuzzleDataset
from model import Model
from metrics import compute_multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
from sklearn.metrics import f1_score

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate chess puzzle classifier with detailed metrics')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to test (default: 1000)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold for classification (if not specified, adaptive thresholding will be used)')
    parser.add_argument('--threshold_steps', type=int, default=35,
                        help='Number of threshold values to test for per-class adaptive thresholding (default: 35)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_resume.pth",
                        help='Checkpoint file to use for testing')
    parser.add_argument('--output_dir', type=str, default="analysis/f1",
                        help='Output directory for metrics files (default: analysis/f1)')
    parser.add_argument('--top', type=int, default=None,
                        help='Only show top N themes in console output (default: all)')
    parser.add_argument('--min_occurrences', type=int, default=5,
                        help='Minimum number of occurrences for a theme to be included in metrics (default: 5)')
    parser.add_argument('--use_cache', action='store_true',
                        help='Force using cache files even if CSV exists')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with detailed progress information')
    parser.add_argument('--no_split', action='store_true',
                        help='Do not split metrics between themes and openings')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    return parser.parse_args()

def calculate_per_label_metrics(confusion_stats, theme_labels, min_occurrences=5):
    """Calculate metrics for each label and sort by F1 score."""
    theme_metrics = []
    
    for i, theme in enumerate(theme_labels):
        tp = confusion_stats['true_positive'][i]
        fp = confusion_stats['false_positive'][i]
        fn = confusion_stats['false_negative'][i]
        tn = confusion_stats['true_negative'][i]
        
        # Calculate metrics (handling division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Only include if theme has enough occurrences
        if tp + fn >= min_occurrences:
            theme_metrics.append({
                'theme': theme,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'occurrences': tp + fn
            })
    
    # Sort by F1 score for better readability
    theme_metrics.sort(key=lambda x: x['f1'], reverse=True)
    
    return theme_metrics

def run_evaluation(use_cache=False, custom_threshold=None, per_class_thresholds=None, quiet=False):
    """Main function that evaluates the model and generates metrics."""
    # Use the global prediction_threshold or the custom_threshold if provided
    global prediction_threshold
    if custom_threshold is not None:
        prediction_threshold = custom_threshold
    
    if not quiet:
        print(f"Using prediction threshold: {prediction_threshold:.4f}")
    
    # Set up dataset
    dataset_file = os.path.join("dataset", "lichess_db_puzzle.csv")
    
    # Point to the processed directory for cache files
    os.environ['TENSOR_CACHE_DIR'] = os.path.join(os.getcwd(), "processed_lichess_puzzle_files")
    
    if not quiet:
        print(f"Using dataset CSV from: {dataset_file}")
        print(f"Using tensor cache from: {os.environ['TENSOR_CACHE_DIR']}")
    
    # Create the dataset with class_conditional_augmentation
    dataset = ChessPuzzleDataset(
        dataset_file, 
        use_cache=use_cache, 
        class_conditional_augmentation=True
    )
    
    # Adjust num_samples if larger than dataset
    global num_samples
    num_samples = min(num_samples, len(dataset))
    if not quiet:
        print(f"Testing with {num_samples} samples")
    
    # Load model configuration
    model_config = {
        "num_labels": 1616,
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
        if not quiet:
            print(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Update config from YAML
            for key, value in yaml_config.items():
                if key in model_config:
                    model_config[key] = value
    
    # Create model
    if not quiet:
        print("Initializing model...")
    model = Model(**model_config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not quiet:
        print(f"Using device: {device}")
    model = model.to(device)
    
    # Load checkpoint using the unified checkpoint utilities
    from checkpoint_utils import load_checkpoint
    
    checkpoint_path = os.path.join("checkpoints", checkpoint_file)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = checkpoint_file
    
    if not quiet:
        print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load the checkpoint with our utility function that handles different formats
    # Set strict=False to allow loading checkpoints between models with different architectures
    checkpoint_info = load_checkpoint(checkpoint_path, model, device=device, strict=False)
    
    if not quiet:
        print(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}, "
              f"global step {checkpoint_info['global_step']}")
        print(f"Note: Using non-strict loading to handle model architecture differences")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get theme labels
    all_theme_labels = dataset.get_theme_names()
    
    # No longer filtering based on themes_only
    filtered_theme_names = all_theme_labels
    theme_mask = np.ones(len(all_theme_labels), dtype=bool)
    
    if not quiet:
        print(f"Evaluating {len(filtered_theme_names)} labels")
    
    # Initialize confusion matrix accumulators
    confusion_stats = {
        'true_positive': np.zeros(len(all_theme_labels)),
        'false_positive': np.zeros(len(all_theme_labels)),
        'false_negative': np.zeros(len(all_theme_labels)),
        'true_negative': np.zeros(len(all_theme_labels))
    }
    
    # Get a mapping from theme names to indices in the model output
    theme_to_index = {name: idx for idx, name in enumerate(dataset.get_theme_names())}
    
    # Process samples
    for i in tqdm(range(num_samples), desc="Evaluating samples", disable=quiet):
        sample = dataset[i]
        target = sample['themes']
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
            out = model(input_tensor)
            out_raw = torch.sigmoid(out).squeeze().cpu()
        
        # Use direct mapping - model output indices correspond to the themes
        # This assumes the model and dataset use the same theme ordering
        predicted = out_raw
        
        # Calculate metrics using the direct model outputs
        stats = compute_multilabel_confusion_matrix(
            predicted, target, 
            threshold=prediction_threshold,
            per_class_thresholds=per_class_thresholds
        )
        
        # Update totals
        for key in confusion_stats:
            confusion_stats[key] += stats[key]
    
    # Calculate per-label metrics
    theme_metrics = calculate_per_label_metrics(confusion_stats, all_theme_labels, min_occurrences=args.min_occurrences)
    
    # No longer filtering based on themes_only since we're now splitting the output
    
    # Calculate overall metrics
    tp_sum = confusion_stats['true_positive'][theme_mask].sum()
    fp_sum = confusion_stats['false_positive'][theme_mask].sum()
    fn_sum = confusion_stats['false_negative'][theme_mask].sum()
    tn_sum = confusion_stats['true_negative'][theme_mask].sum()
    
    # Calculate overall metrics
    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    if not quiet:
        print("\n=== Overall Performance Statistics ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"True Positives: {tp_sum}")
        print(f"False Positives: {fp_sum}")
        print(f"False Negatives: {fn_sum}")
        print(f"True Negatives: {tn_sum}")
    
    return theme_metrics, {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp_sum,
        'fp': fp_sum,
        'fn': fn_sum,
        'tn': tn_sum
    }

def calculate_adaptive_threshold(outputs, targets, theme_names=None, verbose=True, output_dir="analysis/f1", threshold_steps=35):
    """
    Calculate per-class F1-optimized adaptive thresholds and generate precision-recall curves.
    This implements the approach described in docs/per-class-adaptive-thresholding.md.
    
    Args:
        outputs: Model predictions (after sigmoid) - shape [num_samples, num_classes]
        targets: Ground truth labels - shape [num_samples, num_classes]
        theme_names: List of theme names corresponding to the class indices
        verbose: Whether to print detailed threshold information
        output_dir: Directory to save per-class threshold and PR curve data
        threshold_steps: Number of threshold values to test for each class (default: 35)
        
    Returns:
        dict: Dictionary containing per-class thresholds and global fallback threshold
    """
    num_classes = outputs.shape[1]
    class_thresholds = np.zeros(num_classes)
    per_class_pr_data = {}  # Store precision-recall curve data for each class
    
    # Create threshold range for sweeping (configurable density)
    # Split the threshold_steps between logarithmic and linear ranges
    log_steps = max(1, threshold_steps * 2 // 3)  # ~67% logarithmic
    linear_steps = max(1, threshold_steps - log_steps)  # ~33% linear
    
    log_thresholds = np.logspace(-4, -0.3, log_steps)  # From 0.0001 to 0.5
    linear_thresholds = np.linspace(0.01, 0.5, linear_steps)
    all_thresholds = np.unique(np.sort(np.concatenate([log_thresholds, linear_thresholds])))
    
    if verbose:
        print("\nCalculating per-class F1-optimized adaptive thresholds...")
    
    # Calculate F1-optimized threshold for each class
    for i in range(num_classes):
        class_outputs = outputs[:, i]
        class_targets = targets[:, i]
        
        # Skip classes with no positive examples
        if class_targets.sum() == 0:
            class_thresholds[i] = 0.5  # Default threshold
            if verbose and i < 10:
                class_label = theme_names[i] if theme_names and i < len(theme_names) else f"Class {i}"
                print(f"{class_label}: No positive examples, using default threshold=0.5000")
            continue
        
        best_f1 = 0
        best_threshold = 0.5
        pr_curve_data = []
        
        # Sweep thresholds to find F1-optimizing threshold
        for threshold in all_thresholds:
            binary_preds = (class_outputs > threshold).astype(np.int64)
            
            # Calculate metrics
            tp = np.sum((binary_preds == 1) & (class_targets == 1))
            fp = np.sum((binary_preds == 1) & (class_targets == 0))
            fn = np.sum((binary_preds == 0) & (class_targets == 1))
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store data for precision-recall curve
            pr_curve_data.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })
            
            # Update best threshold
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        class_thresholds[i] = best_threshold
        per_class_pr_data[i] = pr_curve_data
        
        if verbose and i < 10:  # Print details for first few classes
            class_label = theme_names[i] if theme_names and i < len(theme_names) else f"Class {i}"
            print(f"{class_label}: optimal_threshold={best_threshold:.4f}, best_f1={best_f1:.4f}")
    
    # Calculate global F1-optimized threshold as fallback
    flat_outputs = outputs.reshape(-1)
    flat_targets = targets.reshape(-1)
    
    if verbose:
        print(f"\nAverage per-class threshold: {np.mean(class_thresholds):.4f}")
        print(f"Min class threshold: {np.min(class_thresholds):.4f}")
        print(f"Max class threshold: {np.max(class_thresholds):.4f}")
        
        print("\nGlobal prediction statistics:")
        print(f"Shape of outputs: {outputs.shape}")
        print(f"Shape of targets: {targets.shape}")
        print(f"Total elements: {len(flat_outputs)}")
        print(f"Positive targets: {flat_targets.sum()} ({flat_targets.sum() / len(flat_targets):.2%})")
        print(f"Output stats - min: {flat_outputs.min():.6f}, max: {flat_outputs.max():.6f}, mean: {flat_outputs.mean():.6f}")
        
        print("\nFinding optimal global threshold...")
        print(f"{'Threshold':>10} {'F1 Score':>10} {'Positives %':>11}")
        print("-" * 35)
    
    best_global_f1 = 0
    best_global_threshold = np.mean(class_thresholds)
    
    for threshold in all_thresholds:
        binary_preds = (flat_outputs > threshold).astype(np.int64)
        positives_percent = binary_preds.sum() / len(binary_preds) * 100
        
        if positives_percent == 0:
            if verbose:
                print(f"{threshold:10.4f} {'N/A':>10} {positives_percent:10.2f}% (skipped - no positives)")
            continue
        
        f1 = f1_score(flat_targets, binary_preds, average='macro', zero_division=0)
        
        if verbose:
            print(f"{threshold:10.4f} {f1:10.4f} {positives_percent:10.2f}%")
        
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_threshold = threshold
    
    if verbose:
        print(f"\nOptimal global threshold: {best_global_threshold:.4f} (F1 score: {best_global_f1:.4f})")
    
    # Save per-class threshold and PR curve data
    os.makedirs(output_dir, exist_ok=True)
    
    # Save per-class thresholds
    threshold_data = []
    for i in range(num_classes):
        class_label = theme_names[i] if theme_names and i < len(theme_names) else f"Class_{i}"
        threshold_data.append({
            'class_index': i,
            'class_name': class_label,
            'optimal_threshold': class_thresholds[i],
            'num_positive_examples': int(targets[:, i].sum())
        })
    
    # Save thresholds to CSV
    import pandas as pd
    df_thresholds = pd.DataFrame(threshold_data)
    threshold_file = os.path.join(output_dir, 'per_class_thresholds.csv')
    df_thresholds.to_csv(threshold_file, index=False)
    
    # Save precision-recall curve data
    pr_curve_file = os.path.join(output_dir, 'per_class_pr_curves.csv')
    pr_rows = []
    for class_idx, pr_data in per_class_pr_data.items():
        class_label = theme_names[class_idx] if theme_names and class_idx < len(theme_names) else f"Class_{class_idx}"
        for point in pr_data:
            pr_rows.append({
                'class_index': class_idx,
                'class_name': class_label,
                **point
            })
    
    df_pr_curves = pd.DataFrame(pr_rows)
    df_pr_curves.to_csv(pr_curve_file, index=False)
    
    if verbose:
        print(f"\nPer-class thresholds saved to: {threshold_file}")
        print(f"Precision-recall curve data saved to: {pr_curve_file}")
    
    return {
        'per_class_thresholds': class_thresholds,
        'global_threshold': best_global_threshold,
        'pr_curve_data': per_class_pr_data,
        'threshold_data': threshold_data
    }

def print_metrics_table(metrics, top=None):
    """Print metrics in a formatted table, optionally limiting to top N rows."""
    headers = ["Theme", "F1 Score", "Precision", "Recall", "TP", "FP", "FN", "TN", "Occurrences"]
    
    # Limit to top N if specified
    if top is not None and top > 0:
        metrics_to_print = metrics[:top]
    else:
        metrics_to_print = metrics
    
    # Format data for tabulate
    table_data = []
    for m in metrics_to_print:
        row = [
            m['theme'],
            f"{m['f1']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            int(m['tp']),
            int(m['fp']),
            int(m['fn']),
            int(m['tn']),
            int(m['occurrences'])
        ]
        table_data.append(row)
    
    # Print table
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    if top is not None and len(metrics) > top:
        print(f"\nShowing top {top} of {len(metrics)} labels. Export to CSV for complete metrics.")

def export_metrics_to_csv(metrics, filename):
    """Export metrics to a CSV file."""
    # Convert metrics to DataFrame
    df = pd.DataFrame(metrics)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Metrics exported to {filename}")

def generate_metrics_chart(metrics, output_file, title=None, max_labels=None):
    """Generate a bar chart of F1 scores for themes/openings.
    
    Args:
        metrics: List of metric dictionaries for each label
        output_file: Path to save the chart
        title: Optional title for the chart
        max_labels: Maximum number of labels to include (default: all)
    """
    # Limit number of labels if specified
    metrics_to_plot = metrics[:max_labels] if max_labels is not None else metrics
    
    # Extract data
    labels = [m['theme'] for m in metrics_to_plot]
    f1_scores = [m['f1'] for m in metrics_to_plot]
    precision = [m['precision'] for m in metrics_to_plot]
    recall = [m['recall'] for m in metrics_to_plot]
    
    # Create chart
    plt.figure(figsize=(14, 10))
    
    # Create bar positions
    x = np.arange(len(labels))
    width = 0.25
    
    # Plot bars with F1 being more prominent
    # Use muted colors and transparency for precision and recall
    plt.bar(x, f1_scores, width, label='F1 Score', color='#1f77b4', zorder=3)  # Prominent color for F1
    plt.bar(x - width, precision, width, label='Precision', color='#d3d3d3', alpha=0.7, zorder=2)  # Light gray with transparency
    plt.bar(x + width, recall, width, label='Recall', color='#a9a9a9', alpha=0.7, zorder=2)  # Darker gray with transparency
    
    # Add gridlines to improve readability
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Add labels and legend
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    else:
        plt.title(f'F1 Score, Precision, and Recall by Label', fontsize=14)
        
    plt.xticks(x, labels, rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Customize legend
    plt.legend(fontsize=12)
    
    # Set y-axis limits for consistency
    plt.ylim(0, 1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save chart
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Metrics chart saved to {output_file}")

if __name__ == "__main__":
    # Get command line arguments
    args = parse_args()
    num_samples = args.num_samples
    checkpoint_file = args.checkpoint
    use_cache = args.use_cache
    verbose = args.verbose and not args.quiet
    quiet = args.quiet
    
    # Use adaptive thresholding if no threshold is specified
    adaptive_threshold = args.threshold is None
    # Initialize with a default, but we'll calculate the optimal threshold if not specified
    prediction_threshold = args.threshold if args.threshold is not None else 0.3
    
    if not quiet:
        threshold_mode = "user-specified" if args.threshold is not None else "adaptive (will be calculated)"
        print(f"Testing with {num_samples} samples and prediction threshold: {threshold_mode}")
        if use_cache:
            print(f"Using cached tensor files (--use_cache flag is set)")
        if args.no_split:
            print("Evaluating all labels together (no split between themes and openings)")
        else:
            print("Evaluating themes and openings separately")
    
    # Start timing
    start_time = time.time()
    
    try:
        # For adaptive thresholding, we need to collect predictions and targets
        all_outputs = []
        all_targets = []
        
        # Run evaluation with initial threshold
        if adaptive_threshold:
            if not quiet:
                print("\nCollecting model predictions for adaptive thresholding...")
            
            # Create dataset
            dataset_file = os.path.join("dataset", "lichess_db_puzzle.csv")
            os.environ['TENSOR_CACHE_DIR'] = os.path.join(os.getcwd(), "processed_lichess_puzzle_files")
            dataset = ChessPuzzleDataset(dataset_file, use_cache=use_cache, class_conditional_augmentation=True)
            
            # Adjust num_samples if larger than dataset
            samples_to_use = min(num_samples, len(dataset))
            
            # Create model and load checkpoint
            model_config = {
                "num_labels": 1616,
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
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    # Update config from YAML
                    for key, value in yaml_config.items():
                        if key in model_config:
                            model_config[key] = value
            
            # Create model
            model = Model(**model_config)
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Load checkpoint
            checkpoint_path = os.path.join("checkpoints", checkpoint_file)
            if not os.path.exists(checkpoint_path):
                checkpoint_path = checkpoint_file
            
            # Import the checkpoint utilities
            from checkpoint_utils import load_checkpoint
            
            # Load the checkpoint with our utility function that handles different formats
            checkpoint_info = load_checkpoint(checkpoint_path, model, device=device, strict=False)
            
            if not quiet:
                print(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}, "
                      f"global step {checkpoint_info['global_step']}")
                print(f"Note: Using non-strict loading to handle model architecture differences")
            
            # Set model to evaluation mode
            model.eval()
            
            # Collect predictions and targets
            if not quiet:
                print(f"Collecting predictions for {samples_to_use} samples...")
            for i in tqdm(range(samples_to_use), desc="Collecting predictions", disable=quiet):
                sample = dataset[i]
                target = sample['themes']
                
                # Get model prediction
                with torch.no_grad():
                    input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
                    out = model(input_tensor)
                    out_raw = torch.sigmoid(out).squeeze().cpu()
                
                all_outputs.append(out_raw.numpy())
                all_targets.append(target.numpy())
            
            # Convert lists to arrays
            all_outputs_array = np.vstack(all_outputs)
            all_targets_array = np.vstack(all_targets)
            
            # Calculate per-class optimal thresholds
            threshold_results = calculate_adaptive_threshold(
                all_outputs_array, all_targets_array, 
                theme_names=dataset.get_theme_names(), 
                verbose=verbose, 
                output_dir=args.output_dir,
                threshold_steps=args.threshold_steps
            )
            per_class_thresholds = threshold_results['per_class_thresholds']
            global_threshold = threshold_results['global_threshold']
            prediction_threshold = global_threshold  # For compatibility
            
            if not quiet:
                print(f"\nUsing per-class adaptive thresholds (global fallback: {global_threshold:.4f})")
        
        # Run the evaluation with per-class thresholds
        if not quiet:
            print("\nRunning evaluation...")
        theme_metrics, overall_metrics = run_evaluation(
            use_cache=use_cache,
            custom_threshold=prediction_threshold,
            per_class_thresholds=per_class_thresholds if adaptive_threshold else None,
            quiet=quiet
        )
        
        # Create base filename components
        # Format threshold with 4 decimal places and replace dot with underscore for filename compatibility
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        threshold_str = f"threshold_{prediction_threshold:.4f}".replace('.', '_')
        samples_str = f"samples_{num_samples}"
        date_str = f"date_{timestamp}"
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # If not splitting, process all metrics together
        if args.no_split:
            # Print results in a table
            if not quiet:
                print("\n=== Per-Label Metrics (sorted by F1 score) ===")
                print_metrics_table(theme_metrics, top=args.top)
            
            # Base filename for all metrics
            base_filename = f"metrics_{threshold_str}_{samples_str}_{date_str}"
            csv_path = os.path.join(args.output_dir, f"{base_filename}.csv")
            chart_path = os.path.join(args.output_dir, f"{base_filename}_chart.png")
            
            # Export metrics to CSV
            export_metrics_to_csv(theme_metrics, csv_path)
            
            # Generate metrics chart
            chart_title = f"F1 Score, Precision, and Recall by Label (threshold={prediction_threshold:.4f}, samples={num_samples})"
            generate_metrics_chart(theme_metrics, chart_path, title=chart_title)
        
        # Split metrics between themes and openings
        else:
            # Reuse existing dataset instance if possible, otherwise create a minimal one just for label classification
            using_heuristic = False
            try:
                if 'dataset' not in locals() or dataset is None:
                    # Create a minimal dataset just for theme classification
                    dataset_file = os.path.join("dataset", "lichess_db_puzzle_small.csv")
                    dataset = ChessPuzzleDataset(dataset_file, use_cache=use_cache)
            except Exception as e:
                print(f"Warning: Could not create dataset for theme classification: {e}")
                print("Attempting to classify themes/openings manually...")
                using_heuristic = True
                
                # Simple heuristic to identify themes vs openings
                # Openings typically start with a capital letter and often have underscores
                is_opening = lambda x: '_' in x and x[0].isupper()
            
            if using_heuristic:
                # Use the heuristic for classification
                theme_only_metrics = [m for m in theme_metrics if not is_opening(m['theme'])]
                opening_metrics = [m for m in theme_metrics if is_opening(m['theme'])]
            else:
                # Use the dataset methods for classification
                theme_only_metrics = [m for m in theme_metrics if dataset.is_theme(m['theme'])]
                opening_metrics = [m for m in theme_metrics if dataset.is_opening_tag(m['theme'])]
            
            # Print theme metrics in a table
            if not quiet:
                print("\n=== Chess Theme Metrics (sorted by F1 score) ===")
                print_metrics_table(theme_only_metrics, top=args.top)
            
            # Print opening metrics in a table if there are any
            if opening_metrics and not quiet:
                print("\n=== Opening Tag Metrics (sorted by F1 score) ===")
                print_metrics_table(opening_metrics, top=args.top)
            
            # Base filenames for themes and openings
            theme_base = f"themes_{threshold_str}_{samples_str}_{date_str}"
            opening_base = f"openings_{threshold_str}_{samples_str}_{date_str}"
            
            # Export theme metrics to CSV
            theme_csv_path = os.path.join(args.output_dir, f"{theme_base}.csv")
            export_metrics_to_csv(theme_only_metrics, theme_csv_path)
            
            # Generate theme metrics chart
            theme_chart_path = os.path.join(args.output_dir, f"{theme_base}_chart.png")
            theme_chart_title = f"Chess Theme Metrics (threshold={prediction_threshold:.4f}, samples={num_samples})"
            generate_metrics_chart(theme_only_metrics, theme_chart_path, title=theme_chart_title)
            
            # Export opening metrics to CSV if there are any
            if opening_metrics:
                opening_csv_path = os.path.join(args.output_dir, f"{opening_base}.csv")
                export_metrics_to_csv(opening_metrics, opening_csv_path)
                
                # Generate opening metrics chart
                opening_chart_path = os.path.join(args.output_dir, f"{opening_base}_chart.png")
                opening_chart_title = f"Opening Tag Metrics (threshold={prediction_threshold:.4f}, samples={num_samples})"
                generate_metrics_chart(opening_metrics, opening_chart_path, title=opening_chart_title)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        if not quiet:
            print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
            print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()