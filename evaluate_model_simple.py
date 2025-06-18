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
from sklearn.metrics import f1_score

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Test chess puzzle classifier')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to test (default: 1000)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold for classification (if not specified, adaptive thresholding will be used)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_resume.pth",
                        help='Checkpoint file to use for testing')
    parser.add_argument('--optimized', action='store_true',
                        help='Use optimized version of the code')
    parser.add_argument('--use_cache', action='store_true',
                        help='Force using cache files even if CSV exists')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with detailed progress information')
    return parser.parse_args()

def create_confusion_matrix(dataset, theme_labels, model, device, prediction_threshold, num_samples):
    """Create confusion matrix from model predictions"""
    
    # Key chess themes we want to track in our matrix
    active_themes = ["advantage", "crushing", "endgame", "hangingPiece", "long", "middlegame", "short"]
    
    # Define common themes mapping for the full model output
    common_themes = {
        1555: "middlegame",
        1566: "crushing",
        1574: "endgame",
        1578: "hangingPiece",
        1585: "long",
        1594: "advantage",
        1606: "short"
    }
    print(f"Creating confusion matrix for {len(active_themes)} themes...")
    
    # Create theme to index mapping for confusion matrix
    theme_to_idx = {theme: idx for idx, theme in enumerate(active_themes)}
    
    # Create mapping from theme names to indices in the dataset's theme list
    theme_name_to_idx = {}
    for idx, theme in enumerate(theme_labels):
        if theme in active_themes:
            theme_name_to_idx[theme] = idx
    
    print(f"Theme indices in dataset: {theme_name_to_idx}")
    
    # Initialize confusion matrix
    matrix = np.zeros((len(active_themes), len(active_themes)))
    
    # Process each sample
    print(f"Using prediction threshold: {prediction_threshold:.4f}")
    for i in tqdm(range(num_samples), desc="Processing samples"):
        sample = dataset[i]
        target = sample['themes']
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
            out = model(input_tensor)
            out_raw = torch.sigmoid(out).squeeze().cpu()
        
        # Print values for key themes we're interested in
        print("\nOutput values for key themes:")
        for theme in active_themes:
            if theme in theme_name_to_idx:
                idx = theme_name_to_idx[theme]
                print(f"{theme} (idx {idx}): {out_raw[idx]:.6f} > {prediction_threshold} = {out_raw[idx] > prediction_threshold}")
            else:
                print(f"{theme}: not found in dataset")
        
        # Print raw output stats
        print(f"\nOutput statistics for sample {i}:")
        print(f"Raw output shape: {out_raw.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Raw output min: {out_raw.min().item()}, max: {out_raw.max().item()}, mean: {out_raw.mean().item()}")
        
        # Print top 5 highest values
        values, indices = out_raw.topk(5)
        print("\nTop 5 highest probabilities:")
        for idx, (val, ind) in enumerate(zip(values, indices)):
            # Try to map the index to a theme name
            theme_name = common_themes.get(ind.item(), f"Unknown-{ind.item()}")
            # Also check if it's in our theme labels
            if ind.item() < len(theme_labels):
                theme_name = theme_labels[ind.item()]
                
            print(f"  {idx+1}. Index {ind.item()} -> {theme_name}: {val.item():.6f}")
        
        # Get actual labels directly from target tensor
        actual_labels = [theme_labels[j] for j in range(len(target)) if target[j] == 1]
        
        # Get predicted labels from model output
        # We need to get the predictions for the actual themes with high probabilities
        predicted_labels = []
        
        # Look at the entire output vector and extract high probabilities
        for idx in range(len(out_raw)):
            if out_raw[idx] > prediction_threshold:
                # Try to identify the theme
                if idx in common_themes:
                    predicted_labels.append(common_themes[idx])
                elif idx < len(theme_labels):
                    predicted_labels.append(theme_labels[idx])
        
        # Debug
        print(f"\nSample {i}:")
        print(f"Actual labels: {actual_labels}")
        print(f"Predicted labels: {predicted_labels}")
        
        # Update confusion matrix for active themes only
        for actual in actual_labels:
            if actual in theme_to_idx:
                for predicted in predicted_labels:
                    if predicted in theme_to_idx:
                        matrix[theme_to_idx[actual], theme_to_idx[predicted]] += 1
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    
    # Normalize matrix by rows (actual occurrences)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums>0)
    
    # Create heatmap
    ax = sns.heatmap(normalized_matrix, 
                xticklabels=active_themes,
                yticklabels=active_themes,
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                annot=True,
                fmt='.2f',
                square=True)
    
    plt.title(f'Theme Co-occurrence Matrix (threshold={prediction_threshold})\n(Row: Actual, Column: Predicted)')
    plt.xlabel('Predicted Theme')
    plt.ylabel('Actual Theme')
    
    # Create directory for output
    os.makedirs('analysis/matrices', exist_ok=True)
    
    # Save the visualization
    output_file = f'analysis/matrices/confusion_matrix_simple_{prediction_threshold}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_file}")
    return matrix

def run_test(use_cache=False, custom_threshold=None):
    """Main test function that evaluates the model"""
    
    # Use the global prediction_threshold or the custom_threshold if provided
    global prediction_threshold
    if custom_threshold is not None:
        prediction_threshold = custom_threshold
    
    # First try using the CSV file from the regular dataset with the use_cache flag
    # This way we can use the CSV for labels but still load tensors from cache
    dataset_file = os.path.join("dataset", "lichess_db_puzzle.csv")
    
    # Point to the processed directory for cache files
    # This is a hack to use the CSV for labels but still load tensors from the processed dir
    os.environ['TENSOR_CACHE_DIR'] = os.path.join(os.getcwd(), "processed_lichess_puzzle_files")
    
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
    
    # Get test labels
    test_label_names = dataset.get_theme_names()
    print(f"Test dataset has {len(test_label_names)} labels")
    
    # Create confusion matrix
    matrix = create_confusion_matrix(dataset, test_label_names, model, device, prediction_threshold, num_samples)
    
    # Calculate metrics
    confusion_stats = evaluate_performance(dataset, test_label_names, model, device, prediction_threshold, num_samples)
    
    print("\nEvaluation complete!")
    return matrix, confusion_stats

def evaluate_performance(dataset, theme_labels, model, device, threshold, num_samples):
    """Evaluate model performance metrics"""
    
    # Define common themes mapping for the full model output
    common_themes = {
        1555: "middlegame",
        1566: "crushing",
        1574: "endgame",
        1578: "hangingPiece",
        1585: "long",
        1594: "advantage",
        1606: "short"
    }
    
    # Initialize confusion matrix accumulators
    confusion_stats = {
        'true_positive': np.zeros(len(theme_labels)),
        'false_positive': np.zeros(len(theme_labels)),
        'false_negative': np.zeros(len(theme_labels)),
        'true_negative': np.zeros(len(theme_labels))
    }
    
    # Process samples
    for i in tqdm(range(num_samples), desc="Calculating metrics"):
        sample = dataset[i]
        target = sample['themes']
        
        # Get model prediction
        with torch.no_grad():
            input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
            out = model(input_tensor)
            out_raw = torch.sigmoid(out).squeeze().cpu()
        
        # Create a tensor that matches the target size
        mapped_output = torch.zeros_like(target)
        
        # Get theme predictions for indices in our active themes mapping
        for test_idx, theme_name in enumerate(theme_labels):
            # Check if this theme is in our common themes mapping
            for model_idx, common_theme in common_themes.items():
                if theme_name == common_theme and model_idx < len(out_raw):
                    # Use the model's prediction for this theme
                    mapped_output[test_idx] = out_raw[model_idx]
        
        # Calculate metrics using the mapped outputs
        stats = compute_multilabel_confusion_matrix(mapped_output, target, threshold=threshold)
        
        # Update totals
        for key in confusion_stats:
            confusion_stats[key] += stats[key]
    
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
    
    return confusion_stats

# Function to calculate adaptive threshold
def calculate_adaptive_threshold(outputs, targets, verbose=True):
    """
    Calculate adaptive thresholds for improved metrics calculation in early training stages.
    This implements the approach described in docs/adaptive_thresholding.md.
    
    There are two approaches implemented:
    1. Class-specific thresholds based on output distribution (primary approach)
    2. Global F1-optimized threshold (fallback approach)
    
    Args:
        outputs: Model predictions (after sigmoid) - shape [num_samples, num_classes]
        targets: Ground truth labels - shape [num_samples, num_classes]
        verbose: Whether to print detailed threshold information
        
    Returns:
        best_threshold: The threshold that maximizes F1 score
    """
    # Class-specific adaptive thresholding (primary approach)
    num_classes = outputs.shape[1]
    class_thresholds = np.zeros(num_classes)
    
    # Calculate statistics
    if verbose:
        print("\nCalculating class-specific adaptive thresholds...")
    
    # For each class, calculate the mean and std of predictions
    for i in range(num_classes):
        class_outputs = outputs[:, i]
        class_mean = np.mean(class_outputs)
        class_std = np.std(class_outputs)
        
        # Calculate adaptive threshold for this class using the formula from docs
        # threshold = clamp(mean - 0.5 * std, min=0.05, max=0.5)
        class_threshold = max(0.05, min(0.5, class_mean - 0.5 * class_std))
        class_thresholds[i] = class_threshold
        
        if verbose and i < 10:  # Print details for first few classes to avoid token explosion
            print(f"Class {i}: mean={class_mean:.4f}, std={class_std:.4f}, threshold={class_threshold:.4f}")
    
    # Calculate average threshold across all classes
    avg_threshold = np.mean(class_thresholds)
    
    if verbose:
        print(f"\nAverage adaptive threshold across all classes: {avg_threshold:.4f}")
        print(f"Min class threshold: {np.min(class_thresholds):.4f}")
        print(f"Max class threshold: {np.max(class_thresholds):.4f}")
    
    # Also calculate global F1-optimized threshold as a fallback
    
    # Flatten the outputs and targets for global threshold calculation
    flat_outputs = outputs.reshape(-1)
    flat_targets = targets.reshape(-1)
    
    if verbose:
        # Print statistics about the inputs
        print("\nGlobal prediction statistics:")
        print(f"Shape of outputs: {outputs.shape}")
        print(f"Shape of targets: {targets.shape}")
        print(f"Total elements: {len(flat_outputs)}")
        print(f"Positive targets: {flat_targets.sum()} ({flat_targets.sum() / len(flat_targets):.2%})")
        print(f"Output stats - min: {flat_outputs.min():.6f}, max: {flat_outputs.max():.6f}, mean: {flat_outputs.mean():.6f}")
        
    # Create logarithmic thresholds for better coverage of low values
    # This is especially important if the model outputs are very small
    log_thresholds = np.logspace(-4, -0.3, 15)  # From 0.0001 to 0.5
    linear_thresholds = np.linspace(0.01, 0.5, 10)
    all_thresholds = np.unique(np.sort(np.concatenate([log_thresholds, linear_thresholds, [avg_threshold]])))
    
    best_f1 = 0
    best_threshold = avg_threshold  # Default to class average
    
    # Test thresholds to find optimal F1 score
    if verbose:
        print("\nFinding optimal global threshold...")
        print(f"{'Threshold':>10} {'F1 Score':>10} {'Positives %':>11}")
        print("-" * 35)
    
    for threshold in all_thresholds:
        # Convert continuous predictions to binary using threshold
        binary_preds = (flat_outputs > threshold).astype(np.int64)
        
        # Calculate percentage of positive predictions
        positives_percent = binary_preds.sum() / len(binary_preds) * 100
        
        # Skip thresholds that produce no positive predictions
        if positives_percent == 0:
            if verbose:
                print(f"{threshold:10.4f} {'N/A':>10} {positives_percent:10.2f}% (skipped - no positives)")
            continue
        
        # Calculate F1 score
        f1 = f1_score(flat_targets, binary_preds, average='macro', zero_division=0)
        
        if verbose:
            print(f"{threshold:10.4f} {f1:10.4f} {positives_percent:10.2f}%")
        
        # Update best threshold if this one is better
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Choose the best approach
    if best_f1 > 0:
        if verbose:
            print(f"\nOptimal global threshold: {best_threshold:.4f} (F1 score: {best_f1:.4f})")
        return best_threshold
    else:
        if verbose:
            print(f"\nNo threshold produced positive predictions with good F1 score.")
            print(f"Using class-specific average threshold: {avg_threshold:.4f}")
        return avg_threshold

if __name__ == "__main__":
    # Get command line arguments
    args = parse_args()
    num_samples = args.num_samples
    checkpoint_file = args.checkpoint
    use_cache = args.use_cache
    verbose = args.verbose
    
    # Use adaptive thresholding if no threshold is specified
    adaptive_threshold = args.threshold is None
    # Initialize with a default, but we'll calculate the optimal threshold if not specified
    prediction_threshold = args.threshold if args.threshold is not None else 0.3
    
    threshold_mode = "user-specified" if args.threshold is not None else "adaptive (will be calculated)"
    print(f"Testing with {num_samples} samples and prediction threshold: {threshold_mode}")
    if use_cache:
        print(f"Using cached tensor files (--use_cache flag is set)")
    
    # Start timing
    start_time = time.time()
    
    try:
        # For adaptive thresholding, we need to collect predictions and targets
        all_outputs = []
        all_targets = []
        
        # Run test with initial threshold
        matrix, confusion_stats = run_test(use_cache=use_cache)
        
        # If adaptive thresholding is enabled, collect predictions and targets
        if adaptive_threshold:
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
            
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Set model to evaluation mode
            model.eval()
            
            # Collect predictions and targets
            print(f"Collecting predictions for {samples_to_use} samples...")
            for i in tqdm(range(samples_to_use), desc="Collecting predictions"):
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
            
            # Calculate and set the optimal threshold
            optimal_threshold = calculate_adaptive_threshold(all_outputs_array, all_targets_array, verbose=verbose)
            prediction_threshold = optimal_threshold
            
            print(f"\nUsing adaptive threshold: {prediction_threshold:.4f}")
            
            # Run the test again with the optimal threshold
            print("\nRunning evaluation with optimal threshold...")
            matrix, confusion_stats = run_test(use_cache=use_cache, custom_threshold=optimal_threshold)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during test execution: {str(e)}")
        import traceback
        traceback.print_exc()