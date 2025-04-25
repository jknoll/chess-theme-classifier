import torch
import numpy as np
import argparse
import time
import cProfile
import pstats
import io
import os
from pstats import SortKey
from dataset import ChessPuzzleDataset
from model import Model
from metrics import jaccard_similarity, compute_multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Test chess puzzle classifier and generate confusion matrices')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to test (default: 1000)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Prediction threshold for classification (default: 0.3)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_resume.pth",
                        help='Checkpoint file to use for testing')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling')
    parser.add_argument('--profile_output', type=str, default="profile_stats.txt",
                        help='Output file for profiling statistics')
    parser.add_argument('--optimized', action='store_true',
                        help='Use optimized version of the code')
    return parser.parse_args()

# Global variable for label filtering (used by the matrix functions)
label_filter = lambda x: True

# Create co-occurrence matrix
def create_cooccurrence_matrix(labels, label_type, dataset, model, device, active_themes=None, cached_predictions=None):
    """Create a co-occurrence matrix for the given labels."""
    global label_filter
    
    if active_themes is None:
        # If no active themes provided, use all themes that have non-zero metrics
        active_themes = [m['theme'] for m in theme_metrics if label_filter(m['theme'])]
    
    # For large matrices, limit to the top labels by F1 score
    max_themes = 75  # Maximum number of themes to show in a matrix
    if len(active_themes) > max_themes:
        print(f"Too many labels ({len(active_themes)}) for a readable matrix, limiting to top {max_themes} by F1 score")
        # Find metrics for the active themes
        filtered_metrics = [m for m in theme_metrics if m['theme'] in active_themes]
        # Sort by F1 score and take top max_themes
        filtered_metrics.sort(key=lambda x: x['f1'], reverse=True)
        active_themes = [m['theme'] for m in filtered_metrics[:max_themes]]
        
    n_themes = len(active_themes)
    if n_themes == 0:
        print(f"No active {label_type}s found with non-zero metrics")
        return None
        
    print(f"Creating co-occurrence matrix for {n_themes} {label_type}s...")
    
    cooccurrence = np.zeros((n_themes, n_themes))
    theme_to_idx = {theme: idx for idx, theme in enumerate(active_themes)}
    
    # Optimized path using cached predictions and vectorized operations
    if args.optimized and cached_predictions is not None:
        matrix_progress = tqdm(range(num_samples), desc=f"Creating {label_type} matrix", unit="sample")
        for i in matrix_progress:
            matrix_progress.set_description(f"Matrix sample {i+1}/{num_samples} ({label_type})")
            
            sample = dataset[i]
            target = sample['themes']
            
            # Use cached predictions instead of recomputing
            out = cached_predictions[i]
            
            # Get actual and predicted labels based on the filter
            actual_labels = [label for j, label in enumerate(labels) if target[j] == 1 and label_filter(label)]
            pred_binary = (out > prediction_threshold).float()
            predicted_labels = [label for j, label in enumerate(labels) if pred_binary[j] == 1 and label_filter(label)]
            
            # Vectorized update of co-occurrence matrix - prepare indices
            actual_indices = [theme_to_idx[label] for label in actual_labels if label in theme_to_idx]
            pred_indices = [theme_to_idx[label] for label in predicted_labels if label in theme_to_idx]
            
            # Update co-occurrence matrix (for small numbers of indices, loops may be faster than creating temporary matrices)
            for a_idx in actual_indices:
                for p_idx in pred_indices:
                    cooccurrence[a_idx, p_idx] += 1
    
    # Original non-optimized path
    else:
        # Fill the co-occurrence matrix with progress bar
        matrix_progress = tqdm(range(num_samples), desc=f"Creating {label_type} matrix", unit="sample")
        for i in matrix_progress:
            matrix_progress.set_description(f"Matrix sample {i+1}/{num_samples} ({label_type})")
            
            sample = dataset[i]
            target = sample['themes']
            
            with torch.no_grad():
                out = model(sample['board'].unsqueeze(0).unsqueeze(0).to(device))
                out = torch.sigmoid(out)
            
            # Get actual and predicted labels based on the filter
            actual_labels = [label for j, label in enumerate(labels) if target[j] == 1 and label_filter(label)]
            pred_binary = (out.squeeze().cpu() > prediction_threshold).float()
            predicted_labels = [label for j, label in enumerate(labels) if pred_binary[j] == 1 and label_filter(label)]
            
            # Update co-occurrence matrix
            for actual in actual_labels:
                if actual in theme_to_idx:
                    for predicted in predicted_labels:
                        if predicted in theme_to_idx:
                            cooccurrence[theme_to_idx[actual], theme_to_idx[predicted]] += 1
    
    # Create normalized confusion matrix (divide by number of actual occurrences)
    actual_counts = cooccurrence.sum(axis=1, keepdims=True)
    normalized_matrix = np.where(actual_counts > 0, cooccurrence / actual_counts, 0)
    
    # Determine appropriate figure size and settings based on number of themes
    if n_themes > 50:
        fig_size = (40, 30)
        tick_font_size = 5
        annot = False  # Turn off cell annotations for very large matrices
        linewidths = 0.1
    elif n_themes > 30:
        fig_size = (35, 25)
        tick_font_size = 6
        annot = True
        linewidths = 0.2
    elif n_themes > 20:
        fig_size = (30, 20)
        tick_font_size = 7
        annot = True
        linewidths = 0.3
    else:
        fig_size = (24, 18)
        tick_font_size = 8
        annot = True
        linewidths = 0.5
    
    # Create figure with custom size
    plt.figure(figsize=fig_size)
    
    # Create custom annotations to reduce clutter
    if annot:
        # Only show values above a threshold to reduce visual noise
        annotations = np.where(normalized_matrix < 0.2, "", 
                            np.round(normalized_matrix, 1).astype(str))
    else:
        annotations = False
    
    # Create heatmap with improved legibility
    ax = sns.heatmap(normalized_matrix, 
                xticklabels=active_themes,
                yticklabels=active_themes,
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                annot=annotations,
                fmt='',  # Using empty format for custom annotations
                square=True,
                linewidths=linewidths,
                linecolor='gray',
                annot_kws={'size': tick_font_size-1} if annot else {},
                cbar_kws={'shrink': .5})
    
    # Improve tick label formatting
    plt.setp(ax.get_xticklabels(), fontsize=tick_font_size, rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), fontsize=tick_font_size)
    
    plt.title(f'{label_type.capitalize()} Co-occurrence Matrix (threshold={prediction_threshold}, samples={num_samples})\n(Row: Actual, Column: Predicted, Values: P(Predicted|Actual))', 
             pad=20, size=14)  # Added padding to title
    plt.xlabel(f'Predicted {label_type.capitalize()}', size=12, labelpad=10)
    plt.ylabel(f'Actual {label_type.capitalize()}', size=12, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', size=10)  # Increased font size
    plt.yticks(rotation=0, size=10)  # Increased font size
    
    # Add more padding around the plot for better label visibility
    plt.tight_layout(pad=2.0)
    
    # Create directory path for matrices
    os.makedirs('analysis/matrices', exist_ok=True)
    
    # Save the matrix info to a text file
    info_filename = f'analysis/matrices/{label_type}_matrix_info_{prediction_threshold}_{num_samples}.txt'
    with open(info_filename, 'w') as f:
        f.write(f"{label_type.capitalize()} co-occurrence matrix with threshold={prediction_threshold}, samples={num_samples}\n")
        f.write(f"Number of labels: {n_themes}\n\n")
        f.write("Labels included in the matrix:\n")
        for i, theme in enumerate(active_themes, 1):
            f.write(f"{i}. {theme}\n")
    
    # Save the plot with higher resolution and include threshold in filename
    output_filename = f'analysis/matrices/{label_type}_confusion_matrix_{prediction_threshold}_{num_samples}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=1.0)
    plt.close()
    
    print(f"{label_type.capitalize()} confusion matrix saved as '{output_filename}'")
    print(f"Label information saved to '{info_filename}'")
    return normalized_matrix

# Function to generate all confusion matrices
def generate_all_matrices(dataset, label_names, theme_metrics, cached_predictions, model, device):
    """Generate all confusion matrices: combined, themes-only, and openings-only at multiple thresholds."""
    print("\nGenerating all confusion matrices...")
    
    global label_filter, prediction_threshold
    
    # Store the original threshold value
    original_threshold = prediction_threshold
    
    # Define the thresholds to use for generating matrices
    thresholds = [0.1, 0.3, 0.5]
    
    # Generate matrices for each threshold
    for threshold in thresholds:
        # Set the global threshold for this iteration
        prediction_threshold = threshold
        print(f"\n=== Generating matrices for threshold {threshold} ===")
        
        # 1. Generate matrix for all labels (both themes and openings)
        print("\nGenerating combined themes+openings matrix...")
        label_filter = lambda x: True  # No filtering, include all labels
        create_cooccurrence_matrix(label_names, "combined", dataset, model, device, 
                                  [m['theme'] for m in theme_metrics], cached_predictions)
    
        # 2. Generate matrix for themes only 
        print("\nGenerating themes-only matrix...")
        themes_only = dataset.get_themes_only()
        label_filter = lambda x: dataset.is_theme(x)  # Only include themes
        create_cooccurrence_matrix(label_names, "theme", dataset, model, device, None, cached_predictions)
    
        # 3. Generate matrix for openings only
        print("\nGenerating openings-only matrix...")
        openings_only = dataset.get_openings_only()
        label_filter = lambda x: dataset.is_opening_tag(x)  # Only include opening tags
        create_cooccurrence_matrix(label_names, "opening", dataset, model, device, None, cached_predictions)
    
    # Restore the original threshold
    prediction_threshold = original_threshold

# Utility function to suppress multiprocessing cleanup errors
def suppress_mp_cleanup_errors():
    """Apply patches to suppress multiprocessing cleanup errors."""
    import os
    import sys
    import multiprocessing.util
    import threading
    
    # Original handlers from multiprocessing
    original_finalizer_handler = multiprocessing.util._run_finalizers
    original_stderr = sys.stderr
    
    # Create a filter to suppress specific error messages
    class MultiprocessingErrorFilter:
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr
            
        def write(self, message):
            # Filter out known benign multiprocessing cleanup errors
            if "OSError: [Errno 16] Device or resource busy: '.nfs" in message:
                return  # Suppress this error message
            if "Exception ignored in" in message and "multiprocessing" in message:
                return  # Suppress generic multiprocessing cleanup errors
            # Pass through all other messages
            self.original_stderr.write(message)
            
        def flush(self):
            self.original_stderr.flush()
            
        def isatty(self):
            return self.original_stderr.isatty()
    
    # Patched handler to silence specific errors
    def patched_finalizer_handler():
        try:
            return original_finalizer_handler()
        except OSError as e:
            if e.errno == 16 and "Device or resource busy: '.nfs" in str(e):
                # Silently ignore NFS device busy errors during cleanup
                pass
            else:
                raise  # Re-raise other errors
    
    # Apply the patches
    multiprocessing.util._run_finalizers = patched_finalizer_handler
    
    # Replace stderr to filter out error messages
    sys.stderr = MultiprocessingErrorFilter(sys.stderr)
    
    # Patch thread excepthook to suppress multiprocessing-related errors
    original_thread_excepthook = threading.excepthook
    
    def patched_thread_excepthook(args):
        # Only pass to original handler if not a multiprocessing cleanup error
        if args.exc_type is OSError and args.exc_value.errno == 16 and ".nfs" in str(args.exc_value):
            return
        original_thread_excepthook(args)
    
    threading.excepthook = patched_thread_excepthook
    
    return original_stderr

# Function to run the main test logic
def run_test():
    """Main test function that evaluates the model and generates metrics."""
    global theme_metrics  # Need to make this global for matrix functions
    
    # Create dataset first to get number of labels
    dataset = ChessPuzzleDataset("lichess_db_puzzle.csv")
    num_labels = len(dataset.get_theme_names())

    # Initialize model with configuration from YAML file if it exists
    import os
    import yaml

    # Default model configuration
    model_config = {
        "num_labels": num_labels,
        "nlayers": 2,
        "embed_dim": 64,
        "inner_dim": 320, 
        "attention_dim": 64,
        "use_1x1conv": True,
        "dropout": 0.5
    }

    # Check if model_config.yaml exists and load it if present
    config_path = 'model_config.yaml'
    if os.path.exists(config_path):
        print(f"Found {config_path}, loading configuration...")
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Update the default config with values from yaml
            for key, value in yaml_config.items():
                if key in model_config:
                    model_config[key] = value
        print("Configuration loaded from YAML file")

    # Always ensure num_labels is set correctly
    model_config["num_labels"] = num_labels

    # Print model configuration
    print("\nInitializing model with configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    # Create model instance with loaded configuration
    model = Model(**model_config)

    # Device setup - use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() and args.optimized else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # Load the checkpoint and extract just the model state dict
    print(f"Using checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    # Get all label names (themes + opening tags)
    label_names = dataset.get_theme_names()

    # Initialize confusion matrix accumulators
    confusion_stats = {
        'true_positive': np.zeros(len(label_names)),
        'false_positive': np.zeros(len(label_names)),
        'false_negative': np.zeros(len(label_names)),
        'true_negative': np.zeros(len(label_names))
    }
    
    # Cache for predictions to avoid duplicate forward passes
    cached_predictions = {}
    
    # Optimized path using batching
    if args.optimized:
        from torch.utils.data import DataLoader
        
        # Create DataLoader for batch processing
        batch_size = 64  # Adjust based on your GPU memory
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if args.optimized else 0
        )
        
        # Process samples in batches
        progress_bar = tqdm(enumerate(dataloader), total=min(num_samples // batch_size + 1, len(dataloader)), 
                            desc="Processing batches", unit="batch")
        
        samples_processed = 0
        for batch_idx, batch in progress_bar:
            if samples_processed >= num_samples:
                break
                
            batch_size = min(batch['board'].size(0), num_samples - samples_processed)
            if batch_size <= 0:
                break
                
            # Update progress description
            progress_bar.set_description(f"Batch {batch_idx+1} ({samples_processed+1}-{samples_processed+batch_size}/{num_samples})")
            
            # Process each sample in the batch
            inputs = batch['board'][:batch_size].unsqueeze(1).to(device)
            targets = batch['themes'][:batch_size]
            fens = [batch['fen'][i] for i in range(batch_size)]
            
            with torch.no_grad():
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)  # Convert logits to probabilities
                
            # Process each sample in the batch
            for i in range(batch_size):
                sample_idx = samples_processed + i
                
                # Cache predictions for later use
                cached_predictions[sample_idx] = outputs[i].cpu()
                
                out = outputs[i]
                target = targets[i]
                
                # Compute metrics and store stats for this sample
                stats = compute_multilabel_confusion_matrix(out.cpu(), target, threshold=prediction_threshold)
                
                # Debug print for first few samples
                if sample_idx < 3:
                    # Temporarily clear the progress bar
                    progress_bar.clear()
                    
                    # Get predicted themes
                    predicted_probs, predicted_indices = torch.where(out > prediction_threshold, out, torch.zeros_like(out)).cpu().squeeze().sort(descending=True)
                    predicted_themes = [(label_names[idx.item()], f"{predicted_probs[j].item():.3f}") 
                                        for j, idx in enumerate(predicted_indices) 
                                        if predicted_probs[j].item() > prediction_threshold]
                    
                    # Get actual themes
                    actual_themes = [label_names[i] for i, is_theme in enumerate(target) if is_theme == 1]
                    
                    print(f"\nDebug - Sample {sample_idx+1}:")
                    print(f"Target sum: {target.sum().item()}")
                    print(f"Predictions > {prediction_threshold}: {(out > prediction_threshold).sum().item()}")
                    
                    # Print predicted themes with their probabilities
                    print("\nPredicted themes (probability):")
                    for theme, prob in predicted_themes:
                        print(f"  {theme}: {prob}")
                        
                    # Print actual themes
                    print("\nActual themes:")
                    print("  " + ", ".join(actual_themes))
                    
                    print("\nPosition FEN:")
                    print(f"  {fens[i]}")
                    print("-" * 80)
                    
                    print("Sample stats:")
                    print(f"TP: {stats['true_positive'].sum()}")
                    print(f"FP: {stats['false_positive'].sum()}")
                    print(f"FN: {stats['false_negative'].sum()}")
                    print(f"TN: {stats['true_negative'].sum()}")
                    
                    # Update progress bar again
                    progress_bar.set_description(f"Batch {batch_idx+1} ({samples_processed+1}-{samples_processed+batch_size}/{num_samples})")
                    progress_bar.update(0)  # Force refresh without incrementing
                
                # Update confusion matrix statistics
                for key in confusion_stats:
                    confusion_stats[key] += stats[key]
            
            samples_processed += batch_size
            
    # Original non-optimized path
    else:
        # Process samples with tqdm progress bar
        progress_bar = tqdm(range(num_samples), desc="Processing samples", unit="sample")
        for i in progress_bar:
            # Update progress bar description to show current sample number
            progress_bar.set_description(f"Processing sample {i+1}/{num_samples}")
            
            sample = dataset[i]
            input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
            target = sample['themes']

            with torch.no_grad():
                out = model(input_tensor)
                out = torch.sigmoid(out)  # Convert logits to probabilities

            # Cache predictions for later use
            cached_predictions[i] = out.squeeze().cpu()
                
            # Calculate Jaccard index on the raw probabilities vs thresholded predictions
            raw_jaccard = jaccard_similarity(out.squeeze().cpu(), target, threshold=prediction_threshold)

            # Get predicted themes (where probability > threshold)
            predicted_probs, predicted_indices = torch.where(out > prediction_threshold, out, torch.zeros_like(out)).squeeze().cpu().sort(descending=True)
            predicted_themes = [(label_names[idx], f"{predicted_probs[i]:.3f}") for i, idx in enumerate(predicted_indices) if predicted_probs[i] > prediction_threshold]
            
            # Get the theme names only from predicted themes (without probabilities)
            predicted_theme_names = [theme for theme, _ in predicted_themes]

            # Get actual themes
            actual_themes = [label_names[i] for i, is_theme in enumerate(target) if is_theme == 1]

            # Debug print for first few samples (outside of tqdm to avoid confusion)
            if i < 3:
                # Temporarily clear the progress bar
                progress_bar.clear()
                
                print(f"\nDebug - Sample {i+1}:")
                print(f"Target sum: {target.sum().item()}")
                print(f"Predictions > {prediction_threshold}: {(out.squeeze() > prediction_threshold).sum().item()}")
                
                # Print predicted themes with their probabilities
                print("\nPredicted themes (probability):")
                for theme, prob in predicted_themes:
                    print(f"  {theme}: {prob}")
                    
                # Print actual themes
                print("\nActual themes:")
                print("  " + ", ".join(actual_themes))
                
                print("\nPosition FEN:")
                print(f"  {sample['fen']}")
                print("-" * 80)
                
                # Update progress bar stats for display
                progress_bar.set_description(f"Processing sample {i+1}/{num_samples}")
                progress_bar.update(0)  # Force refresh without incrementing

            # Compute confusion matrix statistics for this sample
            stats = compute_multilabel_confusion_matrix(out.squeeze().cpu(), target, threshold=prediction_threshold)
            
            # Debug print for first few samples
            if i < 3:
                # Temporarily clear the progress bar
                progress_bar.clear()
                
                print("Sample stats:")
                print(f"TP: {stats['true_positive'].sum()}")
                print(f"FP: {stats['false_positive'].sum()}")
                print(f"FN: {stats['false_negative'].sum()}")
                print(f"TN: {stats['true_negative'].sum()}")
                
                # Update progress bar again
                progress_bar.set_description(f"Processing sample {i+1}/{num_samples}")
                progress_bar.update(0)  # Force refresh without incrementing
            
            for key in confusion_stats:
                confusion_stats[key] += stats[key]

    # Print confusion matrix statistics
    print("\n=== Overall Confusion Matrix Statistics ===")
    print("\nPer-theme statistics:")
    print(f"{'Theme':<30} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print("-" * 85)

    for i, theme in enumerate(label_names):
        tp = confusion_stats['true_positive'][i]
        fp = confusion_stats['false_positive'][i]
        fn = confusion_stats['false_negative'][i]
        tn = confusion_stats['true_negative'][i]
        
        # Calculate metrics (handling division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{theme:<30} {tp:6.0f} {fp:6.0f} {fn:6.0f} {tn:6.0f} {precision:9.3f} {recall:9.3f} {f1:9.3f}")

    # Calculate macro-averaged metrics
    print("\nMacro-averaged metrics:")

    # Calculate per-theme metrics first
    global theme_metrics  # make accessible to matrix functions
    theme_metrics = []
    for i, theme in enumerate(label_names):
        tp = confusion_stats['true_positive'][i]
        fp = confusion_stats['false_positive'][i]
        fn = confusion_stats['false_negative'][i]
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if tp + fp + fn > 0:  # Only include themes that appeared
            theme_metrics.append({
                'theme': theme,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            })

    # Sort by F1 score for better readability
    theme_metrics.sort(key=lambda x: x['f1'], reverse=True)

    print("\nPer-theme metrics (non-zero only):")
    print(f"{'Theme':<30} {'Precision':>9} {'Recall':>9} {'F1':>9} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 80)
    for metric in theme_metrics:
        print(f"{metric['theme']:<30} {metric['precision']:9.3f} {metric['recall']:9.3f} {metric['f1']:9.3f} {metric['tp']:6.0f} {metric['fp']:6.0f} {metric['fn']:6.0f}")

    # Calculate averages
    if theme_metrics:
        avg_precision = np.mean([m['precision'] for m in theme_metrics])
        avg_recall = np.mean([m['recall'] for m in theme_metrics])
        avg_f1 = np.mean([m['f1'] for m in theme_metrics])

        print(f"\nNumber of active themes: {len(theme_metrics)}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"Average F1 Score: {avg_f1:.3f}")
        
        # Also calculate F1 from averaged precision and recall for comparison
        combined_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        print(f"F1 Score (from averaged P&R): {combined_f1:.3f}")

    # Generate visual confusion matrices
    print("\nGenerating visual confusion matrices...")
    generate_all_matrices(dataset, label_names, theme_metrics, cached_predictions, model, device)
    
    return confusion_stats, dataset, label_names, model, device, cached_predictions

# Main execution 
if __name__ == "__main__":
    # Get command line arguments
    args = parse_args()
    num_samples = args.num_samples
    prediction_threshold = args.threshold  # Threshold for converting probabilities to binary predictions
    checkpoint_file = args.checkpoint
    
    print(f"Testing with {num_samples} samples and prediction threshold of {prediction_threshold}")
    print(f"Optimized mode: {'ON' if args.optimized else 'OFF'}")
    
    # Initialize theme_metrics as global for matrix functions
    theme_metrics = []
    
    # Save original stderr before patching
    import sys
    original_stderr = sys.stderr
    
    # Start timing
    start_time = time.time()
    
    # Apply multiprocessing error suppression right at the start
    original_stderr = suppress_mp_cleanup_errors()
    
    try:
        if args.profile:
            # Run with profiling
            print("Running with profiling enabled")
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Run the test
            confusion_stats, dataset, label_names, model, device, cached_predictions = run_test()
            
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
            ps.print_stats(50)  # Print top 50 functions by cumulative time
            
            # Save profiling results to file
            with open(args.profile_output, 'w') as f:
                f.write(s.getvalue())
            print(f"Profiling results saved to {args.profile_output}")
        else:
            # Run without profiling
            confusion_stats, dataset, label_names, model, device, cached_predictions = run_test()
        
        # End timing and display
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        print("\n✅ Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user. Cleaning up...")
    except Exception as e:
        print(f"\n❌ Error during test execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
        
        # Handle any remaining cleanup
        try:
            import gc
            gc.collect()  # Force garbage collection to clean up resources
            
            # Clean up any remaining multiprocessing resources
            import multiprocessing
            if hasattr(multiprocessing, 'active_children'):
                for child in multiprocessing.active_children():
                    try:
                        child.terminate()
                    except:
                        pass
        except Exception as cleanup_error:
            # Don't let cleanup errors mask the original error
            print(f"Note: Additional error during cleanup: {cleanup_error}")
            pass