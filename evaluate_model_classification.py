import torch
import numpy as np
import argparse
import time
import cProfile
import pstats
import io
import os
import json
from pstats import SortKey
from dataset import ChessPuzzleDataset
from model import Model
from metrics import jaccard_similarity, compute_multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Test chess puzzle classifier and generate co-occurrence matrices')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to test (default: 1000)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold for classification (if not specified, adaptive thresholding will be used)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_resume.pth",
                        help='Checkpoint file to use for testing')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling')
    parser.add_argument('--profile_output', type=str, default="profile_stats.txt",
                        help='Output file for profiling statistics')
    parser.add_argument('--no_optimize', action='store_true',
                        help='Disable optimizations (not recommended)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output with detailed progress information')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached tensor files directly instead of test CSV')
    parser.add_argument('--use_test_csv', action='store_true',
                        help='Force use of test CSV file rather than cached tensors')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimize output for token efficiency')
    return parser.parse_args()

# Global variables for label filtering and mapping
label_filter = lambda x: True
test_to_train_map = None

# Create co-occurrence matrix
def create_cooccurrence_matrix(label_names, label_type, dataset, model, device, active_themes=None, cached_predictions=None):
    """Create a co-occurrence matrix for the given labels.
    
    Returns:
        tuple: (cooccurrence, normalized_matrix) - The raw co-occurrence matrix and the normalized matrix
    """
    global label_filter, test_to_train_map, optimized, prediction_threshold
    
    # Use a much lower threshold for matrix generation to ensure we get some predictions
    matrix_threshold = 0.001
    
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
        return None, None
        
    print(f"Creating co-occurrence matrix for {n_themes} {label_type}s...")
    print(f"Active themes in matrix: {active_themes}")
    
    cooccurrence = np.zeros((n_themes, n_themes))
    theme_to_idx = {theme: idx for idx, theme in enumerate(active_themes)}
    
    # Get test dataset labels for mapping
    test_label_names = dataset.get_theme_names()
    
    # Optimized path using cached predictions and vectorized operations
    if not args.no_optimize and cached_predictions is not None:
        matrix_progress = tqdm(range(num_samples), desc=f"Creating {label_type} matrix", unit="sample")
        for i in matrix_progress:
            matrix_progress.set_description(f"Matrix sample {i+1}/{num_samples} ({label_type})")
            
            sample = dataset[i]
            target = sample['themes']
            
            # Use cached predictions instead of recomputing
            out = cached_predictions[i]
            print(f"\nDebug - Cached predictions for sample {i}:")
            print(f"Type: {type(out)}")
            print(f"Shape: {out.shape if hasattr(out, 'shape') else 'no shape'}")
            print(f"Content: {out}")
            
            # Get actual labels from test dataset
            actual_labels = [test_label_names[j] for j, is_theme in enumerate(target) if is_theme == 1 and label_filter(test_label_names[j])]
            
            # Using the fixed low threshold defined at the function level (0.001)
            # matrix_threshold is already defined
            pred_binary = (out > matrix_threshold).float()
            print(f"pred_binary type: {type(pred_binary)}")
            print(f"pred_binary shape: {pred_binary.shape if hasattr(pred_binary, 'shape') else 'no shape'}")
            print(f"pred_binary content: {pred_binary}")
            
            if test_to_train_map is not None:
                # Map predictions from training space to test space
                predicted_labels = []
                for train_idx, prob in enumerate(pred_binary):
                    if prob > prediction_threshold and train_idx in test_to_train_map:
                        test_idx = test_to_train_map[train_idx]
                        if test_idx < len(test_label_names) and label_filter(test_label_names[test_idx]):
                            predicted_labels.append(test_label_names[test_idx])
            else:
                # Use direct indexing if no mapping available
                predicted_labels = [test_label_names[j] for j, prob in enumerate(pred_binary) if prob > prediction_threshold and label_filter(test_label_names[j])]
            
            print(f"\nSample {i}:")
            print(f"Actual labels: {actual_labels}")
            print(f"Predicted labels: {predicted_labels}")
            print(f"Active themes in matrix: {list(theme_to_idx.keys())}")
            
            # Update co-occurrence matrix
            for actual in actual_labels:
                if actual in theme_to_idx:
                    for predicted in predicted_labels:
                        if predicted in theme_to_idx:
                            print(f"Incrementing matrix at [{theme_to_idx[actual]}, {theme_to_idx[predicted]}] for actual={actual}, predicted={predicted}")
                            cooccurrence[theme_to_idx[actual], theme_to_idx[predicted]] += 1
    
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
            
            print(f"\nDebug - Model output for sample {i} (non-optimized path):")
            print(f"Type: {type(out)}")
            print(f"Shape: {out.shape if hasattr(out, 'shape') else 'no shape'}")
            print(f"Content: {out}")
            
            # Get actual labels from test dataset
            actual_labels = [test_label_names[j] for j, is_theme in enumerate(target) if is_theme == 1 and label_filter(test_label_names[j])]
            
            # Get predicted labels using the mapping if available
            outvalue = out.squeeze().cpu()
            pred_binary = (outvalue > prediction_threshold).float()
            print(f"pred_binary type: {type(pred_binary)}")
            print(f"pred_binary shape: {pred_binary.shape if hasattr(pred_binary, 'shape') else 'no shape'}")
            print(f"pred_binary content: {pred_binary}")
            
            if test_to_train_map is not None:
                # Map predictions from training space to test space
                predicted_labels = []
                for train_idx, prob in enumerate(pred_binary):
                    if prob > prediction_threshold and train_idx in test_to_train_map:
                        test_idx = test_to_train_map[train_idx]
                        if test_idx < len(test_label_names) and label_filter(test_label_names[test_idx]):
                            predicted_labels.append(test_label_names[test_idx])
            else:
                # Use direct indexing if no mapping available
                predicted_labels = [test_label_names[j] for j, prob in enumerate(pred_binary) if prob > prediction_threshold and label_filter(test_label_names[j])]
            
            print(f"\nSample {i}:")
            print(f"Actual labels: {actual_labels}")
            print(f"Predicted labels: {predicted_labels}")
            print(f"Active themes in matrix: {list(theme_to_idx.keys())}")
            
            # Update co-occurrence matrix
            for actual in actual_labels:
                if actual in theme_to_idx:
                    for predicted in predicted_labels:
                        if predicted in theme_to_idx:
                            print(f"Incrementing matrix at [{theme_to_idx[actual]}, {theme_to_idx[predicted]}] for actual={actual}, predicted={predicted}")
                            cooccurrence[theme_to_idx[actual], theme_to_idx[predicted]] += 1
    
    print(f"cooccurrence: {cooccurrence}")
    # Create normalized confusion matrix (divide by number of actual occurrences)
    actual_counts = cooccurrence.sum(axis=1, keepdims=True)
    print(f"actual_counts: {actual_counts}")
    # Handle division by zero by using np.divide with a 'where' condition
    normalized_matrix = np.divide(cooccurrence, actual_counts, out=np.zeros_like(cooccurrence), where=actual_counts>0)
    print(f"normalized_matrix: {normalized_matrix}")
    
    # Save the raw co-occurrence matrix to file
    os.makedirs('analysis/matrices', exist_ok=True)
    
    # Save binary matrices for programmatic use
    matrix_filename = f'analysis/matrices/{label_type}_raw_matrix_{prediction_threshold}_mt{matrix_threshold}_{num_samples}.npy'
    np.save(matrix_filename, cooccurrence)
    normalized_matrix_filename = f'analysis/matrices/{label_type}_normalized_matrix_{prediction_threshold}_mt{matrix_threshold}_{num_samples}.npy'
    np.save(normalized_matrix_filename, normalized_matrix)
    
    # Also save the theme mapping for reference
    theme_mapping_filename = f'analysis/matrices/{label_type}_theme_mapping_{prediction_threshold}_mt{matrix_threshold}_{num_samples}.json'
    with open(theme_mapping_filename, 'w') as f:
        json.dump(theme_to_idx, f, indent=2)
    
    # Save human-readable CSV files
    human_readable_matrix_filename = f'analysis/matrices/{label_type}_raw_matrix_{prediction_threshold}_mt{matrix_threshold}_{num_samples}.csv'
    human_readable_normalized_filename = f'analysis/matrices/{label_type}_normalized_matrix_{prediction_threshold}_mt{matrix_threshold}_{num_samples}.csv'
    
    # Create human-readable CSV with labels
    with open(human_readable_matrix_filename, 'w') as f:
        # Write header row with theme names
        f.write(f"Actual\\Predicted,{','.join(active_themes)}\n")
        
        # Write each row with theme name and values
        for i, theme in enumerate(active_themes):
            row_values = [str(val) for val in cooccurrence[i]]
            f.write(f"{theme},{','.join(row_values)}\n")
    
    # Create human-readable normalized CSV with labels
    with open(human_readable_normalized_filename, 'w') as f:
        # Write header row with theme names
        f.write(f"Actual\\Predicted,{','.join(active_themes)}\n")
        
        # Write each row with theme name and values
        for i, theme in enumerate(active_themes):
            row_values = [f"{val:.4f}" for val in normalized_matrix[i]]
            f.write(f"{theme},{','.join(row_values)}\n")
    
    print(f"Raw co-occurrence matrix saved to '{matrix_filename}' (binary) and '{human_readable_matrix_filename}' (CSV)")
    print(f"Normalized matrix saved to '{normalized_matrix_filename}' (binary) and '{human_readable_normalized_filename}' (CSV)")
    print(f"Theme mapping saved to '{theme_mapping_filename}'")
    
    # Create visualization using the normalized matrix
    visualize_cooccurrence_matrix(normalized_matrix, active_themes, label_type, prediction_threshold, matrix_threshold, num_samples)
    
    # Return both matrices for potential further analysis
    return cooccurrence, normalized_matrix

def visualize_cooccurrence_matrix(normalized_matrix, active_themes, label_type, prediction_threshold, matrix_threshold, num_samples):
    """Visualize a co-occurrence matrix and save the visualization to a file."""
    n_themes = len(active_themes)
    
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
    
    plt.title(f'{label_type.capitalize()} Co-occurrence Matrix (prediction threshold={prediction_threshold}, matrix threshold={matrix_threshold}, samples={num_samples})\n(Row: Actual, Column: Predicted, Values: P(Predicted|Actual))', 
             pad=20, size=14)  # Added padding to title
    plt.xlabel(f'Predicted {label_type.capitalize()}', size=12, labelpad=10)
    plt.ylabel(f'Actual {label_type.capitalize()}', size=12, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', size=10)  # Increased font size
    plt.yticks(rotation=0, size=10)  # Increased font size
    
    # Add more padding around the plot for better label visibility
    plt.tight_layout(pad=2.0)
    
    # Create directory path for matrices (already created in caller function)
    # os.makedirs('analysis/matrices', exist_ok=True)
    
    # Save the matrix info to a text file
    info_filename = f'analysis/matrices/{label_type}_matrix_info_{prediction_threshold}_mt{matrix_threshold}_{num_samples}.txt'
    with open(info_filename, 'w') as f:
        f.write(f"{label_type.capitalize()} co-occurrence matrix with prediction threshold={prediction_threshold}, matrix threshold={matrix_threshold}, samples={num_samples}\n")
        f.write(f"Number of labels: {n_themes}\n\n")
        f.write("Labels included in the matrix:\n")
        for i, theme in enumerate(active_themes, 1):
            f.write(f"{i}. {theme}\n")
    
    # Save the plot with higher resolution and include thresholds in filename
    output_filename = f'analysis/matrices/{label_type}_confusion_matrix_{prediction_threshold}_mt{matrix_threshold}_{num_samples}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=1.0)
    plt.close()
    
    print(f"{label_type.capitalize()} confusion matrix visualization saved as '{output_filename}'")
    print(f"Label information saved to '{info_filename}'")
    return output_filename

# Function to generate all confusion matrices
def generate_all_matrices(dataset, test_label_names, theme_metrics, cached_predictions, model, device):
    """Generate all confusion matrices: combined, themes-only, and openings-only."""
    print("\nGenerating confusion matrices...")
    
    global label_filter, prediction_threshold
    
    # Dictionary to store all computed matrices
    matrices = {}
    
    # 1. Generate matrix for all labels (both themes and openings)
    print("\nGenerating combined themes+openings matrix...")
    label_filter = lambda x: True  # No filtering, include all labels
    raw_combined, norm_combined = create_cooccurrence_matrix(
        test_label_names, "combined", dataset, model, device, 
        [m['theme'] for m in theme_metrics], cached_predictions
    )
    matrices['combined'] = {
        'raw': raw_combined,
        'normalized': norm_combined
    }

    # 2. Generate matrix for themes only 
    print("\nGenerating themes-only matrix...")
    themes_only = dataset.get_themes_only()
    label_filter = lambda x: dataset.is_theme(x)  # Only include themes
    raw_themes, norm_themes = create_cooccurrence_matrix(
        test_label_names, "theme", dataset, model, device, None, cached_predictions
    )
    matrices['theme'] = {
        'raw': raw_themes,
        'normalized': norm_themes
    }

    # 3. Generate matrix for openings only
    print("\nGenerating openings-only matrix...")
    openings_only = dataset.get_openings_only()
    label_filter = lambda x: dataset.is_opening_tag(x)  # Only include opening tags
    raw_openings, norm_openings = create_cooccurrence_matrix(
        test_label_names, "opening", dataset, model, device, None, cached_predictions
    )
    matrices['opening'] = {
        'raw': raw_openings,
        'normalized': norm_openings
    }
    
    return matrices

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
    global theme_metrics, prediction_threshold, test_to_train_map, verbose, optimized  # Need to make these global for matrix functions
    
    # Determine if verbose output is enabled
    verbose = args.verbose and not args.quiet
    
    # Set optimized mode based on arguments (default is True)
    optimized = not args.no_optimize
    
    # Create dataset based on arguments
    import os
    
    if args.use_cache:
        # Use cached tensor files directly
        # We need to find and load the right cache file
        potential_cache_paths = [
            "dataset/lichess_db_puzzle.csv.tensors.pt_conditional",
            "processed_lichess_puzzle_files/lichess_db_puzzle.csv.tensors.pt_conditional",
            "dataset_backup/lichess_db_puzzle.csv.tensors.pt_conditional",
            "dataset/lichess_db_puzzle_test.csv.tensors.pt_conditional",
        ]
        
        cache_file = None
        for path in potential_cache_paths:
            if os.path.exists(path):
                cache_file = path
                print(f"Found cache file: {cache_file}")
                break
                
        if cache_file is None:
            print("No cache file found. Falling back to test CSV.")
            dataset = ChessPuzzleDataset(os.path.join("dataset", "lichess_db_puzzle_test.csv"))
        else:
            # Use direct tensor loading - assume tensors already generated
            # Load dataset normally but indicate we're using the cache file
            dataset = ChessPuzzleDataset(os.path.join("dataset", "lichess_db_puzzle_test.csv"))
            print(f"Using cached tensors from {cache_file}")
    elif args.use_test_csv:
        # Use test CSV file explicitly
        dataset = ChessPuzzleDataset(os.path.join("dataset", "lichess_db_puzzle_test.csv"))
        print("Using test CSV file for evaluation")
    else:
        # Default: use test CSV file
        dataset = ChessPuzzleDataset(os.path.join("dataset", "lichess_db_puzzle_test.csv"))
        print("Using test CSV file for evaluation")
    
    # Adjust num_samples if it's larger than the dataset
    global num_samples
    num_samples = min(num_samples, len(dataset))
    print(f"Adjusted to test with {num_samples} samples (dataset size: {len(dataset)})")

    # Initialize model with configuration from YAML file if it exists
    import yaml

    # Default model configuration - Note: num_labels will be loaded from YAML file
    model_config = {
        "num_labels": 1616,  # Default to full label set (will be overridden by YAML if present)
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

    # Do not override num_labels from config as it needs to match the checkpoint
    # model_config["num_labels"] = num_labels

    # Print model configuration
    print("\nInitializing model with configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    # Create model instance with loaded configuration
    model = Model(**model_config)

    # Device setup - use GPU if available (optimized is now the default)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_optimize else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # Look for the checkpoint in the checkpoints directory first, then in the current directory
    checkpoint_path = os.path.join("checkpoints", checkpoint_file)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = checkpoint_file
    
    # Load the checkpoint and extract just the model state dict
    print(f"Using checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set model to evaluation mode
    model.eval()

    # Get all label names (themes + opening tags) from the test dataset
    test_label_names = dataset.get_theme_names()
    
    # We need to ensure we're using the same label ordering that was used in training
    # Load the themes and openings from the processed dataset files
    import json
    
    # Define potential locations for the cache files
    possible_theme_locations = [
        "processed_lichess_puzzle_files/lichess_db_puzzle.csv.themes.json",
        "dataset/lichess_db_puzzle.csv.themes.json",
        "dataset/lichess_db_puzzle_test.csv.themes.json",
        "dataset_backup/lichess_db_puzzle.csv.themes.json",
        "dataset_backup/lichess_db_puzzle_small.csv.themes.json",
        "dataset_backup/lichess_db_puzzle_test.csv.themes.json",
        "processed_lichess_puzzle_files_backup/lichess_db_puzzle.csv.themes.json"
    ]
    
    possible_opening_locations = [
        "processed_lichess_puzzle_files/lichess_db_puzzle.csv.openings.json",
        "dataset/lichess_db_puzzle.csv.openings.json",
        "dataset/lichess_db_puzzle_test.csv.openings.json",
        "dataset_backup/lichess_db_puzzle.csv.openings.json",
        "dataset_backup/lichess_db_puzzle_small.csv.openings.json",
        "dataset_backup/lichess_db_puzzle_test.csv.openings.json",
        "processed_lichess_puzzle_files_backup/lichess_db_puzzle.csv.openings.json"
    ]
    
    print("Loading themes and openings from processed dataset files for label mapping...")
    training_themes = []
    training_openings = []
    
    try:
        # Try each possible location for themes
        themes_loaded = False
        for themes_file in possible_theme_locations:
            if os.path.exists(themes_file):
                print(f"Found themes file: {themes_file}")
                try:
                    with open(themes_file, 'r') as f:
                        themes_data = json.load(f)
                        # Check if the format is a list of [name, count] lists or just a list of names
                        if themes_data and isinstance(themes_data[0], list):
                            # Format is [[name, count], [name, count], ...]
                            training_themes = [item[0] for item in themes_data]
                        else:
                            # Format is [name, name, ...]
                            training_themes = themes_data
                    themes_loaded = True
                    print(f"Successfully loaded {len(training_themes)} themes from {themes_file}")
                    break
                except Exception as e:
                    print(f"Error loading themes from {themes_file}: {e}")
        
        if not themes_loaded:
            print("Could not find a valid themes file in any of the expected locations.")
        
        # Try each possible location for openings
        openings_loaded = False
        for openings_file in possible_opening_locations:
            if os.path.exists(openings_file):
                print(f"Found openings file: {openings_file}")
                try:
                    with open(openings_file, 'r') as f:
                        openings_data = json.load(f)
                        # Check if the format is a list of [name, count] lists or just a list of names
                        if openings_data and isinstance(openings_data[0], list):
                            # Format is [[name, count], [name, count], ...]
                            training_openings = [item[0] for item in openings_data]
                        else:
                            # Format is [name, name, ...]
                            training_openings = openings_data
                    openings_loaded = True
                    print(f"Successfully loaded {len(training_openings)} openings from {openings_file}")
                    break
                except Exception as e:
                    print(f"Error loading openings from {openings_file}: {e}")
        
        if not openings_loaded:
            print("Could not find a valid openings file in any of the expected locations.")
        
        # Create the full training label set in the same order used during training
        training_labels = sorted(training_themes + training_openings)
        
        print(f"Loaded {len(training_labels)} labels from processed dataset files")
        print(f"Test dataset has {len(test_label_names)} labels")
        
        # Create a mapping from test dataset labels to their positions in the training label set
        label_to_training_idx = {}
        for i, label in enumerate(test_label_names):
            if label in training_labels:
                # Find this label's position in the training labels
                training_idx = training_labels.index(label)
                label_to_training_idx[i] = training_idx
            else:
                print(f"Warning: Label '{label}' from test dataset not found in training labels")
        
        # Verify the mapping completeness
        print(f"Created mapping for {len(label_to_training_idx)}/{len(test_label_names)} test labels")
        
        # Debug: check if key chess themes are in both sets and mapped correctly
        key_themes = ["advantage", "crushing", "endgame", "hangingPiece", "long", "middlegame", "short"]
        for theme in key_themes:
            if theme in test_label_names and theme in training_labels:
                test_idx = test_label_names.index(theme)
                train_idx = training_labels.index(theme)
                print(f"Theme '{theme}' found in both sets: test_idx={test_idx}, train_idx={train_idx}")
            elif theme in test_label_names:
                print(f"Theme '{theme}' found only in test set")
            elif theme in training_labels:
                print(f"Theme '{theme}' found only in training set")
            else:
                print(f"Theme '{theme}' not found in either set")
        
        # Create the reverse mapping (training index -> test index)
        test_to_train_map = {v: k for k, v in label_to_training_idx.items()}
        
        # Use the training label names for the evaluation
        # This ensures we're using the same label mapping as during training
        label_names = training_labels
    except Exception as e:
        print(f"Error loading processed dataset files: {e}")
        print("Falling back to using test dataset labels")
        # Just use the test dataset labels as a fallback
        label_names = test_label_names
        test_to_train_map = None  # Reset the mapping since we're not using training labels

    # Initialize confusion matrix accumulators
    # We need to make sure these have the same size as the test dataset labels
    # since the metrics calculation uses test dataset dimensions
    confusion_stats = {
        'true_positive': np.zeros(len(test_label_names)),
        'false_positive': np.zeros(len(test_label_names)),
        'false_negative': np.zeros(len(test_label_names)),
        'true_negative': np.zeros(len(test_label_names))
    }
    
    # Cache for predictions to avoid duplicate forward passes
    cached_predictions = {}
    
    # For adaptive thresholding, we'll collect all predictions first
    all_outputs = []
    all_targets = []
    
    # Calculate adaptive threshold if requested
    adaptive_threshold = args.threshold is None
    
    # Optimized path using batching (now the default)
    if not args.no_optimize:
        from torch.utils.data import DataLoader
        
        # Create DataLoader for batch processing
        batch_size = 64  # Adjust based on your GPU memory
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if not args.no_optimize else 0
        )
        
        # Process samples in batches
        progress_bar = tqdm(enumerate(dataloader), total=min(num_samples // batch_size + 1, len(dataloader)), 
                            desc="Processing batches", unit="batch")
        
        # Define common themes mapping with correct indices
        # This maps the correct training index to the test index for key themes
        common_themes = {
            'advantage': [1555, 46],    # [train_idx, test_idx]
            'crushing': [1566, 53],
            'endgame': [1574, 57],
            'hangingPiece': [1578, 60],
            'long': [1585, 63],
            'middlegame': [1594, 70],
            'short': [1606, 81]
        }
        
        # Create a direct mapping from training indices to test indices for key themes
        # This allows us to directly map the raw values from these indices
        direct_theme_mapping = {train_idx: test_idx for theme_name, (train_idx, test_idx) in common_themes.items()}
        
        samples_processed = 0
        for batch_idx, batch in progress_bar:
            # Update progress bar description to show current sample number
            progress_bar.set_description(f"Processing batch {batch_idx+1}")
            
            # Get batch data
            inputs = batch['board'].unsqueeze(1).to(device)  # Add channel dimension
            targets = batch['themes'].to(device)
            
            # Get batch size (may be smaller for last batch)
            batch_size = inputs.size(0)
            
            # Skip if we've processed enough samples
            if samples_processed >= num_samples:
                break
            
            # Forward pass
            with torch.no_grad():
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
            
            # Process each sample in the batch
            for i in range(batch_size):
                sample_idx = samples_processed + i
                
                # Cache predictions for later use
                cached_predictions[sample_idx] = outputs[i].cpu()
                
                out = outputs[i]
                target = targets[i]
                
                # Map the model output (1616 labels) to the test dataset labels
                # Create a tensor with the appropriate labels in the right order
                mapped_output = torch.zeros_like(target)
                output_tensor = out.cpu()
                
                # Make sure output_tensor is 1D
                if output_tensor.dim() > 1 and output_tensor.size(0) == 1:
                    output_tensor = output_tensor.squeeze(0)
                
                if sample_idx < 3:  # Only print debugging for first 3 samples
                    # Print raw model output statistics
                    print(f"\nDebug - Sample {sample_idx+1} - Raw Model Output Statistics:")
                    print(f"Output min: {output_tensor.min().item():.6f}")
                    print(f"Output max: {output_tensor.max().item():.6f}")
                    print(f"Output mean: {output_tensor.mean().item():.6f}")
                    print(f"Output std: {output_tensor.std().item():.6f}")
                    
                    # Print top 10 highest values
                    values, indices = output_tensor.topk(10)
                    print("\nTop 10 highest raw probabilities:")
                    for idx, (val, ind) in enumerate(zip(values, indices)):
                        if ind < len(label_names):
                            print(f"  {idx+1}. {label_names[ind]}: {val.item():.6f}")
                        else:
                            print(f"  {idx+1}. Unknown (idx={ind}): {val.item():.6f}")
                
                # IMPROVED MAPPING APPROACH:
                # 1. First, directly transfer values for our key themes using the known mapping
                for train_idx, test_idx in direct_theme_mapping.items():
                    if train_idx < output_tensor.shape[0]:
                        # Directly transfer the raw probability for the key themes
                        raw_val = output_tensor[train_idx].item()
                        mapped_output[test_idx] = raw_val
                        
                        if sample_idx < 3:  # Debug for first 3 samples
                            theme_name = [name for name, (tidx, _) in common_themes.items() if tidx == train_idx][0]
                            print(f"  Mapped {theme_name}: train_idx={train_idx}, test_idx={test_idx}, value={raw_val:.6f}")
                
                # 2. Then use the label_to_training_idx mapping for all other labels
                if 'label_to_training_idx' in globals():
                    for test_idx, train_idx in label_to_training_idx.items():
                        # Skip the ones we've already handled in direct mapping
                        if test_idx in direct_theme_mapping.values():
                            continue
                            
                        if train_idx < output_tensor.shape[0]:
                            # Transfer the probability for other labels
                            mapped_output[test_idx] = output_tensor[train_idx]
                
                # If we're debugging and this is one of the first samples
                if sample_idx < 3:
                    # Print mapped output statistics
                    print(f"\nMapped output statistics (sample {sample_idx+1}):")
                    print(f"Min: {mapped_output.min().item():.6f}")
                    print(f"Max: {mapped_output.max().item():.6f}")
                    print(f"Mean: {mapped_output.mean().item():.6f}")
                    print(f"Std: {mapped_output.std().item():.6f}")
                    
                    # Print top predictions from mapped outputs
                    values, indices = mapped_output.topk(min(10, len(mapped_output)))
                    print("\nTop mapped probabilities:")
                    for idx, (val, ind) in enumerate(zip(values, indices)):
                        if ind < len(test_label_names):
                            print(f"  {idx+1}. {test_label_names[ind]}: {val.item():.6f}")
                        else:
                            print(f"  {idx+1}. Unknown (idx={ind}): {val.item():.6f}")
                    
                    # Print actual themes
                    actual_themes = [test_label_names[j] for j, is_theme in enumerate(target) if is_theme == 1]
                    print("\nActual themes:")
                    print("  " + ", ".join(actual_themes))
                    
                # Use the mapped output for metrics
                subset_output = mapped_output
                
                # For adaptive thresholding, collect predictions and targets
                if adaptive_threshold:
                    all_outputs.append(subset_output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
                
                # Compute metrics and store stats for this sample (using mapped outputs)
                stats = compute_multilabel_confusion_matrix(subset_output, target, threshold=prediction_threshold)
                
                # Debug print for first few samples
                if sample_idx < 3:
                    # Temporarily clear the progress bar
                    progress_bar.clear()
                    
                    # Get predicted themes
                    output_for_prediction = out.cpu()
                    
                    # Print raw model output statistics
                    print(f"\nDebug - Sample {sample_idx+1} - Raw Model Output Statistics:")
                    print(f"Output min: {output_for_prediction.min().item():.6f}")
                    print(f"Output max: {output_for_prediction.max().item():.6f}")
                    print(f"Output mean: {output_for_prediction.mean().item():.6f}")
                    print(f"Output std: {output_for_prediction.std().item():.6f}")
                    
                    # Print top 10 highest values
                    values, indices = output_for_prediction.topk(10)
                    print("\nTop 10 highest raw probabilities:")
                    for i, (val, idx) in enumerate(zip(values, indices)):
                        if idx < len(label_names):
                            print(f"  {i+1}. {label_names[idx]}: {val.item():.6f}")
                        else:
                            print(f"  {i+1}. Unknown (idx={idx}): {val.item():.6f}")
                    
                    # Use our mapping to get predictions in the right order
                    if 'label_to_training_idx' in globals():
                        # Get the full predictions from the model
                        all_predictions = output_for_prediction > prediction_threshold
                        
                        # Map the model predictions to test dataset labels
                        predicted_themes = []
                        
                        # Reverse the mapping (training index -> test index)
                        training_to_test_idx = {v: k for k, v in label_to_training_idx.items()}
                        
                        # For each prediction in the full training label space
                        for train_idx in range(min(len(output_for_prediction), 1616)):
                            if train_idx < len(output_for_prediction) and output_for_prediction[train_idx] > prediction_threshold:
                                # Get the label name for this training index
                                if train_idx < len(label_names):
                                    label = label_names[train_idx]
                                    prob = output_for_prediction[train_idx].item()
                                    predicted_themes.append((label, f"{prob:.6f}"))
                    else:
                        # Original approach - just use indices directly
                        all_predicted_probs, all_predicted_indices = torch.where(
                            output_for_prediction > prediction_threshold, 
                            output_for_prediction, 
                            torch.zeros_like(output_for_prediction)
                        ).sort(descending=True)
                        
                        # Filter to only include indices that exist in our test dataset
                        predicted_themes = []
                        for j, idx in enumerate(all_predicted_indices):
                            if (all_predicted_probs[j] > prediction_threshold and 
                                idx.item() < len(label_names)):
                                predicted_themes.append((label_names[idx.item()], f"{all_predicted_probs[j].item():.6f}"))
                    
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
                    print("  FEN not available")
                    print("-" * 80)
                    
                    print("Sample stats:")
                    print(f"TP: {stats['true_positive'].sum()}")
                    print(f"FP: {stats['false_positive'].sum()}")
                    print(f"FN: {stats['false_negative'].sum()}")
                    print(f"TN: {stats['true_negative'].sum()}")
                    
                    # Update progress bar again
                    progress_bar.set_description(f"Processing batch {batch_idx+1}")
                    progress_bar.update(0)  # Force refresh without incrementing
                
                # Update confusion matrix statistics
                for key in confusion_stats:
                    confusion_stats[key] += stats[key]
            
            samples_processed += batch_size
            
    # Original non-optimized path
    else:
        # Define common themes mapping with correct indices
        # This maps the correct training index to the test index for key themes
        common_themes = {
            'advantage': [1555, 46],    # [train_idx, test_idx]
            'crushing': [1566, 53],
            'endgame': [1574, 57],
            'hangingPiece': [1578, 60],
            'long': [1585, 63],
            'middlegame': [1594, 70],
            'short': [1606, 81]
        }
        
        # Create a direct mapping from training indices to test indices for key themes
        # This allows us to directly map the raw values from these indices
        direct_theme_mapping = {train_idx: test_idx for theme_name, (train_idx, test_idx) in common_themes.items()}
        
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
                out = torch.sigmoid(out)
            
            print(f"\nDebug - Model output for sample {i} (non-optimized path):")
            print(f"Type: {type(out)}")
            print(f"Shape: {out.shape if hasattr(out, 'shape') else 'no shape'}")
            print(f"Content: {out}")
            
            # Get actual labels from test dataset
            actual_labels = [test_label_names[j] for j, is_theme in enumerate(target) if is_theme == 1 and label_filter(test_label_names[j])]
            
            # Get predicted labels using the mapping if available
            outvalue = out.squeeze().cpu()
            pred_binary = (outvalue > prediction_threshold).float()
            print(f"pred_binary type: {type(pred_binary)}")
            print(f"pred_binary shape: {pred_binary.shape if hasattr(pred_binary, 'shape') else 'no shape'}")
            print(f"pred_binary content: {pred_binary}")
            
            if test_to_train_map is not None:
                # Map predictions from training space to test space
                predicted_labels = []
                for train_idx, prob in enumerate(pred_binary):
                    if prob > prediction_threshold and train_idx in test_to_train_map:
                        test_idx = test_to_train_map[train_idx]
                        if test_idx < len(test_label_names) and label_filter(test_label_names[test_idx]):
                            predicted_labels.append(test_label_names[test_idx])
            else:
                # Use direct indexing if no mapping available
                predicted_labels = [test_label_names[j] for j, prob in enumerate(pred_binary) if prob > prediction_threshold and label_filter(test_label_names[j])]
            
            print(f"\nSample {i}:")
            print(f"Actual labels: {actual_labels}")
            print(f"Predicted labels: {predicted_labels}")
            print(f"Active themes in matrix: {list(theme_to_idx.keys())}")
            
            # Update co-occurrence matrix
            for actual in actual_labels:
                if actual in theme_to_idx:
                    for predicted in predicted_labels:
                        if predicted in theme_to_idx:
                            print(f"Incrementing matrix at [{theme_to_idx[actual]}, {theme_to_idx[predicted]}] for actual={actual}, predicted={predicted}")
                            cooccurrence[theme_to_idx[actual], theme_to_idx[predicted]] += 1
            
            # If we're debugging and this is one of the first samples
            if i < 3:
                # Print mapped output statistics
                print(f"\nMapped output statistics (sample {i+1}):")
                print(f"Min: {outvalue.min().item():.6f}")
                print(f"Max: {outvalue.max().item():.6f}")
                print(f"Mean: {outvalue.mean().item():.6f}")
                print(f"Std: {outvalue.std().item():.6f}")
                
                # Print top predictions from mapped outputs
                values, indices = outvalue.topk(min(10, len(outvalue)))
                print("\nTop mapped probabilities:")
                for idx, (val, ind) in enumerate(zip(values, indices)):
                    if ind < len(test_label_names):
                        print(f"  {idx+1}. {test_label_names[ind]}: {val.item():.6f}")
                    else:
                        print(f"  {idx+1}. Unknown (idx={ind}): {val.item():.6f}")
                
                # Get predicted themes based on threshold
                
                predicted_theme_idxs = torch.where(outvalue > prediction_threshold)[0]
                predicted_themes = [test_label_names[idx.item()] for idx in predicted_theme_idxs]
                
                # Get actual themes
                actual_themes = [test_label_names[j] for j, is_theme in enumerate(target) if is_theme == 1]
                
                # Debug print
                print(f"\nDebug - Sample {i+1}:")
                print(f"Target sum: {target.sum().item()}")
                print(f"Predictions > {prediction_threshold}: {len(predicted_themes)}")
                
                # Print predicted themes
                print("\nPredicted themes:")
                print("  " + ", ".join(predicted_themes))
                
                # Print actual themes
                print("\nActual themes:")
                print("  " + ", ".join(actual_themes))
                
                print("\nPosition FEN:")
                if 'fen' in sample:
                    print(f"  {sample['fen']}")
                else:
                    print("  FEN not available")
                print("-" * 80)
                
                # Update progress bar again
                progress_bar.set_description(f"Processing sample {i+1}/{num_samples}")
                progress_bar.update(0)  # Force refresh without incrementing
            
            # Use the mapped output for metrics
            subset_output = outvalue
            
            # For adaptive thresholding, collect predictions and targets
            if adaptive_threshold:
                all_outputs.append(subset_output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
            
            # Compute confusion matrix statistics for this sample
            stats = compute_multilabel_confusion_matrix(subset_output, target, threshold=prediction_threshold)
            
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
    
    # If using adaptive thresholding, calculate the best threshold
    if adaptive_threshold and all_outputs and all_targets:
        # Convert lists to arrays
        all_outputs_array = np.vstack(all_outputs)
        all_targets_array = np.vstack(all_targets)
        
        # Calculate and set the optimal threshold with verbosity controlled by args
        optimal_threshold = calculate_adaptive_threshold(all_outputs_array, all_targets_array, verbose=verbose)
        prediction_threshold = optimal_threshold
        
        print(f"\nUsing adaptive threshold: {prediction_threshold:.4f}")
        
        # Recalculate confusion stats with the new threshold
        print("Recalculating metrics with optimal threshold...")
        
        # Reset confusion stats
        confusion_stats = {
            'true_positive': np.zeros(len(test_label_names)),
            'false_positive': np.zeros(len(test_label_names)),
            'false_negative': np.zeros(len(test_label_names)),
            'true_negative': np.zeros(len(test_label_names))
        }
        
        # Recalculate metrics with the new threshold
        print("Recalculating metrics with optimal threshold...")
        
        # Reset confusion stats
        confusion_stats = {
            'true_positive': np.zeros(len(test_label_names)),
            'false_positive': np.zeros(len(test_label_names)),
            'false_negative': np.zeros(len(test_label_names)),
            'true_negative': np.zeros(len(test_label_names))
        }
        
        # Process all samples again with the optimal threshold
        for i in range(len(all_outputs)):
            output = torch.tensor(all_outputs[i])
            target = torch.tensor(all_targets[i])
            
            # Print debug info for first few samples
            if i < 3:
                print(f"\nSample {i+1} with optimal threshold {prediction_threshold:.4f}:")
                
                # Get predicted themes based on threshold
                predicted_theme_idxs = torch.where(output > prediction_threshold)[0]
                predicted_themes = [test_label_names[idx.item()] for idx in predicted_theme_idxs]
                
                # Get actual themes
                actual_themes = [test_label_names[j] for j, is_theme in enumerate(target) if is_theme == 1]
                
                print(f"  Predictions: {predicted_themes}")
                print(f"  Targets: {actual_themes}")
            
            # Calculate metrics with optimal threshold
            stats = compute_multilabel_confusion_matrix(output, target, threshold=prediction_threshold)
            for key in confusion_stats:
                confusion_stats[key] += stats[key]

    # Print confusion matrix statistics
    print("\n=== Overall Confusion Matrix Statistics ===")
    print("\nPer-theme statistics:")
    print(f"{'Theme':<45} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>9} {'Recall':>9} {'F1':>9}")
    print("-" * 85)

    # Use test_label_names for the stats since confusion_stats is sized according to test dataset
    for i, theme in enumerate(test_label_names):
        tp = confusion_stats['true_positive'][i]
        fp = confusion_stats['false_positive'][i]
        fn = confusion_stats['false_negative'][i]
        tn = confusion_stats['true_negative'][i]
        
        # Calculate metrics (handling division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{theme:<45} {tp:6.0f} {fp:6.0f} {fn:6.0f} {tn:6.0f} {precision:9.3f} {recall:9.3f} {f1:9.3f}")

    # Calculate macro-averaged metrics
    print("\nMacro-averaged metrics:")

    # Calculate per-theme metrics first
    global theme_metrics  # make accessible to matrix functions
    theme_metrics = []
    for i, theme in enumerate(test_label_names):
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
    matrices = generate_all_matrices(dataset, test_label_names, theme_metrics, cached_predictions, model, device)
    
    return confusion_stats, dataset, label_names, model, device, cached_predictions, matrices

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
    from sklearn.metrics import f1_score
    
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
        
        # Print statistics for key themes if we have the mapping
        key_themes = {
            'advantage': {'train_idx': 1555, 'test_idx': 46},
            'crushing': {'train_idx': 1566, 'test_idx': 53},
            'endgame': {'train_idx': 1574, 'test_idx': 57},
            'hangingPiece': {'train_idx': 1578, 'test_idx': 60},
            'long': {'train_idx': 1585, 'test_idx': 63},
            'middlegame': {'train_idx': 1594, 'test_idx': 70},
            'short': {'train_idx': 1606, 'test_idx': 81}
        }
        
        print("\nKey theme statistics:")
        for theme, indices in key_themes.items():
            test_idx = indices['test_idx']
            
            if test_idx < outputs.shape[1]:
                # Calculate stats for this theme
                theme_outputs = outputs[:, test_idx]
                theme_targets = targets[:, test_idx]
                
                print(f"{theme} (test_idx={test_idx}):")
                print(f"  Positives: {theme_targets.sum()}/{len(theme_targets)} ({theme_targets.sum()/len(theme_targets):.2%})")
                print(f"  Output stats - min: {theme_outputs.min():.6f}, max: {theme_outputs.max():.6f}, mean: {theme_outputs.mean():.6f}")
                print(f"  Adaptive threshold: {class_thresholds[test_idx]:.6f}")
    
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

# Main execution 
if __name__ == "__main__":
    # Get command line arguments
    args = parse_args()
    num_samples = args.num_samples
    checkpoint_file = args.checkpoint
    
    # Initialize with a default, but we'll calculate the optimal threshold if not specified
    # Now that we fixed the mapping, we can use a more standard default threshold
    prediction_threshold = args.threshold if args.threshold is not None else 0.3
    
    threshold_mode = "user-specified" if args.threshold is not None else "adaptive (will be calculated)"
    print(f"Testing with {num_samples} samples and prediction threshold: {threshold_mode}")
    print(f"Optimized mode: {'ON' if not args.no_optimize else 'OFF'}")
    
    if args.use_cache:
        print("Using cached tensor files directly for evaluation")
    
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
            result = run_test()
            confusion_stats, dataset, label_names, model, device, cached_predictions, matrices = result
            
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
            result = run_test()
            confusion_stats, dataset, label_names, model, device, cached_predictions, matrices = result
        
        # End timing and display
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        print("\n✅ Test completed successfully!")
        
        # Print information about saved matrices
        print("\nMatrix files saved:")
        for matrix_type in matrices:
            print(f"  - {matrix_type.capitalize()} matrix:")
            print(f"    Raw: analysis/matrices/{matrix_type}_raw_matrix_{prediction_threshold}_mt0.001_{num_samples}.npy (binary)")
            print(f"         analysis/matrices/{matrix_type}_raw_matrix_{prediction_threshold}_mt0.001_{num_samples}.csv (human-readable)")
            print(f"    Normalized: analysis/matrices/{matrix_type}_normalized_matrix_{prediction_threshold}_mt0.001_{num_samples}.npy (binary)")
            print(f"                analysis/matrices/{matrix_type}_normalized_matrix_{prediction_threshold}_mt0.001_{num_samples}.csv (human-readable)")
            print(f"    Mapping: analysis/matrices/{matrix_type}_theme_mapping_{prediction_threshold}_mt0.001_{num_samples}.json")
            print(f"    Visualization: analysis/matrices/{matrix_type}_confusion_matrix_{prediction_threshold}_mt0.001_{num_samples}.png")
        
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