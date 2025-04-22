import torch
import numpy as np
import argparse
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
    return parser.parse_args()

# Get command line arguments
args = parse_args()
num_samples = args.num_samples
prediction_threshold = args.threshold  # Threshold for converting probabilities to binary predictions
checkpoint_file = args.checkpoint

print(f"Testing with {num_samples} samples and prediction threshold of {prediction_threshold}")

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

# Process samples with tqdm progress bar
progress_bar = tqdm(range(num_samples), desc="Processing samples", unit="sample")
for i in progress_bar:
    # Update progress bar description to show current sample number
    progress_bar.set_description(f"Processing sample {i+1}/{num_samples}")
    
    sample = dataset[i]
    input = sample['board'].unsqueeze(0).unsqueeze(0)
    target = sample['themes']

    with torch.no_grad():
        out = model(input)
        out = torch.sigmoid(out)  # Convert logits to probabilities

    # Calculate Jaccard index on the raw probabilities vs thresholded predictions
    raw_jaccard = jaccard_similarity(out.squeeze(), target, threshold=prediction_threshold)

    # Get predicted themes (where probability > threshold)
    predicted_probs, predicted_indices = torch.where(out > prediction_threshold, out, torch.zeros_like(out)).squeeze().sort(descending=True)
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
    stats = compute_multilabel_confusion_matrix(out.squeeze(), target, threshold=prediction_threshold)
    
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

    # Calculate Jaccard index using string lists (should match raw_jaccard)
    name_jaccard = jaccard_similarity(predicted_theme_names, actual_themes)

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

# After calculating all metrics, create the visual confusion matrices
print("\nGenerating visual confusion matrices...")

def create_cooccurrence_matrix(labels, label_type, active_themes=None):
    """Create a co-occurrence matrix for the given labels."""
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
    
    # Fill the co-occurrence matrix with progress bar
    matrix_progress = tqdm(range(num_samples), desc=f"Creating {label_type} matrix", unit="sample")
    for i in matrix_progress:
        matrix_progress.set_description(f"Matrix sample {i+1}/{num_samples} ({label_type})")
        
        sample = dataset[i]
        target = sample['themes']
        
        with torch.no_grad():
            out = model(sample['board'].unsqueeze(0).unsqueeze(0))
            out = torch.sigmoid(out)
        
        # Get actual and predicted labels based on the filter
        actual_labels = [label for i, label in enumerate(labels) if target[i] == 1 and label_filter(label)]
        pred_binary = (out.squeeze() > prediction_threshold).float()
        predicted_labels = [label for i, label in enumerate(labels) if pred_binary[i] == 1 and label_filter(label)]
        
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
    
    # Save the matrix info to a text file
    info_filename = f'{label_type}_matrix_info_{prediction_threshold}_{num_samples}.txt'
    with open(info_filename, 'w') as f:
        f.write(f"{label_type.capitalize()} co-occurrence matrix with threshold={prediction_threshold}, samples={num_samples}\n")
        f.write(f"Number of labels: {n_themes}\n\n")
        f.write("Labels included in the matrix:\n")
        for i, theme in enumerate(active_themes, 1):
            f.write(f"{i}. {theme}\n")
    
    # Save the plot with higher resolution and include threshold in filename
    output_filename = f'{label_type}_confusion_matrix_{prediction_threshold}_{num_samples}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=1.0)
    plt.close()
    
    print(f"{label_type.capitalize()} confusion matrix saved as '{output_filename}'")
    print(f"Label information saved to '{info_filename}'")
    return normalized_matrix

# 1. Generate matrix for all labels (both themes and openings)
print("\nGenerating combined themes+openings matrix...")
label_filter = lambda x: True  # No filtering, include all labels
create_cooccurrence_matrix(label_names, "combined", [m['theme'] for m in theme_metrics])

# 2. Generate matrix for themes only 
print("\nGenerating themes-only matrix...")
themes_only = dataset.get_themes_only()
label_filter = lambda x: dataset.is_theme(x)  # Only include themes
create_cooccurrence_matrix(label_names, "theme")

# 3. Generate matrix for openings only
print("\nGenerating openings-only matrix...")
openings_only = dataset.get_openings_only()
label_filter = lambda x: dataset.is_opening_tag(x)  # Only include opening tags
create_cooccurrence_matrix(label_names, "opening")