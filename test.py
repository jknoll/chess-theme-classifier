import torch
import numpy as np
from dataset import ChessPuzzleDataset
from model import Model
from metrics import jaccard_similarity, compute_multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration parameters
num_samples = 2000
prediction_threshold = 0.3  # Threshold for converting probabilities to binary predictions

# Create dataset first to get number of labels
dataset = ChessPuzzleDataset("lichess_db_puzzle.csv")
num_labels = len(dataset.get_theme_names())

# Create model instance with correct number of output labels
model = Model(num_labels=num_labels)

# Load the checkpoint and extract just the model state dict
checkpoint = torch.load("checkpoint_resume.pth", weights_only=True)
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

# Process samples
for i in range(num_samples):
    print(f"Processing sample {i+1} of {num_samples}")
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

    # Debug print for first few samples
    if i < 3:
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

    # Compute confusion matrix statistics for this sample
    stats = compute_multilabel_confusion_matrix(out.squeeze(), target, threshold=prediction_threshold)
    
    # Debug print for first few samples
    if i < 3:
        print("Sample stats:")
        print(f"TP: {stats['true_positive'].sum()}")
        print(f"FP: {stats['false_positive'].sum()}")
        print(f"FN: {stats['false_negative'].sum()}")
        print(f"TN: {stats['true_negative'].sum()}")
    
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
    
    n_themes = len(active_themes)
    if n_themes == 0:
        print(f"No active {label_type}s found with non-zero metrics")
        return None
        
    print(f"Creating co-occurrence matrix for {n_themes} {label_type}s...")
    
    cooccurrence = np.zeros((n_themes, n_themes))
    theme_to_idx = {theme: idx for idx, theme in enumerate(active_themes)}
    
    # Fill the co-occurrence matrix
    for i in range(num_samples):
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
    
    # Create heatmap with improved legibility
    plt.figure(figsize=(20, 16))  # Increased figure size
    sns.heatmap(normalized_matrix, 
                xticklabels=active_themes,
                yticklabels=active_themes,
                cmap='YlOrRd',
                vmin=0,
                vmax=1,
                annot=True,
                fmt='.1f',  # Reduced decimal places
                square=True,
                annot_kws={'size': 8},  # Smaller font for numbers
                cbar_kws={'shrink': .8})  # Smaller colorbar
    
    plt.title(f'{label_type.capitalize()} Co-occurrence Matrix (threshold={prediction_threshold})\n(Row: Actual, Column: Predicted, Values: P(Predicted|Actual))', 
             pad=20, size=14)  # Added padding to title
    plt.xlabel(f'Predicted {label_type.capitalize()}', size=12, labelpad=10)
    plt.ylabel(f'Actual {label_type.capitalize()}', size=12, labelpad=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', size=10)  # Increased font size
    plt.yticks(rotation=0, size=10)  # Increased font size
    
    # Adjust layout to prevent label cutoff and add more spacing
    plt.tight_layout()
    
    # Save the plot with higher resolution and include threshold in filename
    output_filename = f'{label_type}_confusion_matrix_{prediction_threshold}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    print(f"{label_type.capitalize()} confusion matrix saved as '{output_filename}'")
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