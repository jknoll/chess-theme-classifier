import torch
import numpy as np
from dataset import ChessPuzzleDataset
from model import Model
from metrics import jaccard_similarity, compute_multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create model instance
model = Model()

# Load the checkpoint and extract just the model state dict
checkpoint = torch.load("checkpoint_resume.pth", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

# Set model to evaluation mode
model.eval()

dataset = ChessPuzzleDataset("lichess_db_puzzle.csv")

# Get theme names
theme_names = dataset.get_theme_names()

# Initialize confusion matrix accumulators
confusion_stats = {
    'true_positive': np.zeros(len(theme_names)),
    'false_positive': np.zeros(len(theme_names)),
    'false_negative': np.zeros(len(theme_names)),
    'true_negative': np.zeros(len(theme_names))
}

# Process samples
num_samples = 100  # Increased from 10
for i in range(num_samples):
    sample = dataset[i]
    input = sample['board'].unsqueeze(0).unsqueeze(0)
    target = sample['themes']

    with torch.no_grad():
        out = model(input)
        out = torch.sigmoid(out)  # Convert logits to probabilities

    # Calculate Jaccard index on the raw probabilities vs thresholded predictions
    raw_jaccard = jaccard_similarity(out.squeeze(), target, threshold=0.5)

    # Debug print for first few samples
    if i < 3:
        print(f"\nDebug - Sample {i+1}:")
        print(f"Target sum: {target.sum().item()}")
        print(f"Predictions > 0.5: {(out.squeeze() > 0.5).sum().item()}")

    # Compute confusion matrix statistics for this sample
    stats = compute_multilabel_confusion_matrix(out.squeeze(), target)
    
    # Debug print for first few samples
    if i < 3:
        print("Sample stats:")
        print(f"TP: {stats['true_positive'].sum()}")
        print(f"FP: {stats['false_positive'].sum()}")
        print(f"FN: {stats['false_negative'].sum()}")
        print(f"TN: {stats['true_negative'].sum()}")
    
    for key in confusion_stats:
        confusion_stats[key] += stats[key]

    # Get predicted themes (where probability > 0.5)
    predicted_probs, predicted_indices = torch.where(out > 0.5, out, torch.zeros_like(out)).squeeze().sort(descending=True)
    predicted_themes = [(theme_names[idx], f"{predicted_probs[i]:.3f}") for i, idx in enumerate(predicted_indices) if predicted_probs[i] > 0.5]
    
    # Get the theme names only from predicted themes (without probabilities)
    predicted_theme_names = [theme for theme, _ in predicted_themes]

    # Get actual themes
    actual_themes = [theme_names[i] for i, is_theme in enumerate(target) if is_theme == 1]

    # Calculate Jaccard index using string lists (should match raw_jaccard)
    name_jaccard = jaccard_similarity(predicted_theme_names, actual_themes)

    print(f"\n=== Sample {i+1} ===")
    print(f"\nJaccard Index (using thresholded probabilities): {raw_jaccard:.3f}")
    print(f"Jaccard Index (using theme name lists): {name_jaccard:.3f}")

    print("\nPredicted themes (probability):")
    for theme, prob in predicted_themes:
        print(f"{theme}: {prob}")

    print("\nActual themes:")
    print(", ".join(actual_themes))

    print("\nPosition FEN:")
    print(sample['fen'])
    print("="*50)

# Print confusion matrix statistics
print("\n=== Overall Confusion Matrix Statistics ===")
print("\nPer-theme statistics:")
print(f"{'Theme':<30} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>9} {'Recall':>9} {'F1':>9}")
print("-" * 85)

for i, theme in enumerate(theme_names):
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
for i, theme in enumerate(theme_names):
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

# After calculating all metrics, create the visual confusion matrix
print("\nGenerating visual confusion matrix...")

# Create co-occurrence matrix for actual vs predicted themes
active_themes = [m['theme'] for m in theme_metrics]
n_themes = len(active_themes)
cooccurrence = np.zeros((n_themes, n_themes))
theme_to_idx = {theme: idx for idx, theme in enumerate(active_themes)}

# Fill the co-occurrence matrix
for i in range(num_samples):
    sample = dataset[i]
    target = sample['themes']
    
    with torch.no_grad():
        out = model(sample['board'].unsqueeze(0).unsqueeze(0))
        out = torch.sigmoid(out)
    
    # Get actual and predicted themes
    actual_themes = [theme_names[i] for i, is_theme in enumerate(target) if is_theme == 1]
    pred_binary = (out.squeeze() > 0.5).float()
    predicted_themes = [theme_names[i] for i, is_pred in enumerate(pred_binary) if is_pred == 1]
    
    # Update co-occurrence matrix
    for actual in actual_themes:
        if actual in theme_to_idx:
            for predicted in predicted_themes:
                if predicted in theme_to_idx:
                    cooccurrence[theme_to_idx[actual], theme_to_idx[predicted]] += 1

# Create normalized confusion matrix (divide by number of actual occurrences)
actual_counts = cooccurrence.sum(axis=1, keepdims=True)
normalized_matrix = np.where(actual_counts > 0, cooccurrence / actual_counts, 0)

# Create heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(normalized_matrix, 
            xticklabels=active_themes,
            yticklabels=active_themes,
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            annot=True,
            fmt='.2f',
            square=True)

plt.title('Theme Co-occurrence Matrix\n(Row: Actual, Column: Predicted, Values: P(Predicted|Actual))')
plt.xlabel('Predicted Theme')
plt.ylabel('Actual Theme')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('theme_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visual confusion matrix saved as 'theme_confusion_matrix.png'")