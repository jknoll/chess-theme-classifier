import torch
import numpy as np
import argparse
from dataset import ChessPuzzleDataset
from model import Model
from metrics import jaccard_similarity, compute_multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Test chess puzzle classifier and generate confusion matrices')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of samples to test (default: 200)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Prediction threshold for binary classification (default: 0.3)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_modified_model.pth",
                        help='Checkpoint file to use for testing')
    return parser.parse_args()

# Get command line arguments
args = parse_args()
num_samples = args.num_samples
prediction_threshold = args.threshold  # Threshold for converting probabilities to binary predictions
checkpoint_file = args.checkpoint

print(f"Testing with {num_samples} samples and prediction threshold of {prediction_threshold}")
print(f"Using checkpoint: {checkpoint_file}")

# Create dataset first to get number of labels
dataset = ChessPuzzleDataset("lichess_db_puzzle_test.csv")
num_labels = len(dataset.get_theme_names())

# Create model instance with correct number of output labels
model = Model(num_labels=num_labels)

# Load the checkpoint and extract just the model state dict
checkpoint = torch.load(checkpoint_file)
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
for i in range(min(num_samples, len(dataset))):
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

print("\nTop 20 themes by F1 score:")
print(f"{'Theme':<30} {'Precision':>9} {'Recall':>9} {'F1':>9} {'TP':>6} {'FP':>6} {'FN':>6}")
print("-" * 80)
for metric in theme_metrics[:20]:
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