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
    parser = argparse.ArgumentParser(description='Test chess puzzle classifier')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to test (default: 1000)')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Prediction threshold for classification (default: 0.001)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_resume.pth",
                        help='Checkpoint file to use for testing')
    parser.add_argument('--optimized', action='store_true',
                        help='Use optimized version of the code')
    return parser.parse_args()

def create_confusion_matrix(dataset, test_labels, model, device, prediction_threshold, num_samples):
    """Create confusion matrix from model predictions"""
    
    # Filter only to active themes for readability
    active_themes = ["advantage", "crushing", "endgame", "hangingPiece", "long", "middlegame", "short"]
    print(f"Creating confusion matrix for {len(active_themes)} themes...")
    
    # Create theme to index mapping
    theme_to_idx = {theme: idx for idx, theme in enumerate(active_themes)}
    
    # Create mapping from test indices to theme names
    test_idx_to_theme = {
        46: "advantage",
        53: "crushing",
        57: "endgame",
        60: "hangingPiece",
        63: "long",
        70: "middlegame",
        81: "short"
    }
    
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
            out_raw = torch.sigmoid(out).squeeze().cpu()
        
        # Map the raw output using our known theme mapping
        mapped_output = torch.zeros_like(target)
        
        # For each test index in our mapping, get the corresponding training index value
        print("\nMapping output values from training to test indices:")
        for test_idx, train_idx in label_mapping.items():
            if train_idx < out_raw.shape[0] and test_idx < mapped_output.shape[0]:
                mapped_output[test_idx] = out_raw[train_idx]
                theme_name = test_idx_to_theme.get(test_idx, f"unknown-{test_idx}")
                print(f"Mapped {theme_name} (test_idx={test_idx}, train_idx={train_idx}): {out_raw[train_idx]:.6f} > {prediction_threshold} = {out_raw[train_idx] > prediction_threshold}")
        
        # Debug all mapped values
        print("\nMapped output for key themes:")
        for idx, name in test_idx_to_theme.items():
            if idx < mapped_output.shape[0]:
                print(f"{name} (idx {idx}): {mapped_output[idx]:.6f} > {prediction_threshold} = {mapped_output[idx] > prediction_threshold}")
            else:
                print(f"{name} (idx {idx}): out of bounds")
        
        # Print raw output stats
        print(f"\nOutput statistics for sample {i}:")
        print(f"Raw output shape: {out_raw.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Raw output min: {out_raw.min().item()}, max: {out_raw.max().item()}, mean: {out_raw.mean().item()}")
        
        # Print top 5 highest values
        values, indices = out_raw.topk(5)
        print("\nTop 5 highest probabilities:")
        for idx, (val, ind) in enumerate(zip(values, indices)):
            label_idx = ind.item() % len(test_labels)  # Ensure we don't go out of bounds
            print(f"  {idx+1}. Index {ind.item()} -> {test_labels[label_idx] if label_idx < len(test_labels) else 'Unknown'}: {val.item():.6f}")
        
        # Get actual and predicted labels
        actual_labels = [test_labels[j] for j in range(len(test_labels)) if target[j] == 1]
        pred_binary = (mapped_output > prediction_threshold).float()
        predicted_labels = [test_labels[j] for j in range(len(test_labels)) if j < len(pred_binary) and pred_binary[j] == 1]
        
        # Debug
        print(f"\nSample {i}:")
        print(f"Actual labels: {actual_labels}")
        print(f"Predicted labels: {predicted_labels}")
        
        # Update confusion matrix
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

def run_test():
    """Main test function that evaluates the model"""
    
    # Create dataset
    print("Loading test dataset...")
    dataset = ChessPuzzleDataset(os.path.join("dataset", "lichess_db_puzzle_test.csv"))
    
    # Adjust num_samples if larger than dataset
    global num_samples
    num_samples = min(num_samples, len(dataset))
    print(f"Testing with {num_samples} samples")
    
    # Load label mapping between full and test datasets
    # This is hardcoded for the key themes we care about for simplicity
    global label_mapping
    label_mapping = {
        # Maps from training index to test index
        1555: 46,  # advantage
        1566: 53,  # crushing 
        1574: 57,  # endgame
        1578: 60,  # hangingPiece
        1585: 63,  # long
        1594: 70,  # middlegame
        1606: 81,  # short
    }
    # Create reverse mapping for easy lookup
    global reverse_label_mapping
    reverse_label_mapping = {test_idx: train_idx for train_idx, test_idx in label_mapping.items()}
    
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

def evaluate_performance(dataset, test_labels, model, device, threshold, num_samples):
    """Evaluate model performance metrics"""
    
    # Initialize confusion matrix accumulators
    confusion_stats = {
        'true_positive': np.zeros(len(test_labels)),
        'false_positive': np.zeros(len(test_labels)),
        'false_negative': np.zeros(len(test_labels)),
        'true_negative': np.zeros(len(test_labels))
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
        
        # Map the raw output using our known theme mapping
        mapped_output = torch.zeros_like(target)
        
        # For each test index in our mapping, get the corresponding training index value
        for test_idx, train_idx in label_mapping.items():
            if train_idx < out_raw.shape[0] and test_idx < mapped_output.shape[0]:
                mapped_output[test_idx] = out_raw[train_idx]
        
        # Calculate metrics for this sample using the mapped output
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

if __name__ == "__main__":
    # Get command line arguments
    args = parse_args()
    num_samples = args.num_samples
    prediction_threshold = args.threshold
    checkpoint_file = args.checkpoint
    
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