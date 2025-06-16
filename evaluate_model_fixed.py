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
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to test (default: 20)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold for classification (default: None, uses adaptive thresholding)')
    parser.add_argument('--checkpoint', type=str, default="checkpoint_resume.pth",
                        help='Checkpoint file to use for testing')
    parser.add_argument('--optimized', action='store_true',
                        help='Use optimized version of the code')
    return parser.parse_args()

def create_confusion_matrix(dataset, test_labels, model, device, prediction_threshold, num_samples):
    """Create confusion matrix from model predictions"""
    
    # Define the key themes we care about for the matrix
    active_themes = ["advantage", "crushing", "endgame", "hangingPiece", "long", "middlegame", "short"]
    print(f"Creating confusion matrix for {len(active_themes)} themes...")
    
    # Hard-coded mapping between test dataset indices and training dataset indices
    # This mapping was determined by examining the data and finding where each theme is stored
    theme_mapping = {
        # theme_name: [training_idx, test_idx]
        'advantage': [1555, 46],
        'crushing': [1566, 53],
        'endgame': [1574, 57],
        'hangingPiece': [1578, 60],
        'long': [1585, 63],
        'middlegame': [1594, 70],
        'short': [1606, 81]
    }
    
    # Create mappings for theme names and indices
    theme_to_idx = {theme: idx for idx, theme in enumerate(active_themes)}
    theme_name_to_test_idx = {name: indices[1] for name, indices in theme_mapping.items()}
    test_idx_to_theme_name = {indices[1]: name for name, indices in theme_mapping.items()}
    train_idx_to_theme_name = {indices[0]: name for name, indices in theme_mapping.items()}
    
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
        
        # Map model outputs to test dataset indices
        # We only care about the key themes we're tracking
        mapped_output = torch.zeros_like(target)
        
        # For each theme mapping, copy the value from the training index to the test index
        for theme_name, (train_idx, test_idx) in theme_mapping.items():
            if train_idx < out_raw.shape[0] and test_idx < mapped_output.shape[0]:
                mapped_output[test_idx] = out_raw[train_idx]
        
        # Get actual themes from the target tensor
        actual_theme_names = []
        for test_idx, is_present in enumerate(target):
            if is_present == 1 and test_idx in test_idx_to_theme_name:
                actual_theme_names.append(test_idx_to_theme_name[test_idx])
        
        # Get predicted themes using the threshold
        pred_binary = (mapped_output > prediction_threshold).float()
        predicted_theme_names = []
        for test_idx, is_predicted in enumerate(pred_binary):
            if is_predicted == 1 and test_idx in test_idx_to_theme_name:
                predicted_theme_names.append(test_idx_to_theme_name[test_idx])
        
        # For debugging, print the actual and predicted labels for the first few samples
        if i < 3:
            print(f"\nSample {i+1}:")
            print(f"FEN: {sample['fen'] if 'fen' in sample else 'N/A'}")
            print(f"Actual themes: {', '.join(actual_theme_names) if actual_theme_names else 'None'}")
            print(f"Predicted themes: {', '.join(predicted_theme_names) if predicted_theme_names else 'None'}")
            
            # Print mapped values for key themes
            print("\nTheme predictions:")
            for theme_name, (train_idx, test_idx) in theme_mapping.items():
                if test_idx < mapped_output.shape[0]:
                    value = mapped_output[test_idx].item()
                    print(f"  {theme_name}: {value:.4f} > {prediction_threshold} = {value > prediction_threshold}")
        
        # Update confusion matrix
        for actual in actual_theme_names:
            if actual in theme_to_idx:
                for predicted in predicted_theme_names:
                    if predicted in theme_to_idx:
                        matrix[theme_to_idx[actual], theme_to_idx[predicted]] += 1
    
    # Create visualization of the confusion matrix
    plt.figure(figsize=(12, 10))
    
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
                annot=True,
                fmt='.2f',
                square=True)
    
    plt.title(f'Theme Co-occurrence Matrix (threshold={prediction_threshold})\n(Row: Actual, Column: Predicted)')
    plt.xlabel('Predicted Theme')
    plt.ylabel('Actual Theme')
    
    # Create directory for output
    os.makedirs('analysis/matrices', exist_ok=True)
    
    # Save the visualization
    output_file = f'analysis/matrices/confusion_matrix_fixed_{prediction_threshold}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_file}")
    return matrix

def calculate_adaptive_threshold(outputs, targets, theme_indices):
    """
    Calculate optimal threshold based on F1 score
    
    Args:
        outputs: Tensor of predicted probabilities
        targets: Tensor of ground truth labels
        theme_indices: Dictionary mapping theme names to their indices
    
    Returns:
        Best threshold value
    """
    from sklearn.metrics import f1_score
    
    # Extract only key theme indices for threshold calculation
    theme_idxs = list(theme_indices.values())
    filtered_outputs = outputs[:, theme_idxs]
    filtered_targets = targets[:, theme_idxs]
    
    # Flatten for binary classification metrics and convert to numpy
    flat_outputs = filtered_outputs.reshape(-1).cpu().numpy()
    flat_targets = filtered_targets.reshape(-1).cpu().numpy()
    
    # Print statistics
    print("\nAdaptive Threshold Calculation:")
    print(f"Using {len(outputs)} samples with {len(theme_indices)} key themes")
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

def run_test():
    """Main test function that evaluates the model"""
    
    # Create dataset
    print("Loading test dataset...")
    dataset = ChessPuzzleDataset(os.path.join("dataset", "lichess_db_puzzle_test.csv"))
    
    # Adjust num_samples if larger than dataset
    global num_samples, prediction_threshold
    num_samples = min(num_samples, len(dataset))
    print(f"Testing with {num_samples} samples")
    
    # Define key theme mappings
    global theme_mapping
    theme_mapping = {
        # theme_name: [training_idx, test_idx]
        'advantage': [1555, 46],
        'crushing': [1566, 53],
        'endgame': [1574, 57],
        'hangingPiece': [1578, 60],
        'long': [1585, 63],
        'middlegame': [1594, 70],
        'short': [1606, 81]
    }
    
    # Create mappings for easy lookup
    global test_idx_to_theme_name
    test_idx_to_theme_name = {test_idx: name for name, (_, test_idx) in theme_mapping.items()}
    
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
    
    # If using adaptive thresholding, collect outputs and targets first
    if prediction_threshold is None:
        print("\nCalculating adaptive threshold...")
        
        # Collect all predictions and targets
        all_mapped_outputs = []
        all_targets = []
        
        # Create test index to index mapping for key themes
        theme_indices = {name: idx[1] for name, idx in theme_mapping.items()}
        
        # Process all samples to collect data for adaptive thresholding
        for i in tqdm(range(num_samples), desc="Collecting predictions"):
            sample = dataset[i]
            target = sample['themes']
            
            # Get model prediction
            with torch.no_grad():
                input_tensor = sample['board'].unsqueeze(0).unsqueeze(0).to(device)
                out = model(input_tensor)
                out_raw = torch.sigmoid(out).squeeze().cpu()
            
            # Map model outputs to test dataset indices
            mapped_output = torch.zeros_like(target)
            
            # For each theme mapping, copy the value from the training index to the test index
            for name, (train_idx, test_idx) in theme_mapping.items():
                if train_idx < out_raw.shape[0] and test_idx < mapped_output.shape[0]:
                    mapped_output[test_idx] = out_raw[train_idx]
            
            all_mapped_outputs.append(mapped_output.unsqueeze(0))
            all_targets.append(target.unsqueeze(0))
        
        # Combine all outputs and targets
        combined_outputs = torch.cat(all_mapped_outputs, dim=0)
        combined_targets = torch.cat(all_targets, dim=0)
        
        # Calculate adaptive threshold
        prediction_threshold = calculate_adaptive_threshold(
            combined_outputs, combined_targets, theme_indices)
    
    print(f"\nUsing threshold: {prediction_threshold}")
    
    # Create confusion matrix
    matrix = create_confusion_matrix(dataset, test_label_names, model, device, prediction_threshold, num_samples)
    
    # Calculate metrics
    confusion_stats = evaluate_performance(dataset, test_label_names, model, device, prediction_threshold, num_samples)
    
    print("\nEvaluation complete!")
    return matrix, confusion_stats

def evaluate_performance(dataset, test_labels, model, device, threshold, num_samples):
    """Evaluate model performance metrics using mapped outputs"""
    
    # Hard-coded mapping between test dataset indices and training dataset indices
    theme_mapping = {
        # theme_name: [training_idx, test_idx]
        'advantage': [1555, 46],
        'crushing': [1566, 53],
        'endgame': [1574, 57],
        'hangingPiece': [1578, 60],
        'long': [1585, 63],
        'middlegame': [1594, 70],
        'short': [1606, 81]
    }
    
    # Create reverse mappings
    train_idx_to_test_idx = {train_idx: test_idx for _, (train_idx, test_idx) in theme_mapping.items()}
    test_idx_to_theme = {test_idx: name for name, (_, test_idx) in theme_mapping.items()}
    
    # Initialize confusion matrix accumulators
    confusion_stats = {
        'true_positive': np.zeros(len(test_labels)),
        'false_positive': np.zeros(len(test_labels)),
        'false_negative': np.zeros(len(test_labels)),
        'true_negative': np.zeros(len(test_labels))
    }
    
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
            out_raw = torch.sigmoid(out).squeeze().cpu()
        
        # Map model outputs to test dataset indices
        mapped_output = torch.zeros_like(target)
        
        # For each mapped theme, copy the value from the training index to the test index
        for train_idx, test_idx in train_idx_to_test_idx.items():
            if train_idx < out_raw.shape[0] and test_idx < mapped_output.shape[0]:
                mapped_output[test_idx] = out_raw[train_idx]
        
        # Calculate metrics for this sample using the mapped output
        stats = compute_multilabel_confusion_matrix(mapped_output, target, threshold=threshold)
        
        # Update totals
        for key in confusion_stats:
            confusion_stats[key] += stats[key]
        
        # Collect actual and predicted themes for reporting
        pred_binary = (mapped_output > threshold).float()
        for j in range(len(target)):
            if j in test_idx_to_theme:
                theme_name = test_idx_to_theme[j]
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
    
    # Print key theme statistics
    print("\n=== Theme-specific Statistics ===")
    print(f"{'Theme':<15} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 75)
    
    for name, (_, test_idx) in theme_mapping.items():
        if test_idx < len(confusion_stats['true_positive']):
            tp = confusion_stats['true_positive'][test_idx]
            fp = confusion_stats['false_positive'][test_idx]
            fn = confusion_stats['false_negative'][test_idx]
            tn = confusion_stats['true_negative'][test_idx]
            
            # Calculate metrics
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            print(f"{name:<15} {tp:5.0f} {fp:5.0f} {fn:5.0f} {tn:5.0f} {p:10.4f} {r:10.4f} {f:10.4f}")
    
    # Return collected statistics
    return confusion_stats

if __name__ == "__main__":
    # Get command line arguments
    args = parse_args()
    num_samples = args.num_samples
    prediction_threshold = args.threshold
    checkpoint_file = args.checkpoint
    
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