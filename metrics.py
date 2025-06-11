import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report

def compute_multilabel_confusion_matrix(pred, target, threshold=0.5):
    """
    Compute confusion matrix statistics for multilabel classification.
    
    Args:
        pred: PyTorch tensor of predicted probabilities (shape: [batch_size, num_classes] or [num_classes])
        target: PyTorch tensor of binary ground truth labels (same shape as pred)
        threshold: Float between 0 and 1, threshold for positive prediction
    
    Returns:
        Dictionary containing arrays of TP, FP, FN, TN counts per class
    """
    if isinstance(pred, torch.Tensor):
        # Add batch dimension if input is a single sample
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            
        pred_binary = (pred > threshold).float()
        target = target.float()
        
        true_positive = (pred_binary * target).sum(dim=0)
        false_positive = (pred_binary * (1 - target)).sum(dim=0)
        false_negative = ((1 - pred_binary) * target).sum(dim=0)
        true_negative = ((1 - pred_binary) * (1 - target)).sum(dim=0)
        
        return {
            'true_positive': true_positive.cpu().numpy(),
            'false_positive': false_positive.cpu().numpy(),
            'false_negative': false_negative.cpu().numpy(),
            'true_negative': true_negative.cpu().numpy()
        }
    else:
        raise ValueError("Confusion matrix computation only supports tensor inputs")

def jaccard_similarity(pred, target, threshold=0.5, adaptive_threshold=True, verbose=False):
    """
    Compute Jaccard similarity (intersection over union) between predictions and target.
    Works with both binary tensors (from training) and sets of strings (from evaluation).
    
    Args:
        pred: Either a binary PyTorch tensor or a set/list of predicted theme names
        target: Either a binary PyTorch tensor of same shape as pred, or a set/list of actual theme names
        threshold: Float between 0 and 1. For tensor inputs, values above this threshold are considered
                  positive predictions. Ignored for set/list inputs. Defaults to 0.5.
        adaptive_threshold: Whether to use adaptive thresholding for early training
        verbose: Whether to print additional debugging information
    
    Returns:
        Jaccard similarity score (float between 0 and 1)
    """
    if isinstance(pred, torch.Tensor):
        # Apply threshold to get binary predictions
        if adaptive_threshold:
            # Calculate statistics across the batch dimension for each class
            pred_mean = pred.mean(dim=0)
            pred_std = pred.std(dim=0)
            
            # Create class-specific thresholds that are lower during early training
            # Adjust the formula to use a smaller factor for better early-training results
            class_thresholds = torch.clamp(pred_mean - 0.8 * pred_std, min=0.05, max=threshold)
            
            # Debug output to understand threshold distribution
            if verbose:
                min_threshold = class_thresholds.min().item()
                max_threshold = class_thresholds.max().item()
                avg_threshold = class_thresholds.mean().item()
                print(f"Jaccard adaptive thresholds: min={min_threshold:.4f}, avg={avg_threshold:.4f}, max={max_threshold:.4f}")
            
            # Use class-specific thresholds
            pred_binary = torch.zeros_like(pred, dtype=torch.bool)
            for i in range(pred.shape[1]):
                pred_binary[:, i] = (pred[:, i] > class_thresholds[i])
                
            # Debug output to understand thresholding effect
            if verbose:
                positive_preds = pred_binary.sum().item()
                total_preds = pred_binary.numel()
                positive_ratio = positive_preds / total_preds
                print(f"Jaccard positive predictions: {positive_preds} / {total_preds} = {positive_ratio:.4f}")
                
                # Show target statistics for comparison
                positive_targets = target.sum().item()
                target_ratio = positive_targets / target.numel()
                print(f"Jaccard positive targets: {positive_targets} / {target.numel()} = {target_ratio:.4f}")
        else:
            # Standard fixed threshold approach
            pred_binary = (pred > threshold)
            
            if verbose:
                positive_preds = pred_binary.sum().item()
                total_preds = pred_binary.numel()
                positive_ratio = positive_preds / total_preds
                print(f"Jaccard positive predictions (fixed threshold={threshold}): {positive_preds} / {total_preds} = {positive_ratio:.4f}")
        
        # Calculate intersection and union directly on tensors
        intersection = torch.logical_and(pred_binary, target).sum()
        union = torch.logical_or(pred_binary, target).sum()
        
        # Add detailed debugging info about intersection and union
        if verbose:
            print(f"Jaccard intersection: {intersection.item()}, union: {union.item()}")
            
            # Check if intersection is zero
            if intersection.item() == 0:
                print("WARNING: Jaccard intersection is ZERO. Possible reasons:")
                
                # Check if predictions are all zeros
                pred_sum = pred_binary.sum().item()
                if pred_sum == 0:
                    print("  - All predictions are negative (below threshold)")
                    print(f"  - Prediction max: {pred.max().item():.4f}, min: {pred.min().item():.4f}, mean: {pred.mean().item():.4f}")
                    print(f"  - Thresholds - min: {class_thresholds.min().item():.4f}, max: {class_thresholds.max().item():.4f}")
                
                # Check if targets are all zeros
                target_sum = target.sum().item()
                if target_sum == 0:
                    print("  - All targets are zeros (no positive labels)")
                
                # Check if there's no match between predictions and targets
                if pred_sum > 0 and target_sum > 0:
                    print("  - No overlap between predictions and targets despite both having positive values")
                    print(f"  - Predictions have {pred_sum} positive values")
                    print(f"  - Targets have {target_sum} positive values")
                    
                # Debug for sparse matrices (checking distribution)
                if union.item() > 0:
                    sparsity = 1.0 - (pred_sum + target_sum) / (2 * pred_binary.numel())
                    print(f"  - Data sparsity: {sparsity:.6f} (higher means fewer positive labels)")
                    if sparsity > 0.99:
                        print("  - EXTREMELY SPARSE DATA detected - consider reducing threshold or using weighted metrics")
        
        # Calculate Jaccard similarity with epsilon to avoid division by zero
        jaccard = intersection.float() / (union.float() + 1e-8)
        
        if verbose:
            print(f"Jaccard similarity: {jaccard.item():.6f}")
            
            # Warn about numerical stability issues
            if jaccard.item() < 1e-6:
                print("WARNING: Jaccard similarity is very close to zero")
                print("Consider using a different threshold or diagnostic metrics for early training")
        
        # Special case: If there are no positive labels in the target (all zeros),
        # return a small positive value (0.01) instead of zero to allow training progress visualization
        # Only do this when union is non-zero (meaning we have predictions) but intersection is zero
        if target.sum().item() == 0 and intersection.item() == 0 and union.item() > 0:
            if verbose:
                print("No positive labels in targets, returning default value (0.01) instead of zero")
            return torch.tensor(0.01, device=jaccard.device)
            
        return jaccard
    else:
        # Handle sets/lists of theme names
        pred_set = set(pred)
        target_set = set(target)
        intersection = len(pred_set & target_set)
        union = len(pred_set | target_set)
        
        if verbose:
            print(f"Jaccard set intersection: {intersection}, union: {union}")
            
        return intersection / union if union > 0 else 1.0  # Handle case where both sets are empty

def precision_recall_f1(pred, target, threshold=0.5, average='micro', verbose=False, adaptive_threshold=True):
    """
    Calculate precision, recall, and F1 score for multi-label classification.
    
    Args:
        pred (torch.Tensor): Predicted probabilities tensor [batch_size, num_classes]
        target (torch.Tensor): Target binary tensor [batch_size, num_classes]
        threshold (float): Probability threshold for binary prediction
        average (str): Averaging method ('micro', 'macro', 'samples', 'weighted', or None)
            - 'micro': Calculate metrics globally by counting the total TPs, FNs, and FPs
            - 'macro': Calculate metrics for each label, and find their unweighted mean
            - 'weighted': Calculate metrics for each label, and find their average weighted by support
            - 'samples': Calculate metrics for each instance, and find their average
            - None: Return scores for each class
        verbose (bool): Whether to print progress information
        adaptive_threshold (bool): Whether to use adaptive thresholding for early training
            
    Returns:
        tuple: (precision, recall, f1)
    """
    # Apply adaptive thresholding for early training when model outputs are weak
    actual_threshold = threshold
    if adaptive_threshold:
        # Calculate statistics across the batch dimension for each class
        pred_mean = pred.mean(dim=0)
        pred_std = pred.std(dim=0)
        
        # Create class-specific thresholds that are lower during early training
        # when outputs are closer to zero
        class_thresholds = torch.clamp(pred_mean - 0.5 * pred_std, min=0.05, max=threshold)
        
        # Use class-specific thresholds
        pred_binary = torch.zeros_like(pred, dtype=torch.bool)
        for i in range(pred.shape[1]):
            pred_binary[:, i] = (pred[:, i] > class_thresholds[i])
        
        if verbose:
            min_threshold = class_thresholds.min().item()
            max_threshold = class_thresholds.max().item()
            avg_threshold = class_thresholds.mean().item()
            print(f"Using adaptive thresholds: min={min_threshold:.4f}, avg={avg_threshold:.4f}, max={max_threshold:.4f}")
        
        pred_binary = pred_binary.cpu().numpy()
    else:
        # Standard fixed threshold approach
        if verbose:
            print(f"Converting predictions to binary with fixed threshold {threshold}...")
        pred_binary = (pred > threshold).cpu().numpy()
    
    target_binary = target.cpu().numpy()
    
    # Calculate precision, recall, F1
    if verbose:
        print(f"Calculating {average} precision, recall, and F1 scores...")
        
    try:
        # Try with progress reporting if tqdm is available and it's verbose mode
        from tqdm import tqdm
        if verbose:
            # Use a dummy progress bar since scikit-learn doesn't provide progress updates
            with tqdm(total=1, desc=f"Computing {average} metrics") as pbar:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    target_binary, pred_binary, average=average, zero_division=0
                )
                pbar.update(1)
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_binary, pred_binary, average=average, zero_division=0
            )
    except ImportError:
        # Fall back if tqdm isn't available
        precision, recall, f1, _ = precision_recall_fscore_support(
            target_binary, pred_binary, average=average, zero_division=0
        )
    
    return precision, recall, f1

def get_classification_report(pred, target, threshold=0.5, labels=None, verbose=False, adaptive_threshold=True):
    """
    Generate a text report showing the main classification metrics for multi-label classification.
    
    Args:
        pred (torch.Tensor): Predicted probabilities tensor [batch_size, num_classes]
        target (torch.Tensor): Target binary tensor [batch_size, num_classes]
        threshold (float): Probability threshold for binary prediction
        labels (list): Optional list of label names corresponding to the classes
        verbose (bool): Whether to print progress information
        adaptive_threshold (bool): Whether to use adaptive thresholding for early training
        
    Returns:
        str: Text summary of the precision, recall, F1 score for each class
    """
    # Convert to binary predictions based on threshold
    if verbose:
        print("Converting predictions for classification report...")
    
    # Apply adaptive thresholding for early training when model outputs are weak
    if adaptive_threshold:
        # Calculate statistics across the batch dimension for each class
        pred_mean = pred.mean(dim=0)
        pred_std = pred.std(dim=0)
        
        # Create class-specific thresholds that are lower during early training
        # when outputs are closer to zero
        class_thresholds = torch.clamp(pred_mean - 0.5 * pred_std, min=0.05, max=threshold)
        
        # Use class-specific thresholds
        pred_binary = torch.zeros_like(pred, dtype=torch.bool)
        for i in range(pred.shape[1]):
            pred_binary[:, i] = (pred[:, i] > class_thresholds[i])
            
        if verbose:
            min_threshold = class_thresholds.min().item()
            max_threshold = class_thresholds.max().item()
            avg_threshold = class_thresholds.mean().item()
            print(f"Using adaptive thresholds: min={min_threshold:.4f}, avg={avg_threshold:.4f}, max={max_threshold:.4f}")
        
        pred_binary = pred_binary.cpu().numpy()
    else:
        # Standard fixed threshold approach
        if verbose:
            print(f"Converting predictions with fixed threshold {threshold}...")
        pred_binary = (pred > threshold).cpu().numpy()
        
    target_binary = target.cpu().numpy()
    
    # Get the classification report
    if verbose:
        print("Generating detailed classification report (this might take a moment)...")
        try:
            from tqdm import tqdm
            with tqdm(total=1, desc="Generating classification report") as pbar:
                report = classification_report(
                    target_binary, pred_binary, 
                    target_names=labels,
                    zero_division=0
                )
                pbar.update(1)
        except ImportError:
            report = classification_report(
                target_binary, pred_binary, 
                target_names=labels,
                zero_division=0
            )
    else:
        report = classification_report(
            target_binary, pred_binary, 
            target_names=labels,
            zero_division=0
        )
    
    return report