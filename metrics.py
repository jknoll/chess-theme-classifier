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

def jaccard_similarity(pred, target, threshold=0.5):
    """
    Compute Jaccard similarity (intersection over union) between predictions and target.
    Works with both binary tensors (from training) and sets of strings (from evaluation).
    
    Args:
        pred: Either a binary PyTorch tensor or a set/list of predicted theme names
        target: Either a binary PyTorch tensor of same shape as pred, or a set/list of actual theme names
        threshold: Float between 0 and 1. For tensor inputs, values above this threshold are considered
                  positive predictions. Ignored for set/list inputs. Defaults to 0.5.
    
    Returns:
        Jaccard similarity score (float between 0 and 1)
    """
    if isinstance(pred, torch.Tensor):
        # Apply threshold to get binary predictions
        pred_binary = torch.where(pred > threshold, torch.ones_like(pred), torch.zeros_like(pred))
        intersection = torch.logical_and(pred_binary, target).sum()
        union = torch.logical_or(pred_binary, target).sum()
        return intersection.float() / (union.float() + 1e-8)  # add small epsilon to avoid division by zero
    else:
        # Handle sets/lists of theme names
        pred_set = set(pred)
        target_set = set(target)
        intersection = len(pred_set & target_set)
        union = len(pred_set | target_set)
        return intersection / union if union > 0 else 1.0  # Handle case where both sets are empty

def precision_recall_f1(pred, target, threshold=0.5, average='micro'):
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
            
    Returns:
        tuple: (precision, recall, f1)
    """
    # Convert to binary predictions based on threshold
    pred_binary = (pred > threshold).cpu().numpy()
    target_binary = target.cpu().numpy()
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_binary, pred_binary, average=average, zero_division=0
    )
    
    return precision, recall, f1

def get_classification_report(pred, target, threshold=0.5, labels=None):
    """
    Generate a text report showing the main classification metrics for multi-label classification.
    
    Args:
        pred (torch.Tensor): Predicted probabilities tensor [batch_size, num_classes]
        target (torch.Tensor): Target binary tensor [batch_size, num_classes]
        threshold (float): Probability threshold for binary prediction
        labels (list): Optional list of label names corresponding to the classes
        
    Returns:
        str: Text summary of the precision, recall, F1 score for each class
    """
    # Convert to binary predictions based on threshold
    pred_binary = (pred > threshold).cpu().numpy()
    target_binary = target.cpu().numpy()
    
    # Get the classification report
    report = classification_report(
        target_binary, pred_binary, 
        target_names=labels,
        zero_division=0
    )
    
    return report