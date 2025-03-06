import torch

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