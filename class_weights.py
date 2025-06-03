import torch
import numpy as np

def compute_label_weights(dataset):
    """
    Compute class weights for multi-label classification based on inverse frequency.
    
    Args:
        dataset: ChessPuzzleDataset instance with label information
        
    Returns:
        torch.Tensor: Tensor of weights for each label, shape [num_labels]
    """
    # Get the number of labels
    num_labels = len(dataset.all_labels)
    
    # Initialize counters for each label
    label_counts = torch.zeros(num_labels)
    
    # Count occurrences of each label
    total_samples = len(dataset.puzzle_data)
    for i in range(total_samples):
        # Get themes and opening tags
        themes = dataset.puzzle_data.iloc[i]['Themes'].split()
        
        # Add opening tags if they exist
        opening_tags = []
        if 'OpeningTags' in dataset.puzzle_data.columns and \
           dataset.puzzle_data.iloc[i]['OpeningTags'] and \
           not dataset.puzzle_data.iloc[i]['OpeningTags'].isna():
            opening_tags = dataset.puzzle_data.iloc[i]['OpeningTags'].split()
        
        # Update counts for each label
        for label in themes + opening_tags:
            if label in dataset.label_to_idx:
                idx = dataset.label_to_idx[label]
                label_counts[idx] += 1
    
    # Compute inverse frequency (handling zeros safely)
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    inverse_freq = total_samples / (label_counts + epsilon)
    
    # Normalize weights to have mean=1
    normalized_weights = inverse_freq / inverse_freq.mean()
    
    # Optional: Softmax version for smoother weights
    # This softens extreme weights while maintaining the relative importance
    def softmax_normalization(weights, temperature=1.0):
        # Apply exponential scaling with temperature
        exp_weights = torch.exp(torch.log(weights) / temperature)
        # Normalize to mean=1
        return exp_weights * (weights.shape[0] / exp_weights.sum())
    
    # Apply softmax normalization with temperature=2.0 to prevent extreme values
    normalized_weights = softmax_normalization(normalized_weights, temperature=2.0)
    
    # Log statistics about weights
    print(f"Label weight statistics:")
    print(f"  Min weight: {normalized_weights.min().item():.4f}")
    print(f"  Max weight: {normalized_weights.max().item():.4f}")
    print(f"  Mean weight: {normalized_weights.mean().item():.4f}")
    print(f"  Median weight: {torch.median(normalized_weights).item():.4f}")
    
    # Print the top 5 highest and lowest weighted labels
    sorted_indices = torch.argsort(normalized_weights, descending=True)
    print("\nHighest weighted (rare) labels:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i].item()
        label = dataset.all_labels[idx]
        count = label_counts[idx].item()
        weight = normalized_weights[idx].item()
        print(f"  {label}: count={count:.0f}, weight={weight:.4f}")
    
    print("\nLowest weighted (common) labels:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[-(i+1)].item()
        label = dataset.all_labels[idx]
        count = label_counts[idx].item()
        weight = normalized_weights[idx].item()
        print(f"  {label}: count={count:.0f}, weight={weight:.4f}")
    
    return normalized_weights