import torch
import numpy as np
import os
import time
import json

def get_cache_path(dataset):
    """
    Generate the cache file path for class weights.
    
    Args:
        dataset: ChessPuzzleDataset instance with label information
        
    Returns:
        str: Path to the cache file
    """
    # Check if we're using the processed directory path
    if 'processed_lichess_puzzle_files' in dataset.csv_file:
        # Use the same directory as the other cache files
        cache_dir = os.path.dirname(dataset.csv_file)
    else:
        # Use the dataset directory
        cache_dir = 'dataset'
        
    csv_basename = os.path.basename(dataset.csv_file)
    return os.path.join(cache_dir, f"{csv_basename}.class_weights.pt")

def is_cache_valid(dataset):
    """
    Check if a valid class weights cache exists.
    
    Args:
        dataset: ChessPuzzleDataset instance with label information
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    cache_path = get_cache_path(dataset)
    
    # Check if cache file exists
    if not os.path.exists(cache_path):
        return False
    
    # When CSV doesn't exist, we're in cache-only mode
    if hasattr(dataset, 'csv_exists') and not dataset.csv_exists:
        # In cache-only mode, we just check if the cache file exists
        # since we can't compare against the original CSV
        return True
    
    # Check if the CSV file or any of its dependent caches are newer than the cache
    csv_mtime = os.path.getmtime(dataset.csv_file)
    cache_mtime = os.path.getmtime(cache_path)
    
    # Also check tensor cache file timestamp
    tensor_cache_paths = [
        dataset.tensors_cache_file,
        f"{dataset.tensors_cache_file}_conditional",
        f"{dataset.tensors_cache_file}_reflected"
    ]
    
    # Find the most recent modification time among all possible tensor caches
    tensor_cache_mtime = 0
    for tensor_path in tensor_cache_paths:
        if os.path.exists(tensor_path):
            tensor_mtime = os.path.getmtime(tensor_path)
            tensor_cache_mtime = max(tensor_cache_mtime, tensor_mtime)
    
    # Check co-occurrence file if it exists
    cooc_mtime = 0
    if os.path.exists(dataset.label_cooccurrence_file):
        cooc_mtime = os.path.getmtime(dataset.label_cooccurrence_file)
    
    # Get the most recent modification time among all dependencies
    latest_dependency_mtime = max(csv_mtime, tensor_cache_mtime, cooc_mtime)
    
    # Cache is valid if it's newer than all dependencies
    return cache_mtime > latest_dependency_mtime

def save_to_cache(dataset, weights, label_counts):
    """
    Save class weights to cache.
    
    Args:
        dataset: ChessPuzzleDataset instance with label information
        weights: torch.Tensor of class weights
        label_counts: torch.Tensor of label counts
    """
    cache_path = get_cache_path(dataset)
    
    # Save weights, label counts, and metadata to cache
    cache_data = {
        'weights': weights,
        'label_counts': label_counts,
        'all_labels': dataset.all_labels,
        'csv_file': dataset.csv_file,
        'timestamp': time.time(),
        'num_samples': len(dataset.puzzle_data)
    }
    
    print(f"Saving class weights to cache: {cache_path}")
    torch.save(cache_data, cache_path)

def load_from_cache(dataset):
    """
    Load class weights from cache.
    
    Args:
        dataset: ChessPuzzleDataset instance with label information
        
    Returns:
        tuple: (weights, label_counts)
    """
    cache_path = get_cache_path(dataset)
    
    print(f"Loading class weights from cache: {cache_path}")
    cache_data = torch.load(cache_path)
    
    # Verify that the cache is compatible with the current dataset
    current_labels = set(dataset.all_labels)
    cached_labels = set(cache_data['all_labels'])
    
    if current_labels != cached_labels:
        print("⚠️ Warning: Cached labels don't match current dataset labels.")
        print(f"  Current dataset has {len(current_labels)} labels")
        print(f"  Cache has {len(cached_labels)} labels")
        print(f"  Labels added: {current_labels - cached_labels}")
        print(f"  Labels removed: {cached_labels - current_labels}")
        return None, None
    
    # Display cache information
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_data['timestamp']))
    print(f"Using class weights cached on {time_str}")
    print(f"  Original dataset: {cache_data['csv_file']}")
    print(f"  Number of samples: {cache_data['num_samples']}")
    
    return cache_data['weights'], cache_data['label_counts']

def compute_label_weights(dataset):
    """
    Compute class weights for multi-label classification based on inverse frequency.
    Uses on-disk caching for better performance on repeated runs.
    
    Args:
        dataset: ChessPuzzleDataset instance with label information
        
    Returns:
        torch.Tensor: Tensor of weights for each label, shape [num_labels]
    """
    # Check if valid cache exists
    if is_cache_valid(dataset):
        weights, label_counts = load_from_cache(dataset)
        if weights is not None:
            # Log statistics about weights from cache
            log_weight_statistics(dataset, weights, label_counts)
            return weights
    
    # If CSV doesn't exist, we can't compute class weights - must rely on cache or create default weights
    if hasattr(dataset, 'csv_exists') and not dataset.csv_exists:
        # Get the expected path for class weights
        cache_path = get_cache_path(dataset)
        
        # Check if class weights exist at the expected location
        if os.path.exists(cache_path):
            print(f"CSV file not found. Using class weights from: {cache_path}")
            try:
                # Load from the cache location
                cache_data = torch.load(cache_path)
                weights = cache_data['weights']
                label_counts = cache_data['label_counts']
                
                # Log statistics about weights
                log_weight_statistics(dataset, weights, label_counts)
                
                return weights
            except Exception as e:
                print(f"⚠️ Error loading class weights from {cache_path}: {e}")
                print("⚠️ Creating default uniform weights as fallback")
        else:
            print(f"⚠️ CSV file not found and class weights cache not available at {cache_path}.")
            print("⚠️ Creating default uniform weights as fallback")
        
        # Create default uniform weights as fallback
        num_labels = len(dataset.all_labels)
        print(f"Creating uniform weights for {num_labels} labels")
        
        # Create uniform weights and counts
        uniform_weights = torch.ones(num_labels)
        default_counts = torch.ones(num_labels) * 10  # Assume 10 samples per class
        
        # Try to save these default weights to cache for future use
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            # Save default weights to cache
            cache_data = {
                'weights': uniform_weights,
                'label_counts': default_counts,
                'all_labels': dataset.all_labels,
                'csv_file': dataset.csv_file,
                'timestamp': time.time(),
                'num_samples': num_labels * 10,  # Fake sample count
                'is_default': True  # Flag that these are default weights
            }
            
            print(f"Saving default class weights to cache: {cache_path}")
            torch.save(cache_data, cache_path)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save default weights to cache: {e}")
        
        print("Using uniform class weights (no label will be prioritized)")
        return uniform_weights
    
    print("Computing class weights (no valid cache found)...")
    
    # Get the number of labels
    num_labels = len(dataset.all_labels)
    
    # Initialize counters for each label
    label_counts = torch.zeros(num_labels)
    
    # Count occurrences of each label - use tqdm for progress
    total_samples = len(dataset.puzzle_data)
    
    # Use tqdm progress bar if available
    try:
        from tqdm import tqdm
        iterator = tqdm(range(total_samples), desc="Computing class weights")
    except ImportError:
        print(f"Processing {total_samples} samples to compute class weights...")
        iterator = range(total_samples)
    
    for i in iterator:
        # Get themes and opening tags
        themes = dataset.puzzle_data.iloc[i]['Themes'].split()
        
        # Add opening tags if they exist
        opening_tags = []
        if 'OpeningTags' in dataset.puzzle_data.columns:
            opening_tag_value = dataset.puzzle_data.iloc[i]['OpeningTags']
            # Check if the value is a non-empty string (not NaN, None, or empty)
            if isinstance(opening_tag_value, str) and opening_tag_value.strip():
                opening_tags = opening_tag_value.split()
        
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
    
    # Apply softmax normalization with temperature=2.0 to prevent extreme values
    normalized_weights = softmax_normalization(normalized_weights, temperature=2.0)
    
    # Log statistics about weights
    log_weight_statistics(dataset, normalized_weights, label_counts)
    
    # Save weights to cache
    save_to_cache(dataset, normalized_weights, label_counts)
    
    return normalized_weights

def softmax_normalization(weights, temperature=1.0):
    """
    Apply softmax normalization to weights to prevent extreme values.
    
    Args:
        weights: torch.Tensor of weights
        temperature: Temperature parameter for scaling (higher = smoother)
        
    Returns:
        torch.Tensor: Normalized weights
    """
    # Apply exponential scaling with temperature
    exp_weights = torch.exp(torch.log(weights) / temperature)
    # Normalize to mean=1
    return exp_weights * (weights.shape[0] / exp_weights.sum())

def log_weight_statistics(dataset, weights, label_counts):
    """
    Log statistics about class weights.
    
    Args:
        dataset: ChessPuzzleDataset instance with label information
        weights: torch.Tensor of class weights
        label_counts: torch.Tensor of label counts
    """
    print(f"Label weight statistics:")
    print(f"  Min weight: {weights.min().item():.4f}")
    print(f"  Max weight: {weights.max().item():.4f}")
    print(f"  Mean weight: {weights.mean().item():.4f}")
    print(f"  Median weight: {torch.median(weights).item():.4f}")
    
    # Print the top 5 highest and lowest weighted labels
    sorted_indices = torch.argsort(weights, descending=True)
    print("\nHighest weighted (rare) labels:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i].item()
        label = dataset.all_labels[idx]
        count = label_counts[idx].item()
        weight = weights[idx].item()
        print(f"  {label}: count={count:.0f}, weight={weight:.4f}")
    
    print("\nLowest weighted (common) labels:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[-(i+1)].item()
        label = dataset.all_labels[idx]
        count = label_counts[idx].item()
        weight = weights[idx].item()
        print(f"  {label}: count={count:.0f}, weight={weight:.4f}")