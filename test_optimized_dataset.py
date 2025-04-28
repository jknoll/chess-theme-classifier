#!/usr/bin/env python3
"""
Test script for the optimized dataset implementation.

This script loads a small dataset using both the original and optimized implementations,
and compares their performance.
"""

import time
import torch
import os
from dataset import ChessPuzzleDataset

def test_dataset_performance():
    """
    Test the performance of the original vs optimized dataset implementations.
    """
    # Small dataset for quick testing
    csv_file = "lichess_db_puzzle_test.csv"
    
    # Make sure we're using the optimized implementation
    print("Testing dataset performance:")
    print("-" * 50)
    
    # Measure initialization time
    start_time = time.time()
    dataset = ChessPuzzleDataset(csv_file)
    init_time = time.time() - start_time
    print(f"Dataset initialized in {init_time:.2f} seconds")
    print(f"Dataset size: {len(dataset)} puzzles")
    
    # Check if we're using the optimized implementation
    is_optimized = hasattr(dataset, '_using_optimized') and dataset._using_optimized
    print(f"Using optimized implementation: {is_optimized}")
    
    # Test __getitem__ performance
    start_time = time.time()
    num_samples = min(1000, len(dataset))
    
    # First iteration: warm up cache
    for i in range(num_samples):
        _ = dataset[i]
    
    # Second iteration: measure performance
    start_time = time.time()
    for i in range(num_samples):
        item = dataset[i]
    getitem_time = time.time() - start_time
    
    print(f"Accessed {num_samples} items in {getitem_time:.2f} seconds")
    print(f"Average time per item: {getitem_time/num_samples*1000:.2f} ms")
    print(f"Items per second: {num_samples/getitem_time:.2f}")
    
    # Check tensor shape
    item = dataset[0]
    print(f"Board tensor shape: {item['board'].shape}")
    print(f"Themes tensor shape: {item['themes'].shape}")
    
    # Test batch loading with DataLoader
    from torch.utils.data import DataLoader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    start_time = time.time()
    num_batches = 5
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        boards = batch['board']
        themes = batch['themes']
        
        # Print batch information
        if i == 0:
            print(f"Batch shapes - boards: {boards.shape}, themes: {themes.shape}")
    
    batch_time = time.time() - start_time
    print(f"Loaded {num_batches} batches of size {batch_size} in {batch_time:.2f} seconds")
    print(f"Average time per batch: {batch_time/num_batches*1000:.2f} ms")
    print(f"Items per second (batch mode): {num_batches*batch_size/batch_time:.2f}")
    
    return dataset

def check_cache_files():
    """Check what cache files were created"""
    print("\nCache files:")
    print("-" * 50)
    
    # List all cache files in the current directory
    cache_files = [f for f in os.listdir('.') if f.endswith('.json') or f.endswith('.pt')]
    
    for file in sorted(cache_files):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        mtime = os.path.getmtime(file)
        print(f"{file}: {size_mb:.2f} MB, last modified: {time.ctime(mtime)}")

def main():
    # Run tests
    dataset = test_dataset_performance()
    
    # Check cache files
    check_cache_files()
    
    # Verify results match expectations
    print("\nVerification:")
    print("-" * 50)
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of themes: {len(dataset.get_theme_names())}")
    
    # Get a sample item
    item = dataset[0]
    print(f"Sample FEN: {item['fen']}")
    
    # Verify that reflections work
    reflections = dataset.create_reflected_boards(item)
    print(f"Number of valid reflections: {sum(1 for r in reflections.values() if r is not None)}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()