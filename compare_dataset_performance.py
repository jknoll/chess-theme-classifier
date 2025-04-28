#!/usr/bin/env python3
"""
Performance comparison between original and optimized dataset implementations.

This script tests the original ChessPuzzleDataset implementation against the
optimized implementation with tensor caching.
"""

import time
import torch
import os
import pandas as pd
from chess import Board
from dataset import ChessPuzzleDataset
from optimized_dataset import OptimizedChessPuzzleDataset
from torch.utils.data import DataLoader

def original_implementation(csv_file, num_samples):
    """Test the performance of the original implementation"""
    print("\nOriginal Implementation:")
    print("-" * 50)
    
    class OriginalChessPuzzleDataset(torch.utils.data.Dataset):
        def __init__(self, csv_file):
            self.puzzle_data = pd.read_csv(csv_file)
            
            # Map piece symbols to indices
            self.piece_to_idx = {
                'K': 0,   # white king
                'Q': 1,   # white queen
                'R': 2,   # white rook
                'B': 3,   # white bishop
                'N': 4,   # white knight
                'P': 5,   # white pawn
                'p': 7,   # black pawn
                'n': 8,   # black knight
                'b': 9,   # black bishop
                'r': 10,  # black rook
                'q': 11,  # black queen
                'k': 12,  # black king
            }
        
        def __len__(self):
            return len(self.puzzle_data)
        
        def __getitem__(self, idx):
            fen = self.puzzle_data.iloc[idx]['FEN']
            
            # Convert FEN to board
            board = Board(fen)
            
            # Create tensor
            tensor = torch.full((8, 8), 6)  # Initialize with empty squares (6)
            
            for i in range(64):
                rank, file = i // 8, i % 8
                piece = board.piece_at(i)
                if piece is not None:
                    tensor[rank, file] = self.piece_to_idx[piece.symbol()]
            
            return tensor.to(dtype=torch.float32)
    
    # Start timing
    start_time = time.time()
    dataset = OriginalChessPuzzleDataset(csv_file)
    init_time = time.time() - start_time
    print(f"Dataset initialized in {init_time:.4f} seconds")
    
    # Access items
    start_time = time.time()
    tensors = []
    for i in range(num_samples):
        tensor = dataset[i]
        tensors.append(tensor)
    access_time = time.time() - start_time
    print(f"Accessed {num_samples} items in {access_time:.4f} seconds")
    print(f"Average time per item: {access_time/num_samples*1000:.4f} ms")
    print(f"Items per second: {num_samples/access_time:.2f}")
    
    return init_time, access_time

def optimized_implementation(csv_file, num_samples):
    """Test the performance of the optimized implementation"""
    print("\nOptimized Implementation:")
    print("-" * 50)
    
    # Remove existing cache file to force regeneration
    tensors_cache_file = os.path.join('.', f"{os.path.basename(csv_file)}.tensors.pt")
    if os.path.exists(tensors_cache_file):
        os.remove(tensors_cache_file)
        print(f"Removed existing tensor cache file: {tensors_cache_file}")
    
    # Start timing
    start_time = time.time()
    dataset = OptimizedChessPuzzleDataset(csv_file)
    init_time = time.time() - start_time
    print(f"Dataset initialized with cache creation in {init_time:.4f} seconds")
    
    # Now test loading with existing cache
    start_time = time.time()
    dataset = OptimizedChessPuzzleDataset(csv_file)
    cached_init_time = time.time() - start_time
    print(f"Dataset initialized with existing cache in {cached_init_time:.4f} seconds")
    
    # Access items
    start_time = time.time()
    tensors = []
    for i in range(num_samples):
        tensor = dataset[i]['board']
        tensors.append(tensor)
    access_time = time.time() - start_time
    print(f"Accessed {num_samples} items in {access_time:.4f} seconds")
    print(f"Average time per item: {access_time/num_samples*1000:.4f} ms")
    print(f"Items per second: {num_samples/access_time:.2f}")
    
    return init_time, cached_init_time, access_time

def test_dataloader_performance(csv_file, batch_size=32):
    """Test performance with DataLoader for both implementations"""
    print("\nDataLoader Performance:")
    print("-" * 50)
    
    # Original dataset
    class OriginalDataLoader(torch.utils.data.Dataset):
        def __init__(self, csv_file):
            self.puzzle_data = pd.read_csv(csv_file)
            
            # Map piece symbols to indices
            self.piece_to_idx = {
                'K': 0,   # white king
                'Q': 1,   # white queen
                'R': 2,   # white rook
                'B': 3,   # white bishop
                'N': 4,   # white knight
                'P': 5,   # white pawn
                'p': 7,   # black pawn
                'n': 8,   # black knight
                'b': 9,   # black bishop
                'r': 10,  # black rook
                'q': 11,  # black queen
                'k': 12,  # black king
            }
        
        def __len__(self):
            return len(self.puzzle_data)
        
        def __getitem__(self, idx):
            fen = self.puzzle_data.iloc[idx]['FEN']
            
            # Convert FEN to board
            board = Board(fen)
            
            # Create tensor
            tensor = torch.full((8, 8), 6)  # Initialize with empty squares (6)
            
            for i in range(64):
                rank, file = i // 8, i % 8
                piece = board.piece_at(i)
                if piece is not None:
                    tensor[rank, file] = self.piece_to_idx[piece.symbol()]
            
            return tensor.to(dtype=torch.float32)
    
    # Create datasets
    original_dataset = OriginalDataLoader(csv_file)
    optimized_dataset = OptimizedChessPuzzleDataset(csv_file)
    num_samples = len(original_dataset)
    
    # Create DataLoaders
    original_loader = DataLoader(original_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    optimized_loader = DataLoader(optimized_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Test original DataLoader
    start_time = time.time()
    for batch in original_loader:
        pass
    original_time = time.time() - start_time
    print(f"Original DataLoader: {original_time:.4f} seconds for {num_samples} items")
    print(f"Items per second: {num_samples/original_time:.2f}")
    
    # Test optimized DataLoader
    start_time = time.time()
    for batch in optimized_loader:
        # Get the board tensor from each item
        boards = batch['board']
    optimized_time = time.time() - start_time
    print(f"Optimized DataLoader: {optimized_time:.4f} seconds for {num_samples} items")
    print(f"Items per second: {num_samples/optimized_time:.2f}")
    
    # Calculate speedup
    speedup = original_time / optimized_time
    print(f"\nDataLoader speedup: {speedup:.2f}x")
    
    return original_time, optimized_time, speedup

def main():
    """Main function to run the performance tests"""
    # Use a small dataset for quick testing
    csv_file = "lichess_db_puzzle_test.csv"
    
    # Load the dataset to get the number of samples
    data = pd.read_csv(csv_file)
    num_samples = len(data)
    print(f"Dataset size: {num_samples} puzzles")
    
    # Run performance tests
    original_init, original_access = original_implementation(csv_file, num_samples)
    optimized_init, optimized_cached_init, optimized_access = optimized_implementation(csv_file, num_samples)
    
    # Test DataLoader performance
    original_loader, optimized_loader, loader_speedup = test_dataloader_performance(csv_file)
    
    # Print summary
    print("\nPerformance Summary:")
    print("=" * 50)
    print(f"Dataset size: {num_samples} puzzles")
    print("\nInitialization time:")
    print(f"  Original implementation: {original_init:.4f} seconds")
    print(f"  Optimized implementation (first run): {optimized_init:.4f} seconds")
    print(f"  Optimized implementation (with cache): {optimized_cached_init:.4f} seconds")
    print(f"  Speedup with cache: {original_init/optimized_cached_init:.2f}x")
    
    print("\nAccess time:")
    print(f"  Original implementation: {original_access:.4f} seconds")
    print(f"  Optimized implementation: {optimized_access:.4f} seconds")
    print(f"  Speedup: {original_access/optimized_access:.2f}x")
    
    print("\nDataLoader performance:")
    print(f"  Original implementation: {original_loader:.4f} seconds")
    print(f"  Optimized implementation: {optimized_loader:.4f} seconds")
    print(f"  Speedup: {loader_speedup:.2f}x")
    
    # Estimate full dataset performance (assumes linear scaling)
    full_dataset_size = 4956459  # From previous profiling
    original_full_estimate = original_access / num_samples * full_dataset_size
    optimized_full_estimate = optimized_access / num_samples * full_dataset_size
    
    print("\nEstimated full dataset (4,956,459 puzzles) access time:")
    print(f"  Original implementation: {original_full_estimate:.2f} seconds = {original_full_estimate/60:.2f} minutes")
    print(f"  Optimized implementation: {optimized_full_estimate:.2f} seconds = {optimized_full_estimate/60:.2f} minutes")
    print(f"  Estimated time saved: {(original_full_estimate - optimized_full_estimate)/60:.2f} minutes")

if __name__ == "__main__":
    main()