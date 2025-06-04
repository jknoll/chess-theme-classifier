#!/usr/bin/env python3
"""
Profile FEN to Tensor Conversion Performance

This script profiles the time it takes to convert the entire dataset from FEN notation to tensor format.
It measures both the total time and the per-puzzle conversion time.
"""

import time
import cProfile
import pstats
import io
from dataset import ChessPuzzleDataset
from chess import Board
import torch
import argparse
import os


def profile_dataset_loading(csv_file):
    """Load the dataset and profile the time taken"""
    print(f"Profiling dataset loading from {csv_file}...")
    
    # Time the dataset initialization (which processes all themes and opening tags)
    start_time = time.time()
    dataset = ChessPuzzleDataset(csv_file)
    init_time = time.time() - start_time
    
    print(f"Dataset initialization: {init_time:.2f} seconds")
    print(f"Total puzzles: {len(dataset):,}")
    
    return dataset, init_time


def profile_fen_to_tensor_conversion(dataset, num_samples=None, iterations=10):
    """Profile the time it takes to convert from FEN to tensor format"""
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    print(f"Profiling FEN to tensor conversion for {num_samples:,} puzzles over {iterations} iterations...")
    
    # Time the conversion for each puzzle
    total_time = 0
    board_tensors = []
    
    # First run without timing to ensure caches are warmed up
    print("Warming up caches...")
    for i in range(min(1000, num_samples)):
        _ = dataset[i]
    
    # Now profile the actual conversion
    print(f"Starting profiling...")
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    # Run multiple iterations for more accurate timing
    for _ in range(iterations):
        board_tensors = []
        for i in range(num_samples):
            item = dataset[i]
            board_tensors.append(item['board'])
    end_time = time.time()
    
    pr.disable()
    
    # Calculate stats - adjust for iterations
    total_time = (end_time - start_time) / iterations
    avg_time_per_puzzle = total_time / num_samples
    puzzles_per_second = num_samples / total_time
    
    print(f"\nResults:")
    print(f"Total time for {num_samples:,} puzzles: {total_time:.2f} seconds")
    print(f"Average time per puzzle: {avg_time_per_puzzle*1000:.2f} ms")
    print(f"Conversion rate: {puzzles_per_second:.2f} puzzles/second")
    
    # Output cProfile stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    print("\nDetailed profiling statistics:")
    print(s.getvalue())
    
    return total_time, avg_time_per_puzzle, puzzles_per_second, board_tensors


def profile_batch_conversion(csv_file, num_samples=None, iterations=10):
    """
    Profile a batch conversion approach where we first load all FENs, 
    then convert them to tensors in batch
    """
    from dataset import ChessPuzzleDataset
    import pandas as pd
    from chess import Board
    import torch
    
    # Map piece symbols to indices (same as in ChessPuzzleDataset._board_to_tensor)
    piece_to_idx = {
        'K': 0,  # white king
        'Q': 1,  # white queen
        'R': 2,  # white rook
        'B': 3,  # white bishop
        'N': 4,  # white knight
        'P': 5,  # white pawn
        'p': 7,  # black pawn
        'n': 8,  # black knight
        'b': 9,  # black bishop
        'r': 10, # black rook
        'q': 11, # black queen
        'k': 12, # black king
    }
    
    print(f"\nProfiling alternate batch conversion approach over {iterations} iterations...")
    
    # Load CSV directly
    start_time = time.time()
    df = pd.read_csv(csv_file)
    if num_samples is not None:
        df = df.head(num_samples)
    csv_load_time = time.time() - start_time
    print(f"CSV loading time: {csv_load_time:.2f} seconds")
    
    fen_list = df['FEN'].tolist()
    puzzles_count = len(fen_list)
    
    # Function to convert a single FEN to tensor
    def fen_to_tensor(fen):
        board = Board(fen)
        tensor = torch.full((8, 8), 6)  # Initialize with empty squares (6)
        
        for i in range(64):
            rank, file = i // 8, i % 8
            piece = board.piece_at(i)
            if piece is not None:
                tensor[rank, file] = piece_to_idx[piece.symbol()]
                
        return tensor
    
    # Warm up the cache
    print("Warming up caches...")
    for i in range(min(1000, puzzles_count)):
        if i < len(fen_list):
            _ = fen_to_tensor(fen_list[i])
    
    # Profile the batch conversion
    print("Starting profiling...")
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    # Run multiple iterations for more accurate timing
    for _ in range(iterations):
        board_tensors = []
        for fen in fen_list:
            tensor = fen_to_tensor(fen)
            board_tensors.append(tensor)
    end_time = time.time()
    
    pr.disable()
    
    # Calculate stats - adjust for iterations
    batch_total_time = (end_time - start_time) / iterations
    batch_avg_time = batch_total_time / puzzles_count
    batch_puzzles_per_second = puzzles_count / batch_total_time
    
    print(f"\nBatch Conversion Results:")
    print(f"Total time for {puzzles_count:,} puzzles: {batch_total_time:.2f} seconds")
    print(f"Average time per puzzle: {batch_avg_time*1000:.2f} ms")
    print(f"Conversion rate: {batch_puzzles_per_second:.2f} puzzles/second")
    
    # Output cProfile stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    print("\nDetailed batch profiling statistics:")
    print(s.getvalue())
    
    return batch_total_time, batch_avg_time, batch_puzzles_per_second


def main():
    parser = argparse.ArgumentParser(description='Profile FEN to tensor conversion')
    parser.add_argument('--csv', type=str, default=os.path.join('dataset', 'lichess_db_puzzle_test.csv'),
                       help='Path to the CSV file (default: dataset/lichess_db_puzzle_test.csv)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of puzzles to process (default: all)')
    parser.add_argument('--output', type=str, default='profile_fen_to_tensor.txt',
                       help='Output file for profiling results (default: profile_fen_to_tensor.txt)')
    parser.add_argument('--batch', action='store_true',
                       help='Also test a batch conversion approach')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations to run for more accurate profiling (default: 10)')
    
    args = parser.parse_args()
    
    # Ensure the file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found.")
        return
    
    # Redirect output to file if specified
    if args.output:
        original_stdout = sys.stdout
        f = open(args.output, 'w')
        sys.stdout = f
    
    # Print system info
    print("=" * 50)
    print("FEN to Tensor Conversion Profiling")
    print("=" * 50)
    print(f"Dataset: {args.csv}")
    print(f"Samples: {'All' if args.samples is None else args.samples}")
    print("-" * 50)
    
    # Run the dataset loading profile
    dataset, init_time = profile_dataset_loading(args.csv)
    
    # Run the FEN to tensor conversion profile
    total_time, avg_time, puzzles_per_sec, _ = profile_fen_to_tensor_conversion(dataset, args.samples, args.iterations)
    
    # Optionally profile the batch conversion approach
    if args.batch:
        batch_total_time, batch_avg_time, batch_puzzles_per_sec = profile_batch_conversion(args.csv, args.samples, args.iterations)
    
    # Print summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Dataset: {args.csv}")
    print(f"Total puzzles processed: {args.samples if args.samples else len(dataset):,}")
    print(f"Dataset initialization time: {init_time:.2f} seconds")
    print(f"Item-by-item conversion time: {total_time:.2f} seconds")
    print(f"Average per puzzle: {avg_time*1000:.2f} ms")
    print(f"Conversion rate: {puzzles_per_sec:.2f} puzzles/second")
    
    if args.batch:
        print("\nBatch conversion approach:")
        print(f"Total conversion time: {batch_total_time:.2f} seconds")
        print(f"Average per puzzle: {batch_avg_time*1000:.2f} ms")
        print(f"Conversion rate: {batch_puzzles_per_sec:.2f} puzzles/second")
        
        # Compare approaches
        speedup = puzzles_per_sec / batch_puzzles_per_sec if batch_puzzles_per_sec > 0 else float('inf')
        print(f"\nComparison: {'Item-by-item' if puzzles_per_sec > batch_puzzles_per_sec else 'Batch'} approach is " +
              f"{max(speedup, 1/speedup):.2f}x faster")
    
    print("=" * 50)
    
    # Reset stdout
    if args.output:
        sys.stdout = original_stdout
        f.close()
        print(f"Profiling results written to {args.output}")


if __name__ == "__main__":
    import sys
    main()