#!/usr/bin/env python3
"""
Manually combine existing chunk files into the final cache.
This tests the streaming approach without running the full dataset creation.
"""

import torch
import os
import glob
from tqdm import tqdm
import gc

def combine_existing_chunks():
    """Combine existing chunk files into final cache"""
    temp_dir = '/tmp/chess_cache_t6xbqc8o'
    output_file = '/home/j/Documents/git/chess-theme-classifier/processed_lichess_puzzle_files/lichess_db_puzzle.csv.tensors.pt_conditional_full'
    
    # Get all chunk files
    chunk_files = sorted(glob.glob(os.path.join(temp_dir, 'chunk_*.pt')))
    print(f"Found {len(chunk_files)} chunk files")
    
    if not chunk_files:
        print("No chunk files found")
        return
    
    # Initialize combined data
    combined_data = {
        'tensors': [],
        'labels': [],
        'format_version': 2
    }
    
    total_entries = 0
    
    # Combine chunks with memory monitoring
    for i, chunk_file in enumerate(tqdm(chunk_files, desc="Combining chunks")):
        try:
            chunk_data = torch.load(chunk_file, weights_only=False)
            
            combined_data['tensors'].extend(chunk_data['tensors'])
            combined_data['labels'].extend(chunk_data['labels'])
            
            total_entries += len(chunk_data['tensors'])
            
            # Clean up
            del chunk_data
            gc.collect()
            
            # Memory monitoring every 50 chunks
            if i % 50 == 0:
                try:
                    import psutil
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                    print(f"  Memory usage: {memory_usage:.1f} GB, processed {total_entries:,} entries")
                except ImportError:
                    print(f"  Processed {total_entries:,} entries")
                    
                # If memory gets too high, break early for testing
                try:
                    import psutil
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                    if memory_usage > 25:  # Stop if over 25GB
                        print(f"Memory limit reached ({memory_usage:.1f} GB), stopping for safety")
                        break
                except ImportError:
                    pass
            
        except Exception as e:
            print(f"Error loading {chunk_file}: {e}")
            continue
    
    print(f"Total entries combined: {total_entries:,}")
    
    # Convert to tensors
    print("Converting to tensors...")
    try:
        combined_data['tensors'] = torch.stack(combined_data['tensors'])
        combined_data['labels'] = torch.stack(combined_data['labels'])
    except Exception as e:
        print(f"Error converting to tensors: {e}")
        print("This might be due to memory constraints or data inconsistencies")
        return
    
    print(f"Final tensor shapes:")
    print(f"  tensors: {combined_data['tensors'].shape}")
    print(f"  labels: {combined_data['labels'].shape}")
    
    # Save final cache
    print(f"Saving to {output_file}...")
    torch.save(combined_data, output_file)
    
    print("✅ Successfully created final cache file")
    
    # Verify file exists and has content
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024**3)  # GB
        print(f"File size: {file_size:.2f} GB")
        
        # Test loading the file
        print("Testing cache load...")
        test_data = torch.load(output_file, weights_only=False)
        print(f"✅ Cache loads successfully with {len(test_data['tensors']):,} entries")

if __name__ == '__main__':
    combine_existing_chunks()