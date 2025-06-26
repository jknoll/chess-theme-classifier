#!/usr/bin/env python3
"""
Script to count the number of boards in cached tensor files.
"""

import torch
import os
import sys
from pathlib import Path

def count_tensors_in_file(filepath):
    """Count the number of boards/tensors in a cached tensor file."""
    try:
        print(f"Loading: {filepath}")
        tensor_data = torch.load(filepath)
        
        if isinstance(tensor_data, dict):
            if 'tensors' in tensor_data:
                count = len(tensor_data['tensors'])
                print(f"  Type: dict with 'tensors' key (list)")
                print(f"  Number of boards: {count}")
                if len(tensor_data['tensors']) > 0:
                    print(f"  First tensor shape: {tensor_data['tensors'][0].shape}")
            elif 'boards' in tensor_data:
                count = tensor_data['boards'].shape[0]
                print(f"  Type: dict with 'boards' key (tensor)")
                print(f"  Number of boards: {count}")
                print(f"  Boards tensor shape: {tensor_data['boards'].shape}")
            else:
                print(f"  Keys: {list(tensor_data.keys())}")
                count = 0
                for key, value in tensor_data.items():
                    if hasattr(value, 'shape') and len(value.shape) > 0:
                        count = value.shape[0]
                        print(f"  Using {key} tensor with shape: {value.shape}")
                        break
                    elif isinstance(value, list):
                        count = len(value)
                        print(f"  Using {key} list with length: {count}")
                        break
        elif hasattr(tensor_data, 'shape'):
            count = tensor_data.shape[0]
            print(f"  Type: tensor")
            print(f"  Shape: {tensor_data.shape}")
            print(f"  Number of boards: {count}")
        else:
            print(f"  Type: {type(tensor_data)}")
            count = 0
            
        return count
        
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return 0

def main():
    # Check if a specific file was provided as argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if os.path.exists(filepath):
            count_tensors_in_file(filepath)
        else:
            print(f"File not found: {filepath}")
        return
    
    # Otherwise, search for tensor files in current directory
    current_dir = Path('.')
    tensor_files = list(current_dir.glob('*.tensors.pt*'))
    
    if not tensor_files:
        print("No tensor files found in current directory.")
        print("Usage: python count_cached_tensors.py [filepath]")
        return
    
    total_boards = 0
    for tensor_file in sorted(tensor_files):
        count = count_tensors_in_file(tensor_file)
        total_boards += count
        print()
    
    if len(tensor_files) > 1:
        print(f"Total boards across all files: {total_boards}")

if __name__ == "__main__":
    main()