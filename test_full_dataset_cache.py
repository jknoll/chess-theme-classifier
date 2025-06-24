#!/usr/bin/env python3
"""
Test script to create and verify the full dataset tensor cache for Milestone 1.
This script tests the improved dataset.py with fixes for MAX_FENS limit and
parallel processing error handling.
"""

import os
import sys
import time
from dataset import ChessPuzzleDataset

def test_full_dataset_cache():
    """Test creating a tensor cache for the full dataset."""
    
    # Use the main CSV file
    csv_file = "lichess_db_puzzle.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Please ensure the lichess_db_puzzle.csv file is in the current directory.")
        return False
    
    print("🧪 Testing Full Dataset Tensor Cache Creation")
    print("=" * 60)
    
    # Test parameters
    test_params = {
        'csv_file': csv_file,
        'cache_size': 1000,  # Small cache for memory efficiency
        'augment_with_reflections': False,  # Start with basic tensors only
        'class_conditional_augmentation': False,  # Disable for Milestone 1
        'low_memory': True,  # Enable low memory mode for safety
        'use_cache': False  # Force regeneration
    }
    
    print(f"📋 Test Parameters:")
    for key, value in test_params.items():
        print(f"   {key}: {value}")
    print()
    
    try:
        print("🚀 Starting dataset creation...")
        start_time = time.time()
        
        # Create dataset instance - this will trigger tensor cache creation
        dataset = ChessPuzzleDataset(**test_params)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"\n📊 Dataset Creation Results:")
        print(f"   Total time: {elapsed:.2f} seconds")
        print(f"   Dataset length: {len(dataset):,}")
        print(f"   Processing rate: {len(dataset) / elapsed:.2f} items/second")
        
        # Verify the tensor cache file exists
        expected_cache_file = f"{csv_file}.tensors.pt"
        if os.path.exists(expected_cache_file):
            file_size = os.path.getsize(expected_cache_file) / (1024 * 1024)  # MB
            print(f"   Cache file: {expected_cache_file} ({file_size:.1f} MB)")
        else:
            print(f"   ⚠️ Cache file not found: {expected_cache_file}")
        
        # Test accessing a few samples
        print(f"\n🔍 Testing sample access:")
        for i in [0, len(dataset)//2, len(dataset)-1]:
            try:
                sample = dataset[i]
                print(f"   Sample {i}: ✅ (board shape: {sample['board'].shape})")
            except Exception as e:
                print(f"   Sample {i}: ❌ Error: {e}")
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_dataset_cache()
    sys.exit(0 if success else 1)