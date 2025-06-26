#!/usr/bin/env python3
"""
Script to create the full dataset tensor cache for all 4,956,460 puzzles.
This implements Milestone 1 from full-class-conditional-dataset-augmentation.md

Usage:
    python create_full_dataset_cache.py [--verbose]
    
Options:
    --verbose    Show detailed progress bars and timing information

This will create:
    - lichess_db_puzzle.csv.tensors.pt (full dataset tensor cache)
    - lichess_db_puzzle.csv.themes.json (theme labels cache)  
    - lichess_db_puzzle.csv.openings.json (opening labels cache)
"""

import os
import sys
import time
import argparse
from dataset import ChessPuzzleDataset

def create_full_dataset_cache(verbose=False):
    """Create the full dataset tensor cache for Milestone 1."""
    
    # Use the main CSV file
    csv_file = "processed_lichess_puzzle_files/lichess_db_puzzle.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Please ensure the lichess_db_puzzle.csv file is in the processed_lichess_puzzle_files directory.")
        return False
    
    print("🚀 Creating Full Dataset Tensor Cache (Milestone 1)")
    print("=" * 60)
    print("This will process all 4,956,460 puzzles and create tensor caches.")
    print("Estimated time: 1-3 hours depending on system performance.")
    print()
    
    # Production parameters for full dataset
    params = {
        'csv_file': csv_file,
        'cache_size': 1000,  # Keep memory cache small
        'augment_with_reflections': False,  # Basic tensors only for Milestone 1
        'class_conditional_augmentation': False,  # Milestone 2 feature
        'low_memory': True,  # Safe for large datasets
        'use_cache': False,  # Force regeneration
        'verbose_progress': verbose,  # Pass verbose flag to dataset
        'num_workers': 0  # Force single-threaded processing to avoid multiprocessing issues
    }
    
    print(f"📋 Processing Parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    print()
    
    try:
        print("🚀 Starting full dataset processing...")
        start_time = time.time()
        
        # Create dataset instance - this will create the tensor cache
        dataset = ChessPuzzleDataset(**params)
        
        end_time = time.time()
        elapsed = end_time - start_time
        hours = elapsed / 3600
        
        print(f"\n🎉 Full Dataset Cache Creation Complete!")
        print(f"📊 Final Results:")
        print(f"   Total time: {hours:.2f} hours ({elapsed:.0f} seconds)")
        print(f"   Dataset length: {len(dataset):,}")
        print(f"   Processing rate: {len(dataset) / elapsed:.2f} items/second")
        
        # Verify cache files exist and report sizes
        cache_files = [
            f"{csv_file}.tensors.pt",
            f"{csv_file}.themes.json",
            f"{csv_file}.openings.json"
        ]
        
        print(f"\n📁 Generated Cache Files:")
        total_size = 0
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                total_size += size_mb
                print(f"   ✅ {os.path.basename(cache_file)}: {size_mb:.1f} MB")
            else:
                print(f"   ❌ {os.path.basename(cache_file)}: Missing")
        
        print(f"   📦 Total cache size: {total_size:.1f} MB")
        
        # Test accessing samples across the dataset
        print(f"\n🔍 Verification - Testing sample access:")
        test_indices = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4, len(dataset)-1]
        all_passed = True
        
        for i in test_indices:
            try:
                sample = dataset[i]
                board_shape = sample['board'].shape
                num_themes = sample['themes'].sum().item()
                print(f"   ✅ Sample {i:,}: board {board_shape}, {num_themes} active labels")
            except Exception as e:
                print(f"   ❌ Sample {i:,}: Error - {e}")
                all_passed = False
        
        if all_passed:
            print(f"\n✅ Milestone 1 COMPLETED Successfully!")
            print(f"The full dataset tensor cache is ready for training and evaluation.")
        else:
            print(f"\n⚠️ Milestone 1 completed with some verification issues.")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to create full dataset cache: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create full dataset tensor cache for all 4,956,460 puzzles"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed progress bars and timing information"
    )
    
    args = parser.parse_args()
    success = create_full_dataset_cache(verbose=args.verbose)
    sys.exit(0 if success else 1)