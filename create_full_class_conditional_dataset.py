#!/usr/bin/env python3
"""
Script to create the full class-conditionally augmented dataset for Milestone 2.
This builds on Milestone 1's full dataset cache to create an augmented dataset 
with improved class balance through intelligent reflection augmentation.

Usage:
    python create_full_class_conditional_dataset.py [--verbose]
    
This will create:
    - lichess_db_puzzle.csv.tensors.pt_conditional_full (full augmented dataset)
    - lichess_db_puzzle.csv.tensors.pt_conditional_full.augmented_indices.json
    - Updated cooccurrence analysis for the full dataset
"""

import os
import sys
import time
import argparse
from dataset import ChessPuzzleDataset

def create_full_class_conditional_dataset(verbose=False, test_resume=None):
    """Create the full class-conditionally augmented dataset for Milestone 2."""
    
    print("Creating Full Class-Conditional Augmented Dataset (Milestone 2)")
    print("=" * 70)
    print("This will process all 4,956,460 puzzles and create an intelligently")
    print("augmented dataset with improved class balance.")
    print()
    
    # Use the main CSV file
    csv_file = "processed_lichess_puzzle_files/lichess_db_puzzle.csv"
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        print("Please ensure the lichess_db_puzzle.csv file is in the processed_lichess_puzzle_files directory.")
        return False
    
    # Check for Milestone 1 dependencies (full dataset cache)
    required_milestone1_files = [
        "processed_lichess_puzzle_files/lichess_db_puzzle.csv.tensors.pt",
        "processed_lichess_puzzle_files/lichess_db_puzzle.csv.themes.json",
        "processed_lichess_puzzle_files/lichess_db_puzzle.csv.openings.json"
    ]
    
    missing_m1_files = [f for f in required_milestone1_files if not os.path.exists(f)]
    if missing_m1_files:
        print("Missing Milestone 1 dependencies:")
        for f in missing_m1_files:
            print(f"   - {f}")
        print()
        print("Please run 'python create_full_dataset_cache.py' first to create")
        print("the full dataset tensor cache (Milestone 1).")
        return False
    
    print("Milestone 1 dependencies found:")
    for cache_file in required_milestone1_files:
        size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"   {os.path.basename(cache_file)}: {size_mb:.1f} MB")
    print()
    
    # Parameters for full class-conditional augmentation
    params = {
        'csv_file': csv_file,
        'cache_size': 1000,  # Keep memory cache small
        'class_conditional_augmentation': True,  # Enable intelligent augmentation
        'augment_with_reflections': False,  # Let class-conditional handle reflections
        'low_memory': True,  # Safe for large datasets
        'use_cache': True,  # Use Milestone 1 cache as foundation
        'verbose_progress': verbose,  # Pass verbose flag
        'rarity_threshold': None,  # Auto-determine from full dataset
        'full_class_conditional': True,  # Enable full class-conditional mode (Milestone 2)
        'test_resume': test_resume  # Pass test_resume parameter for testing
    }
    
    print("Class-Conditional Augmentation Parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    print()
    
    try:
        print("Starting full class-conditional dataset creation...")
        start_time = time.time()
        
        # Create dataset instance - this will create the full augmented cache
        dataset = ChessPuzzleDataset(**params)
        
        end_time = time.time()
        elapsed = end_time - start_time
        hours = elapsed / 3600
        
        print()
        print("Full Class-Conditional Dataset Creation Complete!")
        print("=" * 60)
        print(f"Results:")
        print(f"   Total time: {hours:.2f} hours ({elapsed:.0f} seconds)")
        print(f"   Original dataset: 4,956,459 puzzles")
        print(f"   Augmented dataset: {len(dataset):,} samples")
        print(f"   Augmentation ratio: {len(dataset) / 4956459:.2f}x")
        print(f"   Processing rate: {len(dataset) / elapsed:.2f} items/second")
        
        # Calculate augmentation statistics
        if hasattr(dataset, 'augmented_indices'):
            augmented_count = len(dataset.augmented_indices)
            total_count = len(dataset)
            original_count = total_count - augmented_count
            print(f"   Original samples: {original_count:,}")
            print(f"   Augmented samples: {augmented_count:,}")
            if original_count > 0:
                print(f"   Augmentation rate: {(augmented_count/original_count)*100:.1f}%")
            else:
                print(f"   Augmentation rate: N/A (no original samples)")
        
        # Verify cache files exist and report sizes
        conditional_cache = f"{csv_file}.tensors.pt_conditional_full"
        augmented_indices = f"{conditional_cache}.augmented_indices.json"
        
        cache_files = [conditional_cache, augmented_indices]
        
        print()
        print("Generated Cache Files:")
        total_size = 0
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                size_mb = os.path.getsize(cache_file) / (1024 * 1024)
                total_size += size_mb
                print(f"   {os.path.basename(cache_file)}: {size_mb:.1f} MB")
            else:
                print(f"   {os.path.basename(cache_file)}: Missing")
        
        print(f"   Total augmented cache size: {total_size:.1f} MB")
        
        # Test accessing samples across the dataset
        print()
        print("Verification - Testing sample access:")
        test_indices = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4, len(dataset)-1]
        all_passed = True
        
        reflection_count = 0
        for i, idx in enumerate(test_indices):
            try:
                sample = dataset[idx]
                board_shape = sample['board'].shape
                num_themes = sample['themes'].sum().item()
                is_reflection = sample.get('is_reflection', False)
                if is_reflection:
                    reflection_count += 1
                status = "reflection" if is_reflection else "original"
                print(f"   Sample {idx:,}: board {board_shape}, {num_themes} themes ({status})")
            except Exception as e:
                print(f"   Sample {idx:,}: Error - {e}")
                all_passed = False
        
        print(f"   Reflections in test samples: {reflection_count}/{len(test_indices)}")
        
        if all_passed:
            print()
            print("Milestone 2 COMPLETED Successfully!")
            print("The full class-conditionally augmented dataset is ready for training.")
            print()
            print("Next steps:")
            print("- Use '--full_class_conditional' flag with train.py")
            print("- Compare training performance vs. full dataset and original conditional dataset")
        else:
            print()
            print("Milestone 2 completed with some verification issues.")
            
        return True
        
    except Exception as e:
        print(f"Failed to create full class-conditional dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create full class-conditionally augmented dataset (Milestone 2)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed progress bars and timing information"
    )
    parser.add_argument(
        "--test_resume", 
        choices=["chunks", "segments", "final"],
        help="Test resume functionality by aborting at different stages: "
             "chunks (after 10 chunks), segments (after 3 segments), final (before final save)"
    )
    
    args = parser.parse_args()
    success = create_full_class_conditional_dataset(verbose=args.verbose, test_resume=args.test_resume)
    sys.exit(0 if success else 1)