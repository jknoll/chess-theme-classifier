#!/usr/bin/env python3
"""
Test the complete augmentation flow to debug why 0 augmented samples are created.
"""

import pandas as pd
import numpy as np
from dataset import ChessPuzzleDataset

def test_augmentation_flow():
    """Test the complete augmentation flow with a small sample"""
    
    print("Testing Full Class-Conditional Augmentation Flow")
    print("=" * 60)
    
    # Create a small test dataset to verify the flow
    csv_file = "processed_lichess_puzzle_files/lichess_db_puzzle.csv"
    
    print("1. Loading small sample for initial testing...")
    df = pd.read_csv(csv_file)
    print(f"   Total dataset size: {len(df):,} puzzles")
    
    # Create a small sample for testing
    test_size = 1000
    test_df = df.head(test_size)
    print(f"   Using test sample: {len(test_df):,} puzzles")
    
    # Save test sample
    test_csv = "test_sample_1000.csv"
    test_df.to_csv(test_csv, index=False)
    
    print("\n2. Creating dataset with full class-conditional augmentation...")
    
    try:
        dataset = ChessPuzzleDataset(
            csv_file=test_csv,
            cache_size=1000,
            class_conditional_augmentation=True,
            full_class_conditional=True,
            augment_with_reflections=False,  # Let class-conditional handle this
            low_memory=True,
            use_cache=False,  # Force fresh creation
            verbose_progress=True,
            rarity_threshold=None
        )
        
        print(f"\n3. Results:")
        print(f"   Dataset length: {len(dataset):,}")
        print(f"   Tensor cache length: {len(dataset.tensor_cache) if dataset.tensor_cache else 0:,}")
        print(f"   Augmented indices: {len(dataset.augmented_indices):,}")
        
        # Check some samples
        if len(dataset) > 0:
            print(f"\n4. Sample verification:")
            for i in [0, len(dataset)//4, len(dataset)//2, len(dataset)-1]:
                if i < len(dataset):
                    sample = dataset[i]
                    is_augmented = "augmented" if i in dataset.augmented_indices else "original"
                    themes_count = sample['themes'].sum().item()
                    print(f"   Sample {i}: {sample['board'].shape} board, {themes_count} themes ({is_augmented})")
        
        # Clean up
        import os
        if os.path.exists(test_csv):
            os.remove(test_csv)
            
        return len(dataset.augmented_indices)
        
    except Exception as e:
        print(f"   Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == '__main__':
    augmented_count = test_augmentation_flow()
    print(f"\nFinal result: {augmented_count} augmented samples created")