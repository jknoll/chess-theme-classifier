"""
This file provides a drop-in replacement for the original ChessPuzzleDataset
to support the optimizations without changing the existing interface.
"""

import os
import time
import torch
from torch.utils.data import Dataset
from optimized_dataset import OptimizedChessPuzzleDataset

# Override and extend the import path to use the optimized dataset
from dataset import ChessPuzzleDataset as OriginalChessPuzzleDataset

class ChessPuzzleDataset(OriginalChessPuzzleDataset):
    """
    Drop-in replacement for the original ChessPuzzleDataset that uses
    the optimized implementation under the hood.
    """
    
    def __init__(self, csv_file):
        """
        Initialize the dataset with optimized implementation.
        
        Args:
            csv_file (str): Path to the CSV file with chess puzzles
        """
        # Skip the original __init__ method and use the optimized dataset instead
        # We're overriding the inheritance initialization
        Dataset.__init__(self)
        
        # Start the timer to measure initialization time
        start_time = time.time()
        
        # Create the optimized dataset
        self._optimized_dataset = OptimizedChessPuzzleDataset(csv_file)
        
        # Copy attributes from the optimized dataset
        self.csv_file = csv_file
        self.puzzle_data = self._optimized_dataset.puzzle_data
        self.all_themes = self._optimized_dataset.all_themes
        self.all_opening_tags = self._optimized_dataset.all_opening_tags
        self.all_labels = self._optimized_dataset.all_labels
        self.label_to_idx = self._optimized_dataset.label_to_idx
        self.themes_cache_file = self._optimized_dataset.themes_cache_file
        self.openings_cache_file = self._optimized_dataset.openings_cache_file
        
        # Print initialization time
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Optimized dataset initialized in {elapsed:.2f} seconds")
    
    def __len__(self):
        return len(self._optimized_dataset)
    
    def __getitem__(self, idx):
        """Get item from optimized dataset."""
        return self._optimized_dataset[idx]
    
    # All other methods will use the original implementation from OriginalChessPuzzleDataset