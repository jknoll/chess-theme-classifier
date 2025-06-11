# test_dataset_conditional_cache.py
# Tests the class conditional tensor cache functionality.
# This test is designed to verify that the conditional tensor cache is working correctly.
# It creates a small test dataset with known FENs and labels, and then tests that the cache file is created and contains the correct labels.
# At one point, this test was failing because the cache file was not being created correctly and contained no labels at all.
# It also tests that the cache file can be used to load the dataset in cache-only mode, and that the labels are correctly recovered.

import os
import sys
import pytest
import torch
import pandas as pd
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
from torch.serialization import add_safe_globals
from chess import Board, Piece, SQUARES

# Add the parent directory to sys.path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import ChessPuzzleDataset

# Add necessary classes to PyTorch's serialization allowlist to avoid FutureWarning
add_safe_globals([
    # Core modules and classes
    np, torch, pd, Board, Piece, SQUARES, ChessPuzzleDataset, 
    # Container types
    list, dict, set, frozenset, tuple, 
    # PyTorch tensor types
    torch.Tensor, torch.int8, torch.float32,
    # Basic Python types
    int, float, bool, str
])

class TestConditionalCache:
    """Test suite for the class conditional tensor cache functionality."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory for dataset files and cleanup afterwards."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_dataset(self, temp_dataset_dir):
        """Create a small test dataset with known FENs and labels."""
        # Create a small dataset with varied theme combinations
        data = {
            'FEN': [
                'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',  # Initial position
                'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',  # After e4
                'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',  # After e5
                'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2',  # After Nf3
            ],
            'Themes': [
                'opening',
                'middlegame',
                'endgame',
                'opening middlegame'
            ],
            'OpeningTags': [
                'e4',
                'e4 e5',
                'e4 e5',
                'e4 e5 Nf3'
            ]
        }
        df = pd.DataFrame(data)
        
        # Save to the temporary directory
        csv_file = os.path.join(temp_dataset_dir, 'lichess_db_puzzle.csv')
        df.to_csv(csv_file, index=False)
        
        return csv_file

    def test_conditional_cache_includes_labels(self, test_dataset):
        """Test that the conditional tensor cache includes both tensors and labels."""
        # Create dataset with class conditional augmentation
        dataset = ChessPuzzleDataset(test_dataset, class_conditional_augmentation=True)
        
        # Get the cache file path
        cache_file = f"{test_dataset}.tensors.pt_conditional"
        
        # Check that the cache file was created
        assert os.path.exists(cache_file), f"Cache file {cache_file} was not created"
        
        # Load the cache file directly to check its contents
        tensor_cache = torch.load(cache_file, weights_only=False)
        
        # Verify that cache is not empty
        assert len(tensor_cache) > 0, "Tensor cache is empty"
        
        # Check if the dataset object can get labels for its items
        test_idx = 0  # First item
        item = dataset[test_idx]
        
        # Now check if we can access the themes/labels vector for this item
        assert 'themes' in item, "Dataset item does not contain 'themes' key"
        
        # Verify that themes is a non-empty tensor with expected dimensions
        assert isinstance(item['themes'], torch.Tensor), "Themes should be a tensor"
        assert item['themes'].dim() == 1, "Themes should be a 1D tensor"
        assert item['themes'].size(0) > 0, "Themes tensor should not be empty"
        
        # This is the critical test - do we have any non-zero values in the themes tensor?
        # If the issue exists, all theme values will be zero as described in the bug report
        assert item['themes'].sum() > 0, "Themes tensor contains all zeros, no labels are being loaded"
        
        # Verify we can recover the original theme labels from the CSV
        # First, get the original themes from the CSV
        original_df = pd.read_csv(test_dataset)
        original_themes = original_df.iloc[test_idx]['Themes'].split()
        
        # Get theme indices from the tensor (where value is 1)
        non_zero_indices = torch.nonzero(item['themes']).squeeze().tolist()
        
        # Convert to list if there's only a single value
        if isinstance(non_zero_indices, int):
            non_zero_indices = [non_zero_indices]
        
        # Convert indices back to theme names
        recovered_themes = [dataset.all_labels[idx] for idx in non_zero_indices]
        
        # Check that we can recover at least some of the original themes
        # The test needs to check intersection rather than exact equality
        # because all_labels combines themes and opening tags
        assert set(original_themes).issubset(set(recovered_themes)), \
            f"Could not recover original themes: {original_themes} from tensor labels: {recovered_themes}"
        
        # The following assertion should fail with the current implementation
        # This test verifies we can reconstruct labels when using cache-only mode
        # Create a copy of the dataset without the CSV to simulate cache-only mode
        csv_backup = test_dataset + ".backup"
        shutil.copy(test_dataset, csv_backup)
        try:
            # Remove the original CSV to force cache-only mode
            os.remove(test_dataset)
            
            # Load dataset in cache-only mode
            cache_only_dataset = ChessPuzzleDataset(test_dataset, class_conditional_augmentation=True)
            
            # Get an item
            cache_item = cache_only_dataset[test_idx]
            
            # Check that we still have a themes tensor
            assert 'themes' in cache_item, "Cache-only dataset item does not contain 'themes' key"
            
            # Verify that themes tensor has non-zero values (the failing assertion)
            assert cache_item['themes'].sum() > 0, \
                "Cache-only themes tensor contains all zeros, no labels are stored in the cache"
            
            # Recover themes from cache-only dataset
            cache_non_zero_indices = torch.nonzero(cache_item['themes']).squeeze().tolist()
            if isinstance(cache_non_zero_indices, int):
                cache_non_zero_indices = [cache_non_zero_indices]
            
            cache_recovered_themes = [cache_only_dataset.all_labels[idx] for idx in cache_non_zero_indices]
            
            # Check that the recovered themes match the original
            assert set(original_themes).issubset(set(cache_recovered_themes)), \
                f"Cache-only mode: Could not recover themes: {original_themes} from: {cache_recovered_themes}"
                
        finally:
            # Restore the CSV
            if os.path.exists(csv_backup):
                shutil.copy(csv_backup, test_dataset)
                os.remove(csv_backup)

if __name__ == "__main__":
    # This allows running tests directly with python
    pytest.main(["-xvs", __file__])