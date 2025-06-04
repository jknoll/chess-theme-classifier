import pytest
import torch
import pandas as pd
import numpy as np
import os
from dataset import ChessPuzzleDataset

@pytest.fixture
def test_dataset():
    # Create a small test dataset
    data = {
        'FEN': ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],  # Initial chess position
        'Themes': ['opening'],
        'OpeningTags': ['e4']
    }
    df = pd.DataFrame(data)
    csv_file = 'test_dataset_reflection.csv'
    df.to_csv(csv_file, index=False)
    
    yield csv_file
    
    # Clean up
    if os.path.exists(csv_file):
        os.remove(csv_file)
    if os.path.exists(f"{csv_file}.themes.json"):
        os.remove(f"{csv_file}.themes.json")
    if os.path.exists(f"{csv_file}.openings.json"):
        os.remove(f"{csv_file}.openings.json")
    if os.path.exists(f"{csv_file}.tensors.pt"):
        os.remove(f"{csv_file}.tensors.pt")
    if os.path.exists(f"{csv_file}.tensors.pt_reflected"):
        os.remove(f"{csv_file}.tensors.pt_reflected")

def test_reflect_tensor_horizontally():
    """Test the static method for horizontal reflection of a tensor"""
    # Create a test tensor with distinct values in each position
    test_tensor = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20, 21, 22, 23],
        [24, 25, 26, 27, 28, 29, 30, 31],
        [32, 33, 34, 35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44, 45, 46, 47],
        [48, 49, 50, 51, 52, 53, 54, 55],
        [56, 57, 58, 59, 60, 61, 62, 63]
    ], dtype=torch.int8)

    # Apply horizontal reflection
    reflected = ChessPuzzleDataset.reflect_tensor_horizontally(test_tensor)
    
    # Check that each row is reversed
    for i in range(8):
        assert torch.all(reflected[i] == torch.flip(test_tensor[i], [0]))
    
    # Check specific positions
    assert reflected[0, 0].item() == test_tensor[0, 7].item()
    assert reflected[3, 2].item() == test_tensor[3, 5].item()
    assert reflected[7, 7].item() == test_tensor[7, 0].item()

def test_augmented_dataset(test_dataset):
    """Test the dataset with augmentation enabled"""
    csv_file = test_dataset
    
    # Create dataset with augmentation
    augmented_dataset = ChessPuzzleDataset(csv_file, augment_with_reflections=True)
    
    # Check that the length is doubled
    assert len(augmented_dataset) == 2
    
    # Get the original and reflected items
    original_item = augmented_dataset[0]
    reflected_item = augmented_dataset[1]
    
    # Check that the original item is not marked as a reflection
    assert not original_item['is_reflection']
    
    # Check that the reflected item is marked as a reflection
    assert reflected_item['is_reflection']
    
    # Check that the labels are the same
    assert torch.all(original_item['themes'] == reflected_item['themes'])
    
    # Check that the FEN strings are the same (reference to original)
    assert original_item['fen'] == reflected_item['fen']
    
    # Check that the reflected board is correctly flipped
    original_tensor = original_item['board']
    reflected_tensor = reflected_item['board']
    
    # Manually flip the original board and check equality with the reflected board
    manually_reflected = ChessPuzzleDataset.reflect_tensor_horizontally(original_tensor)
    assert torch.all(manually_reflected == reflected_tensor)
    
    # Check specific pieces (in the starting position)
    # King and Queen positions should be swapped in the reflection
    assert original_tensor[7, 4].item() == reflected_tensor[7, 3].item()  # White King
    assert original_tensor[7, 3].item() == reflected_tensor[7, 4].item()  # White Queen
    assert original_tensor[0, 4].item() == reflected_tensor[0, 3].item()  # Black King
    assert original_tensor[0, 3].item() == reflected_tensor[0, 4].item()  # Black Queen

def test_tensor_cache_creation(test_dataset):
    """Test that tensor cache is created correctly with reflections"""
    csv_file = test_dataset
    
    # Create dataset with augmentation
    dataset = ChessPuzzleDataset(csv_file, augment_with_reflections=True)
    
    # Check that tensor cache exists and has the right size
    assert len(dataset.tensor_cache) == 2
    
    # Check that the first tensor is the original and the second is reflected
    assert torch.all(ChessPuzzleDataset.reflect_tensor_horizontally(dataset.tensor_cache[0]) == dataset.tensor_cache[1])
    
    # Create a regular dataset for comparison
    regular_dataset = ChessPuzzleDataset(csv_file, augment_with_reflections=False)
    
    # Check that tensor cache exists and has the right size
    assert len(regular_dataset.tensor_cache) == 1
    
    # Check that the tensors in the regular dataset match the originals in the augmented dataset
    assert torch.all(regular_dataset.tensor_cache[0] == dataset.tensor_cache[0])

def test_with_small_real_dataset():
    """Test with a real dataset file"""
    dataset_path = os.path.join('dataset', 'lichess_db_puzzle_test.csv')
    if not os.path.exists(dataset_path):
        pytest.skip("Test dataset not available")
    
    # Load the dataset with and without augmentation
    regular_dataset = ChessPuzzleDataset(dataset_path, augment_with_reflections=False)
    augmented_dataset = ChessPuzzleDataset(dataset_path, augment_with_reflections=True)
    
    # Check that the augmented dataset is twice as large
    assert len(augmented_dataset) == 2 * len(regular_dataset)
    
    # Sample a few indices to check
    for i in range(0, len(regular_dataset), max(1, len(regular_dataset) // 10)):
        # Get items at this position
        regular_item = regular_dataset[i]
        aug_original = augmented_dataset[i*2]
        aug_reflected = augmented_dataset[i*2 + 1]
        
        # Check that the original in augmented matches regular
        assert torch.all(aug_original['board'] == regular_item['board'])
        assert torch.all(aug_original['themes'] == regular_item['themes'])
        
        # Check that the reflection is correctly done
        manually_reflected = ChessPuzzleDataset.reflect_tensor_horizontally(regular_item['board'])
        assert torch.all(manually_reflected == aug_reflected['board'])
        
        # Check that labels are the same between original and reflection
        assert torch.all(aug_original['themes'] == aug_reflected['themes'])

if __name__ == "__main__":
    pytest.main(["-v", __file__])