import os
import sys
import pytest
import subprocess
import torch
from pathlib import Path

# Add the parent directory to sys.path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the modules we want to test
from train import parse_args, init_distributed
from dataset import ChessPuzzleDataset
from model import Model

def test_parse_args():
    """Test that argument parsing works correctly"""
    # Test with minimal args
    test_args = parse_args([
        '--test_mode',
    ])
    assert test_args.test_mode is True
    assert test_args.single_gpu is False
    
    # Test with single_gpu flag
    test_args = parse_args([
        '--test_mode',
        '--single_gpu'
    ])
    assert test_args.test_mode is True
    assert test_args.single_gpu is True

def test_train_script_runs():
    """Test that the train.py script runs without crashing in test mode"""
    # Run the train.py script with --test_mode
    cmd = [sys.executable, str(Path(__file__).parent.parent / 'train.py'), 
           '--test_mode', '--single_gpu', '--epochs', '1']
    
    try:
        # Set a timeout to avoid hanging in case of issues
        # We're not capturing output to avoid making the test log too verbose
        result = subprocess.run(cmd, timeout=120, capture_output=True)
        
        # Check that the process completed successfully
        assert result.returncode == 0, f"Process failed with return code {result.returncode}: {result.stderr.decode()}"
        
    except subprocess.TimeoutExpired:
        pytest.fail("Training script timed out after 120 seconds")

def test_dataset_loads():
    """Test that the dataset can be loaded"""
    try:
        # Use the test dataset to keep it fast
        csv_file = os.path.join('dataset', 'lichess_db_puzzle_test.csv')
        
        # Check if file exists, otherwise look in processed_lichess_puzzle_files
        if not os.path.exists(csv_file):
            csv_file = os.path.join('processed_lichess_puzzle_files', 'lichess_db_puzzle_test.csv')
        
        dataset = ChessPuzzleDataset(csv_file, class_conditional_augmentation=True, low_memory=True)
        
        # Verify that the dataset has content
        assert len(dataset) > 0, "Dataset is empty"
        
        # Verify that we can access an item
        item = dataset[0]
        assert 'board' in item, "Dataset item does not contain 'board'"
        assert 'themes' in item, "Dataset item does not contain 'themes'"
        
    except Exception as e:
        pytest.fail(f"Dataset loading failed: {str(e)}")

def test_model_creation():
    """Test that the model can be created with valid parameters"""
    try:
        # Create a model with a small number of labels
        num_labels = 10
        model = Model(num_labels=num_labels, nlayers=2, embed_dim=32, inner_dim=128, attention_dim=32)
        
        # Verify model parameters
        assert model.fc3.out_features == num_labels, f"Model output size {model.fc3.out_features} doesn't match num_labels {num_labels}"
        
        # Test a forward pass with a dummy input (batch size 2, channels 1, height 8, width 8)
        dummy_input = torch.zeros(2, 1, 8, 8, dtype=torch.float32)
        output = model(dummy_input)
        
        # Check output shape
        assert output.shape == (2, num_labels), f"Model output shape {output.shape} doesn't match expected shape (2, {num_labels})"
        
    except Exception as e:
        pytest.fail(f"Model creation failed: {str(e)}")

if __name__ == "__main__":
    # This allows running tests directly with python
    pytest.main(["-xvs", __file__])