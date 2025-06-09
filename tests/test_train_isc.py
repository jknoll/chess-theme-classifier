import os
import sys
import pytest
import torch
from pathlib import Path

# Add the parent directory to sys.path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mark these tests as ISC tests so they'll be skipped unless --run-isc is used
pytestmark = pytest.mark.isc

# Try to import the modules we want to test
try:
    from train_isc import get_args_parser
    from cycling_utils import MetricsTracker, InterruptableDistributedSampler, AtomicDirectory
except ImportError as e:
    # In case of import error, we'll skip the tests
    pytest.skip(f"Failed to import from train_isc.py: {e}", allow_module_level=True)

def test_arg_parsing():
    """Test that argument parsing works correctly for train-isc.py"""
    parser = get_args_parser()
    
    # Test with test_mode flag
    args = parser.parse_args(['--test-mode'])
    assert args.test_mode is True
    assert args.bs == 64  # Default batch size
    
    # Test with custom batch size
    args = parser.parse_args(['--test-mode', '--bs', '32'])
    assert args.test_mode is True
    assert args.bs == 32

def test_cycling_utils_imports():
    """Test that all required cycling_utils modules can be imported"""
    # If the imports failed, the test would have been skipped already
    assert 'MetricsTracker' in globals()
    assert 'InterruptableDistributedSampler' in globals()
    assert 'AtomicDirectory' in globals()
    
    # Create a simple metrics tracker and test it
    metrics = MetricsTracker()
    metrics.update({"test": 1.0})
    assert "test" in metrics.local
    assert metrics.local["test"] == 1.0

def test_metrics_tracker():
    """Test the MetricsTracker functionality"""
    metrics = MetricsTracker()
    
    # Test update and local access
    metrics.update({"loss": 0.5, "accuracy": 0.8})
    assert metrics.local["loss"] == 0.5
    assert metrics.local["accuracy"] == 0.8
    
    # Test reset_local
    metrics.reset_local()
    assert "loss" not in metrics.local
    assert "accuracy" not in metrics.local

# Important note for developers:
# DO NOT run the isc train command in automated tests as it incurs costs
# Any test that would execute 'isc train' should be manually run only when needed

if __name__ == "__main__":
    # This allows running tests directly with python
    # Note: This will still skip ISC tests unless --run-isc is passed
    pytest.main(["-xvs", "--run-isc", __file__])