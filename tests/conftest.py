import os
import sys
import pytest
from pathlib import Path

# Add the parent directory to sys.path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Custom command line option to enable ISC tests
def pytest_addoption(parser):
    parser.addoption(
        "--run-isc", action="store_true", default=False, help="Run ISC tests that might incur costs"
    )

# Custom marker for ISC tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "isc: mark tests that might incur costs on ISC infrastructure"
    )

# Skip ISC tests unless --run-isc is used
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-isc"):
        skip_isc = pytest.mark.skip(reason="Need --run-isc option to run")
        for item in items:
            if "isc" in item.keywords:
                item.add_marker(skip_isc)

# Define fixtures that can be used across tests
@pytest.fixture
def test_dataset_path():
    """Fixture to provide the path to the test dataset"""
    csv_file = os.path.join('dataset', 'lichess_db_puzzle_test.csv')
    
    # Check if file exists, otherwise look in processed_lichess_puzzle_files
    if not os.path.exists(csv_file):
        csv_file = os.path.join('processed_lichess_puzzle_files', 'lichess_db_puzzle_test.csv')
    
    # If still not found, raise error
    if not os.path.exists(csv_file):
        pytest.skip("Test dataset not found, skipping test")
        
    return csv_file