#!/usr/bin/env python3
"""
Pytest test file to verify the theme and opening frequency functionality in the ChessPuzzleDataset.
"""

import os
import json
import pytest
from dataset import ChessPuzzleDataset

@pytest.fixture
def dataset():
    """Create a dataset fixture for tests"""
    if os.path.exists("lichess_db_puzzle_small.csv"):
        return ChessPuzzleDataset("lichess_db_puzzle_small.csv")
    elif os.path.exists("lichess_db_puzzle_test.csv"):
        return ChessPuzzleDataset("lichess_db_puzzle_test.csv")
    else:
        pytest.skip("No suitable dataset found for testing")

def test_cache_files_created(dataset):
    """Test that cache files are created with correct format"""
    cache_dir = os.path.dirname(dataset.csv_file) or '.'
    themes_cache = dataset.themes_cache_file
    openings_cache = dataset.openings_cache_file
    
    # Verify files exist
    assert os.path.exists(themes_cache), "Themes cache file not created"
    assert os.path.exists(openings_cache), "Openings cache file not created"
    
    # Verify content structure
    with open(themes_cache, 'r') as f:
        themes_data = json.load(f)
        assert isinstance(themes_data, list), "Themes data should be a list"
        assert len(themes_data) > 0, "Themes data should not be empty"
        # Check first item has [name, count] format
        assert isinstance(themes_data[0], list), "Theme items should be [name, count] pairs"
        assert len(themes_data[0]) == 2, "Theme items should have exactly 2 elements"
        assert isinstance(themes_data[0][0], str), "First element should be theme name"
        assert isinstance(themes_data[0][1], int), "Second element should be theme count"
    
    with open(openings_cache, 'r') as f:
        openings_data = json.load(f)
        assert isinstance(openings_data, list), "Openings data should be a list"
        if len(openings_data) > 0:  # Some datasets might not have openings
            assert isinstance(openings_data[0], list), "Opening items should be [name, count] pairs"
            assert len(openings_data[0]) == 2, "Opening items should have exactly 2 elements"
            assert isinstance(openings_data[0][0], str), "First element should be opening name"
            assert isinstance(openings_data[0][1], int), "Second element should be opening count"

def test_frequency_methods(dataset):
    """Test that frequency methods return expected data"""
    # Test theme frequencies
    theme_freqs = dataset.get_theme_frequencies()
    assert isinstance(theme_freqs, list), "Theme frequencies should be a list"
    assert len(theme_freqs) > 0, "Theme frequencies should not be empty"
    
    # Verify sorting (descending by frequency)
    if len(theme_freqs) > 1:
        assert theme_freqs[0][1] >= theme_freqs[1][1], "Themes should be sorted by frequency (descending)"
    
    # Test opening frequencies
    opening_freqs = dataset.get_opening_frequencies()
    assert isinstance(opening_freqs, list), "Opening frequencies should be a list"
    
    # Test getting individual frequencies
    if theme_freqs:
        top_theme = theme_freqs[0][0]
        theme_count = dataset.get_theme_frequency(top_theme)
        assert theme_count > 0, "Top theme should have positive frequency"
        assert theme_count == theme_freqs[0][1], "get_theme_frequency should match get_theme_frequencies"
    
    if opening_freqs:
        top_opening = opening_freqs[0][0]
        opening_count = dataset.get_opening_frequency(top_opening)
        assert opening_count == opening_freqs[0][1], "get_opening_frequency should match get_opening_frequencies"
    
    # Test non-existent theme/opening
    assert dataset.get_theme_frequency("nonexistent_theme") == 0, "Non-existent theme should have frequency 0"
    assert dataset.get_opening_frequency("nonexistent_opening") == 0, "Non-existent opening should have frequency 0"

def test_theme_set_matches_frequency_keys(dataset):
    """Test that all_themes and theme_counts have the same keys"""
    all_themes = dataset.all_themes
    theme_counts_keys = set(dataset.theme_counts.keys())
    assert all_themes == theme_counts_keys, "Theme set should match theme frequency keys"
    
    all_openings = dataset.all_opening_tags
    opening_counts_keys = set(dataset.opening_counts.keys())
    assert all_openings == opening_counts_keys, "Opening set should match opening frequency keys"