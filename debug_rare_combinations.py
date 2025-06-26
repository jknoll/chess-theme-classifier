#!/usr/bin/env python3

import json
import pandas as pd

# Test the rare combinations loading logic
def test_rare_combinations():
    # Load the co-occurrence data
    cooc_file = "/home/j/Documents/git/chess-theme-classifier/processed_lichess_puzzle_files/lichess_db_puzzle.csv.cooccurrence.json"
    
    with open(cooc_file, 'r') as f:
        cache_data = json.load(f)
    
    print(f"Cache data keys: {list(cache_data.keys())}")
    print(f"Number of rare combinations in cache: {len(cache_data.get('rare_combinations', []))}")
    print(f"Rarity threshold: {cache_data.get('rarity_threshold')}")
    
    # Convert rare combinations from strings to frozensets (same as dataset.py)
    rare_combinations = {
        frozenset(eval(combo)) for combo in cache_data.get('rare_combinations', [])
    }
    
    print(f"Number of rare combinations after conversion: {len(rare_combinations)}")
    print(f"Sample rare combinations (first 5):")
    for i, combo in enumerate(list(rare_combinations)[:5]):
        print(f"  {i+1}: {combo}")
    
    # Now test with some actual puzzle data to see if any match
    csv_file = "/home/j/Documents/git/chess-theme-classifier/processed_lichess_puzzle_files/lichess_db_puzzle.csv"
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return rare_combinations
        
    print(f"\nTesting with actual puzzle data...")
    puzzle_data = pd.read_csv(csv_file)
    
    matches_found = 0
    total_puzzles_checked = min(10000, len(puzzle_data))  # Check first 10K puzzles
    
    for idx in range(total_puzzles_checked):
        themes = puzzle_data.iloc[idx]['Themes'].split()
        theme_labels = frozenset(themes)
        
        if theme_labels in rare_combinations:
            matches_found += 1
            if matches_found <= 5:  # Show first 5 matches
                print(f"  Match {matches_found}: {theme_labels}")
    
    print(f"\nFound {matches_found} matches out of {total_puzzles_checked} puzzles checked")
    print(f"Match rate: {matches_found/total_puzzles_checked*100:.2f}%")
    
    return rare_combinations

if __name__ == "__main__":
    import os
    rare_combinations = test_rare_combinations()