#!/usr/bin/env python3
"""
Interactive script to test and demonstrate the dataset frequency features.
"""

from dataset import ChessPuzzleDataset

# Load dataset - using the small dataset for quick testing
dataset = ChessPuzzleDataset("lichess_db_puzzle.csv")
print(f"Loaded dataset with {len(dataset)} puzzles")

# Display theme frequencies
print("\nTheme frequencies:")
theme_freqs = dataset.get_theme_frequencies()
for theme, count in theme_freqs:
    print(f"{theme}: {count}")

# Display opening frequencies (if any)
print("\nOpening frequencies:")
opening_freqs = dataset.get_opening_frequencies()
for opening, count in opening_freqs:
    print(f"{opening}: {count}")

# Individual lookup
if theme_freqs:
    top_theme = theme_freqs[0][0]
    print(f"\nFrequency of most common theme '{top_theme}': {dataset.get_theme_frequency(top_theme)}")

# Demonstrate cache reading (load dataset again)
print("\nLoading dataset again to demonstrate cache reading:")
dataset2 = ChessPuzzleDataset("lichess_db_puzzle_small.csv")
print(f"Reloaded dataset with {len(dataset2)} puzzles")

# Verify theme frequencies are the same
theme_freqs2 = dataset2.get_theme_frequencies()
if theme_freqs == theme_freqs2:
    print("Theme frequencies match after reloading")
else:
    print("WARNING: Theme frequencies don't match after reloading!")

print("\nDone.")