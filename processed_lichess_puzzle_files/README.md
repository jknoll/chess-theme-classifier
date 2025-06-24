# Processed Lichess Puzzle Files

This directory contains pre-processed cache files derived from the original `lichess_db_puzzle.csv` dataset. These files enable faster training by skipping the time-consuming preprocessing steps.

The source file is distributed as a .csv.zst. After decompressing, it has this hash. These files can be re-derived automatically by placing a newer version of the .csv (i.e. a newer mtime) in the directory and loading the dataset.

Beware, however, as newer versions of the file from lichess may contain incrementally more labels (when a previously-unobserved opening subcategory is observed in an included puzzle), which implies a change in model architecture at the output layer.

```bash
md5sum processed_lichess_puzzle_files/lichess_db_puzzle.csv
7b58528b802e297d71e130c4b604f9e6  processed_lichess_puzzle_files/lichess_db_puzzle.csv
```
## Files and Their Purpose

1. **lichess_db_puzzle.csv.themes.json**
   - Contains all unique chess puzzle themes found in the dataset
   - Generated during the first run of ChessPuzzleDataset to avoid re-parsing theme labels

2. **lichess_db_puzzle.csv.openings.json**
   - Contains all unique chess opening tags found in the dataset
   - Generated during the first run of ChessPuzzleDataset to avoid re-parsing opening labels

3. **lichess_db_puzzle.csv.cooccurrence.json**
   - Contains statistics about label co-occurrence patterns
   - Used for class-conditional augmentation to identify rare label combinations
   - Includes information about which theme combinations appear frequently/rarely

4. **lichess_db_puzzle.csv.tensors.pt_conditional**
   - Pre-converted tensor representations of all chess board positions
   - Includes augmented (reflected) board positions for rare theme combinations
   - Significantly speeds up training by avoiding FEN-to-tensor conversion at runtime

5. **lichess_db_puzzle.csv.tensors.pt_conditional.augmented_indices.json**
   - Tracks which puzzle indices were augmented with reflections
   - Used to correctly map between original and augmented samples during training

6. **lichess_db_puzzle.csv.class_weights.pt**
   - Contains pre-computed class weights for handling label imbalance
   - Used with weighted loss functions to give more emphasis to rare themes
   - Improves model training on imbalanced datasets

These files are sufficient for a full training run without requiring the original CSV file. They contain all pre-processed data needed for efficient model training on a cluster.