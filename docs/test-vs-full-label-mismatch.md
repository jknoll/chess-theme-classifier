# Test vs Full Dataset Label Mismatch Investigation

## Issue Summary

There appears to be a discrepancy between the labels in the test dataset (`lichess_db_puzzle_test.csv`) and the full dataset (`lichess_db_puzzle.csv`). Specifically, when computing F1 scores:
- The full dataset appears to have no positive labels
- The test dataset contains positive labels

This is unexpected since the test dataset is supposed to be the first N lines of the full dataset.

## Investigation

### Dataset Loading

After examining the codebase, I identified the root causes of this discrepancy:

1. **Dataset Path Configuration**: 
   - When using the full dataset, the code in `train.py` and `diagnose_jaccard.py` uses the `processed_lichess_puzzle_files` directory
   - When using the test dataset, it uses the `dataset` directory

2. **Directory Structure Issues**:
   - The full dataset path in `processed_lichess_puzzle_files` doesn't contain the actual CSV file, only cache files
   - The test dataset path in `dataset` contains both the CSV and cache files

3. **Cache-Only Mode Behavior**:
   - When the original CSV doesn't exist, the `ChessPuzzleDataset` class falls back to cache-only mode
   - In cache-only mode, the `__getitem__` method has specific logic (lines 835-843) that returns empty label vectors when `self.csv_exists` is `False`

```python
# In cache-only mode, we won't have labels from the CSV
# Instead, we'll rely on the labels embedded in the cache files
if not self.csv_exists:
    # In cache-only mode, since we don't have the original labels,
    # we'll use an empty label vector
    # The model will use this for prediction but not for training
    pass
else:
    # Always include theme labels
    for theme in themes:
        if theme in self.label_to_idx:  # Ensure theme exists in our index
            theme_idx = self.label_to_idx[theme]
            label_vector[theme_idx] = 1
```

This means that when running with the full dataset in `processed_lichess_puzzle_files`, the dataset returns empty label vectors (no positive labels) because it operates in cache-only mode.

## Root Cause

The main issue is that the `processed_lichess_puzzle_files` directory contains cache files but not the original CSV, which triggers the cache-only mode in the dataset class. In this mode, the dataset doesn't populate the label vectors, resulting in all-zero labels.

The test dataset works correctly because the CSV file exists in the `dataset` directory, allowing the dataset class to properly populate the label vectors.

## Solution

To fix this issue, implement one of the following solutions:

1. **Copy the CSV to the processed directory**:
   - Copy `lichess_db_puzzle.csv` to the `processed_lichess_puzzle_files` directory

2. **Modify the dataset class to load labels from cache**:
   - Update the `__getitem__` method to load labels from cache files when in cache-only mode
   - This would require adding a new cache file specifically for labels

3. **Use consistent directory structure**:
   - Ensure all datasets (test and full) are accessed from the same directory structure
   - Update the file paths in the code to reference this consistent structure

4. **Fix the dataset loading logic**:
   - Modify the training scripts to use the `dataset` directory for both test and full datasets
   - Ensure the appropriate CSV files exist in this directory

## Recommended Solution

The most straightforward solution is to copy the original CSV file to the `processed_lichess_puzzle_files` directory. This preserves the existing code structure while fixing the issue:

```bash
cp dataset/lichess_db_puzzle.csv processed_lichess_puzzle_files/
```

Alternatively, modify the dataset initialization in `train.py` and `diagnose_jaccard.py` to use the `dataset` directory for both test and full datasets:

```python
# Instead of using processed directory for full dataset
csv_file = os.path.join('dataset', csv_filename)
```

This change ensures consistent behavior between test and full dataset evaluation.