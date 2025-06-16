# Model Evaluation

This document describes the evaluation scripts for the chess theme classifier.

## Overview

We have three different evaluation scripts:

1. `evaluate_model_classification.py` - The original evaluation script
2. `evaluate_model_simple.py` - A simplified version for basic testing
3. `evaluate_model_fixed.py` - The recommended evaluation script with proper label mapping

## Key Features

### evaluate_model_fixed.py

This is the recommended evaluation script that:

- Correctly maps between training dataset indices and test dataset indices
- Supports adaptive thresholding to automatically find the optimal threshold
- Generates confusion matrices for key chess themes
- Provides detailed per-theme performance metrics
- Works with any model checkpoint

#### Usage

```bash
# Basic usage with adaptive thresholding (recommended)
python evaluate_model_fixed.py --num_samples=50

# With fixed threshold
python evaluate_model_fixed.py --num_samples=50 --threshold=0.3

# With specific checkpoint
python evaluate_model_fixed.py --checkpoint=checkpoints/my_checkpoint.pth
```

#### Output

The script outputs:
- Confusion matrix visualizations in `analysis/matrices/`
- Per-theme precision, recall, and F1 scores
- Overall model performance metrics
- Debugging information for the first few samples

### evaluate_model_simple.py

A simplified evaluation script that:
- Uses a basic approach to evaluate model performance
- Focuses only on key chess themes
- Provides simpler output than the full evaluation script

#### Usage

```bash
python evaluate_model_simple.py --num_samples=20 --threshold=0.3
```

### evaluate_model_classification.py

The original evaluation script that:
- Evaluates model performance on the entire label set
- Has more complex code and outputs
- Requires careful handling of label mapping

#### Usage

```bash
python evaluate_model_classification.py --num_samples=1000 --threshold=0.3
```

## Evaluation Process

1. **Label Mapping**: The scripts map between training dataset indices (1616 labels) and test dataset indices (86 labels)
2. **Adaptive Thresholding**: If no threshold is specified, the optimal threshold is calculated automatically
3. **Confusion Matrix Generation**: Creates visual confusion matrices showing the relationships between actual and predicted themes
4. **Performance Metrics**: Calculates precision, recall, and F1 score for each theme and overall

## Key Themes

The evaluation focuses on these key chess themes:
- advantage
- crushing
- endgame
- hangingPiece
- long
- middlegame
- short

## Tips

- Use adaptive thresholding when possible for the best results
- Start with a small number of samples (e.g., 20) for quick testing
- For thorough evaluation, use 100+ samples
- Focus on F1 score as the primary metric for overall performance
- Check the confusion matrix to understand which themes are being confused