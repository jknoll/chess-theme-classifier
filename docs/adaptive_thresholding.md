# Adaptive Thresholding for Metrics Calculation

## Problem Statement

During the early stages of training, our model outputs logits that are clustered around zero. When these outputs are passed through a sigmoid function to get probabilities, most values remain below the standard classification threshold of 0.5. This causes most predictions to be classified as negative, resulting in:

1. Zero or near-zero values for precision, recall, and F1 metrics
2. Inability to effectively track model progress in early training
3. Misleading metrics that suggest the model isn't learning when it actually is

The issue is particularly noticeable when running without `--test_mode`, as standard mode uses a fixed threshold of 0.5 for all binary classification decisions.

## Solution: Adaptive Thresholding

We've implemented an adaptive thresholding mechanism that dynamically adjusts classification thresholds during early training stages based on the distribution of model outputs. This approach allows for more meaningful metrics calculation when model outputs are still weak.

### Implementation Details

The adaptive thresholding algorithm:

1. Calculates class-specific statistics (mean and standard deviation) of prediction probabilities
2. Computes a lower threshold for each class using: `threshold = clamp(mean - 0.5 * std, min=0.05, max=0.5)`
3. Applies these adaptive thresholds on a per-class basis

This implementation has been added to all metric calculation functions:
- `jaccard_similarity`
- `precision_recall_f1`
- `get_classification_report`

### Benefits

1. **Non-zero metrics from the beginning**: Enables useful metric tracking from the very first epoch
2. **Class-specific adaptation**: Adjusts thresholds based on how difficult each class is to predict
3. **Smooth transition**: As training progresses and model outputs become stronger, thresholds gradually increase toward the standard 0.5
4. **Better debugging**: Provides more meaningful feedback during the critical early stages of training

### Example

Early in training, model outputs might have:
- Mean probability: 0.1
- Standard deviation: 0.05

The adaptive threshold would be:
- 0.1 - (0.5 * 0.05) = 0.075, clamped to min=0.05, max=0.5

This allows the model to register positive predictions for the most confident outputs, even when they're below the standard 0.5 threshold.

## Usage

Adaptive thresholding is enabled by default for all metrics calculations. If you want to disable it and use fixed thresholds:

```python
# For Jaccard similarity
jaccard = jaccard_similarity(outputs, labels, adaptive_threshold=False)

# For precision, recall, F1
precision, recall, f1 = precision_recall_f1(outputs, labels, adaptive_threshold=False)

# For classification report
report = get_classification_report(outputs, labels, adaptive_threshold=False)
```

## Technical Notes

- The minimum threshold is set to 0.05 to prevent extremely low thresholds that might lead to excessive false positives
- As model training progresses, output distributions typically shift toward higher values, causing the adaptive thresholds to gradually increase toward the standard 0.5
- This approach is particularly valuable for multi-label classification with imbalanced datasets, where some classes might take longer to learn than others