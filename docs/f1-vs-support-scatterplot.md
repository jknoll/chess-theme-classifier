# F1 vs Support Scatter Plot

This document describes the F1 vs Support scatter plot visualization for the chess theme classifier.

## Overview

The F1 vs Support scatter plot shows the relationship between model performance (F1 score) and data availability (support/number of positive examples) across all chess themes and openings.

## Implementation Tasks

[x] **Verify support calculation**: Confirmed that `num_positive_examples` in the thresholds file equals support (sum of positive labels per class) by inspecting `evaluate_model_metrics.py:392`

[x] **Extract F1 scores**: Extract maximum F1 scores for each class from `analysis/f1/per_class_pr_curves.csv` without running new evaluation

[x] **Create standalone script**: Developed `f1_vs_support_scatterplot.py` as a standalone script similar to the PR curve generator

[x] **Color-coded visualization**: Distinguish between themes (blue) and openings (red) using different colors in the scatter plot

[x] **Output directory**: Save plot to `analysis/scatter/` directory

[x] **Generate visualization**: Successfully generated scatter plot with 65 themes and 596 openings plotted

## Script Usage

```bash
# Generate F1 vs Support scatter plot
python f1_vs_support_scatterplot.py

# With verbose output
python f1_vs_support_scatterplot.py --verbose

# Custom input/output paths
python f1_vs_support_scatterplot.py --pr_curves_file analysis/f1/per_class_pr_curves.csv --thresholds_file analysis/f1/per_class_thresholds.csv --output_dir analysis/scatter
```

## Features

- **Data Integration**: Combines PR curve data (for F1 scores) with threshold data (for support)
- **Class Classification**: Automatically distinguishes themes from openings based on naming conventions
- **Visual Distinction**: Uses blue for themes, red for openings
- **Statistical Summary**: Displays mean F1 scores and support values for both categories
- **Full Coverage**: Includes all classes with positive examples (661 total classes)

## Output

The script generates:
- `analysis/scatter/f1_vs_support_scatter.png`: Scatter plot visualization
- Console summary with theme/opening counts and statistics
- Statistical overlay showing mean performance for each category

## Insights

The visualization reveals:
- Performance differences between themes and openings
- Relationship between data availability (support) and model performance
- Distribution of classes across the performance-support space
- Identification of well-performing themes vs. challenging openings with limited data