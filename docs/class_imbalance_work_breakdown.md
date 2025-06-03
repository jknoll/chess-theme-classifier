
# 📊 Class Imbalance Mitigation Work Breakdown (Multi-Label Chess Position Classifier)

## ✅ Overview

This document outlines the implementation plan to address class imbalance in a multi-label classification task involving chess position labeling by themes and openings. It integrates data-level and algorithm-level strategies.

---

## 1. 🧪 Data Augmentation

> **Note**: Only horizontal flipping is applicable and has already been implemented on a per-board level. When performing data augmentation by horizontally flipping, we need to ensure that we strip the opening labels and preserve only the theme labels for the board which results from the horizontal flip, since a horizontal flip destroys structure which is necessary to classify a particular opening. In all places below within Data Augmentation where "labels" are mentioned, we refer to the theme labels.

### ✅ Status
- [x] Horizontal flipping implemented.
- [x] Selective flipping during resampling implemented.

### 🔧 Next Steps
- ✅ **Task**: Implement class-conditional augmentation
  - ✅ **Detail**: Analyze class frequency (co-occurrence of themes).
  - ✅ **Detail**: Apply horizontal flip *only* to underrepresented multi-theme combinations. Do this by referring to multilabel_reflection_algorithm.md to understand the approach. Critique it if anything seems wrong.
  - ✅ **Detail**: Track and log augmentation so augmented samples can be traced.

- **Task**: Optionally: apply SMOTE-for-multilabel (MLSMOTE), if categorical features can be encoded.

📚 *Reference*: Charte et al., "MLSMOTE: Approaching imbalanced multilabel learning through synthetic instance generation" (2015) – [DOI:10.1016/j.knosys.2015.03.013](https://doi.org/10.1016/j.knosys.2015.03.013)

---

## 3. ⚖️ Cost-Sensitive Learning

> Adjust model training to penalize errors more heavily on minority labels. For this modification, we are concerned with both themes and openings.

### ✅ High-Level Tasks
- [x] Integrate per-label class weights into loss function.

### 🔧 Implementation Steps
- [x] **Task**: Compute inverse frequency of each individual label.
- [x] **Task**: Convert to normalized weights (e.g., softmax of inverse frequencies).
- [x] **Task**: Use `BCEWithLogitsLoss(pos_weight=...)` in PyTorch for weighted multilabel loss.
- [x] **Task**: Validate impact on precision/recall curves.

📚 *Reference*: Japkowicz & Stephen, "The class imbalance problem: A systematic study" (2002) – [DOI:10.1613/jair.953](https://doi.org/10.1613/jair.953)

---

## 4. 📏 Evaluation Metrics

> Shift from accuracy to multi-label metrics sensitive to imbalance.

### ✅ High-Level Tasks
- [x] Compute per-label precision, recall, F1.
- [x] Compute macro/micro-averaged metrics.

### 🔧 Implementation Steps
- [x] **Task**: Use `sklearn.metrics.classification_report()` with `average='micro'`, `'macro'`, and `'samples'`.
- [x] **Task**: Visualize label-wise performance to track minority-class behavior.
- [x] **Task**: Add metrics to training/validation logs and dashboards.
- [x] **Task**: Optionally: use label ranking loss or subset accuracy as additional metrics.

📚 *Reference*: Zhang & Zhou, "A Review on Multi-Label Learning Algorithms" (2013) – [DOI:10.1109/TKDE.2013.39](https://doi.org/10.1109/TKDE.2013.39)

Generate a histogram of the themes and openings distributions (see generate_histograms.py) before adding class-conditional augmentation and with it in effect.
