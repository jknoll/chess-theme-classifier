In docs/adaptive_thresholding.md we added a mechanism to do adaptive thresholding globally to evaluate_model_metrics.py. The mechanism used is to sweep a threshold and evaluate F1 loss and to choose the global threshold which minimizes F1 loss.

We also have some logic in `evaluate_model_metrics.py`, which calculates some values related to per-class thresholding. 

The calculations, however, are based on observing the typical activations in the network for a class, not on the F1 loss minimizing threshold sweep.

Replace those calculations. Instead, we wish to do the same threshold sweep and compute the threshold which causes F1 loss minimization, but we wish to do this on a per-class basis and to write an output file which includes these calculated per-class thresholds.

As we sweep the parameter, we will be generating calculated precision and recall for a class. Include these calculated precision and recall scores in the output file so that the file is suitable for use in constructing a precision-recall curve for each class.

Once these per-class thresholds are calculated, use them for calculating the F1 score, precision, and recall to be plotted in the graph in the current implementation of `evaluate_model_metrics.py`.

We will test this implementation by calling `evaluate_model_metrics.py` with `num_samples` equal to one.

## Implementation Tasks

- [x] Analyze current per-class thresholding logic in evaluate_model_metrics.py
- [x] Replace per-class threshold calculations with F1-optimizing threshold sweep
- [x] Generate precision-recall curves for each class during threshold sweep  
- [x] Update F1/precision/recall calculations to use per-class thresholds
- [x] Test implementation with num_samples=1 