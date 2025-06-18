The co-occurrence matrix is not a very well-suited visualization for a multilabel classifier settings:
	•	There is no unique “ground truth label” per instance—each sample can have multiple correct labels.
	•	True positives may appear off-diagonal, since multiple labels can co-occur legitimately.

We apply row normalization before rendering the matrix, so that each row becomes scannable as a probability distribution of labels likely to be predicted given a an actual label. This is valuable, but it means that any intuition about the diagonal getting higher and higher values (approaching probability one) is undercut by the fact there there may be many correct labels and the more there are, the more even a perfect prediction will have low values on its diagonal class=class cell.

Instead, we will focus on f1 score, precision, and recall. This is handled by `evaluate_model_metrics.py`. Remaining tasks:

[ ] Create a new folder /analysis/f1 and write the output of this script into that directory.
[ ] Ensure that we are using the same adaptive thresholding system as `evaluate_model_simple.py`, and include the threshold used and the number of samples in the filenames.
[ ] Modify the plot color scheme/alpha so that F1 score is more prominent and precision and recall are less prominent. If there is a way to set alpha/transparency, use lower values for precision and recall. If not, use a more vibrant/saturated color for F1 and more muted/desaturated color for P and R.
[ ] Remove the restriction which generates only a few labels of interest. Generate the graph and data for all labels.
[ ] Split the output and visualization into themes vs. openings.