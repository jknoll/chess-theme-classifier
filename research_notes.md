We are training a multilabel classifier to take an input chess game and output a set of label probabilities. We use Binary Cross Entropy with Logits as the loss function.
The model at present is a very simple convnet. It doesn't use embeddings to represent the board.
We then threshold the label probabilities (e.g. 0.5) to represent the chosen labels.
We generate a co-occurrence matrix to see how well the classifier is performing. We also calculate the Jaccard Index.
A strong/highlighted diagonal is the sign of a good classification result in the co-occurrence matrix.

2025-03-18
We currently only label with a small subset of the labels from the training dataset. I'm going to change the logic to save the values and output a few versions of the co-occurrence matrix with different threshholding. We could also explore PR curves or ROC curves, though I suspect the model is just too simple.

Generating some UMAP or other reduced dimensionality projections of the trained chess embeddings to understand how they work could also help.

2025-04-02
I've partially added coverage for the labeled opening positions, reviewing to see how far I got previously and pickup up with adding opening labels.
It appears that the last checkpoint file was trained with a version of the model which didn't include the openings (which changes the shape), so I'll do a training run first to generate a compatible checkpoint. It tries to resume from checkpoint_resume.pth, so I need to move it to checkpoint_resume_backup.pth.

