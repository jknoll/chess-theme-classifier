1. [x] Verify that the necessary functionality is present in train.py to train locally on a single GPU
2.  [x] Delete train_locally_single_gpu.py to save tokens on unecessary updates.
3. [x] Make tensorboard launch with --logdir ./logs, so that one can compare across runs.
4. [x] Add tqdm progress to "Analyzing theme co-occurrence patterns..."
5. [x] Add tqdm progress for "Processing 4,956,459 FENs in 49565 chunks using 1 workers (low_memory=True)"
6. [x] Generate the complete dataset and caches, import them to S3 and export them to SC as a dataset. This will eliminate the heavy preprocessing being run redundantly on each node in the training cluster.