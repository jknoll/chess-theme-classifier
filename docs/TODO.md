1. [x] Verify that the necessary functionality is present in train.py to train locally on a single GPU
2.  [x] Delete train_locally_single_gpu.py to save tokens on unecessary updates.
3. [x] Make tensorboard launch with --logdir ./logs, so that one can compare across runs.
4. [x] Add tqdm progress to "Analyzing theme co-occurrence patterns..."
5. [x] Add tqdm progress for "Processing 4,956,459 FENs in 49565 chunks using 1 workers (low_memory=True)"
6. [x] Generate the complete dataset and caches, import them to S3 and export them to SC as a dataset. This will eliminate the heavy preprocessing being run redundantly on each node in the training cluster.
7. [x] Port metrics from train.py to train-isc.py.
8. [x] Run train-isc.py in interruptible mode with GPUs=6,
9. [x] Create a tests folder with some simple pytest tests.
10. [ ] Observe tensorboard output to see if stats are flowing into all reports when run for longer than a cycle job.
11. [x] Retrieve a checkpoint and run the co-occurrence analysis on it.
12. [ ] It appears that we have a venv inside a venv. Running commands when in the outer venv fails unless we activate an inner one? Fix.
13. [ ] Add instructions to README.md for installing S3 tool and downloading the processed dataset from S3 into the local filesystem.