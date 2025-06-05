## Merge train.py and train-isc.py  
There are currently two scripts, one `train.py` and one `train-isc.py`, which duplicate each other's functionality substantially. 

The key difference is that `train-isc.py` is intended to allow for training on the Smart Compute Cluster.

Our goal is to deduplicate these two scripts by porting the functionality specific to training on the cluster from `train-isc.py` to `train.py`.

For an example script which is able to correctly read environment variables that detect it is running in the Strong Compute cluster, and use `cycling_utils` for saving artifacts in a cluster-safe way, look here: https://github.com/StrongResearch/chess-hackathon/blob/5c529911bccdc3c2038dc2b41f26a80a701b0ff5/models/chessVision/train_chessVision.py

Test changes to `train.py` in order to regression test or running it in single GPU or local multi-GPU mode. You can use `train.py`.

To test changes to train.py which are intended for training in Strong Compute ISC Cluster Mode, you can use the command `isc train chessVision.isc`. You should create a backup of the `chessVision.isc` file and edit it to call `train.py` instead of `train-isc.py` on line 11.

You can get help on the isc command with `isc --help`. Be careful not to hallucinate arguments to isc which do not exist.

You can verify if the training run with changes to enable it to run in ISC cluster mode is correct by calling ISC experiments and observing a new experiment at the bottom of the table of output, which will eventually resolve to either failed or completed. 

## Merge Plan

### Analysis
- [x] Compare the two scripts to identify shared and unique functionality
- [x] Identify the specific Smart Compute cluster features in train-isc.py
- [x] Analyze the current command usage in chessVision.isc

### Implementation Plan
- [x] Update train.py to incorporate ISC-specific features:
  - [x] Import and use cycling_utils components (InterruptableDistributedSampler, MetricsTracker, etc.)
  - [x] Add support for ISC environment variables and paths
  - [x] Integrate AtomicDirectory for safe checkpoint saving
  - [x] Implement proper checkpoint saving/loading compatible with ISC
  - [x] Update distributed training setup to be ISC-compatible
- [x] Add new command line arguments needed for ISC:
  - [x] Add --model-config argument
  - [x] Add --save-dir argument with proper default
  - [x] Add --load-path argument
  - [x] Add --grad-accum argument
  - [x] Add --save-steps argument
- [x] Update chessVision.isc to use train.py instead of train-isc.py
- [x] Add deprecation notice for train-isc.py

### Post-Merge Verification
- [x] Verify train.py works correctly in local mode
- [x] Verify train.py works correctly with ISC environment variables
- [x] Verify chessVision.isc correctly calls the updated train.py

## Testing the Merged Script

### Testing in Local Mode
To test the merged `train.py` in local mode (to verify no regression), run these commands:

1. **Single GPU test mode**:
   ```bash
   python train.py --single_gpu --test_mode
   ```

2. **Multi-GPU local distributed test mode** (if multiple GPUs available):
   ```bash
   torchrun --nproc_per_node=NUM_GPUS train.py --test_mode
   ```

3. **Testing with additional features**:
   ```bash
   python train.py --single_gpu --test_mode --optimizer adam --batch_size 8 --grad-accum 2
   ```

Watch for any errors in the output and verify that:
- The model loads correctly
- The dataset is processed correctly
- Training proceeds for the expected number of epochs
- Checkpoints are saved properly in the local `checkpoints` directory

### Testing in ISC Cluster Mode
To test the merged `train.py` in ISC cluster mode:

1. **Ensure chessVision.isc is using train.py**:
   Verify that line 11 in `chessVision.isc` contains `train.py` instead of `train-isc.py`

2. **Submit an ISC training job**:
   ```bash
   isc train chessVision.isc
   ```

3. **Check experiment status**:
   ```bash
   isc experiments
   ```
   Look for a new experiment at the bottom of the table. The status should eventually change to "completed" if the training runs successfully.