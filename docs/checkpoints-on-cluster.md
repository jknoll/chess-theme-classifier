# Checkpoint Handling on ISC Training Cluster

## Issue Description

When we train using `train.py` on a remote ISC training cluster, the logs show attempts to write checkpoints to:

```bash
2025-06-29 09:30:17,120 - checkpoint_utils - INFO - tr /mnt/checkpoints/checkpoint_20250629-093015_2000.pth
```

But when training on the cluster, this directory does not seem to be writable or is not reachable from the controlling container.

## Solution Implemented ✅

- [x] **Implemented AtomicDirectory for cluster checkpoint saving**
- [x] **Added comprehensive environment variable validation and logging**
- [x] **Added write permission testing for checkpoint directories**
- [x] **Enhanced ISC mode detection with detailed logging**
- [x] **Added graceful error handling with informative error messages**
- [x] **Fixed AtomicDirectory initialization timing to work with distributed setup**
- [x] **Implemented proper AtomicDirectory usage following train_chessVision.py reference exactly**

## Implementation Details

### Key Changes Made:

1. **AtomicDirectory Integration**: 
   - Imported `AtomicDirectory` and `atomic_torch_save` from `cycling_utils`
   - Used for safe, atomic checkpoint writing in cluster mode
   - Creates proper symlinks to latest checkpoints
   - **Uses `atomic_torch_save` instead of `torch.save()` in cluster mode** (critical difference from regular checkpoint saving)

2. **Environment Variable Validation**:
   - Comprehensive logging of `LOSSY_ARTIFACT_PATH` detection
   - Validation and logging of `CHECKPOINT_ARTIFACT_PATH` environment variable (matching train_chessVision.py reference)
   - Clear error messages when required variables are missing

3. **Directory Validation**:
   - Tests directory existence and accessibility
   - Verifies write permissions by creating test files
   - Validates directory structure before attempting saves

4. **Enhanced Logging**:
   - Detailed checkpoint directory validation logs
   - Clear indication of cluster vs local mode
   - Informative error messages with troubleshooting guidance
   - **Comprehensive checkpoint write logging**: Clear [ATTEMPT]/[SUCCESS]/[COMPLETE] messages for each checkpoint operation
   - **Step-by-step progress**: Logs for both timestamped and resume checkpoint writes
   - **Operation summary**: Complete summary with file paths and mode information

5. **Early Validation Strategy**:
   - **Two-stage validation**: Early validation before distributed setup + re-validation after
   - **Immediate feedback**: Catches checkpoint issues before training starts (no need to wait for first checkpoint save)
   - **Fail-fast approach**: Program exits immediately if checkpoint directory issues detected
   - **Fixed timing issue**: AtomicDirectory creation is deferred until after distributed process group initialization
   - **Proper collective operations**: All ranks participate in prepare_checkpoint_directory() and symlink_latest() calls as required by cycling_utils
   - **Batch-based triggers**: Uses (batch + 1) % save_steps == 0 like reference for proper distributed synchronization
   - **Reference data structure**: Uses "model"/"optimizer" keys matching train_chessVision.py exactly

### Function Added:

```python
def validate_and_setup_checkpoint_directory(isc_mode, is_master=True, defer_atomic_directory=False):
    """
    Validate and setup checkpoint directory with comprehensive logging and sanity checks.
    
    Args:
        defer_atomic_directory (bool): If True, skip AtomicDirectory initialization (for early validation)
    
    Returns: (checkpoint_dir, atomic_saver) where atomic_saver is None for local mode or deferred
    """
```

### Checkpoint Saving Logic (Following train_chessVision.py Reference):

- **Cluster Mode**: 
  - All ranks call `prepare_checkpoint_directory()` (collective operation)
  - Only master rank calls `atomic_torch_save()` to save checkpoint.pt
  - All ranks call `symlink_latest()` (collective operation)
  - Uses "model"/"optimizer" keys matching reference format
- **Local Mode**: Uses standard `save_checkpoint()` utility with `torch.save()` to `./checkpoints/`
- **Trigger**: Uses `(batch + 1) % save_steps == 0` for proper distributed synchronization
- **Key Fix**: All ranks participate in collective operations simultaneously

## Reference Implementation

Based on the example from: 
https://github.com/StrongResearch/chess-hackathon/blob/5c529911bccdc3c2038dc2b41f26a80a701b0ff5/models/chessVision/train_chessVision.py

## Testing

The implementation includes comprehensive validation that will:
- Exit with clear error messages if required environment variables are missing
- Exit with specific errors if directories don't exist or aren't writable
- Log detailed information about checkpoint directory setup for debugging

## Validation Timing

**Early Detection (Before Training Starts)**:
1. **Stage 1**: Immediate validation after argument parsing and ISC mode detection
2. **Stage 2**: Re-validation after distributed setup when `args.is_master` is properly set

**Benefits**:
- **No waiting**: Checkpoint issues are detected within seconds of starting `train.py`
- **Time saving**: No need to wait for `checkpoint_steps` iterations to test checkpoint writing
- **Clear feedback**: Detailed validation logs show exactly what's working or failing
- **Early exit**: Program terminates immediately if checkpoint directory is not usable

**Example output**:
```
🔧 EARLY CHECKPOINT DIRECTORY VALIDATION
============================================================
🔍 Checkpoint Directory Validation:
   ISC Mode: True
   Is Master Process: True
   LOSSY_ARTIFACT_PATH: /path/to/logs
✅ ISC cluster mode detected
   Checking OUTPUT_PATH environment variable...
   OUTPUT_PATH: /path/to/checkpoints
✅ OUTPUT_PATH found: /path/to/checkpoints
✅ Directory exists: /path/to/checkpoints
✅ Directory is writable: /path/to/checkpoints
✅ AtomicDirectory initialized successfully
============================================================
```