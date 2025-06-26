# Full Class Conditional Augmentation - Container Implementation

This document describes the implementation and fixes applied when running the full class conditional dataset augmentation on the container environment. This supplements the main documentation in [docs/full-class-conditional-augmentation.md](./full-class-conditional-augmentation.md).

## Container Environment

- **System**: Linux container with 377GB RAM (353GB available)
- **Dataset size**: 4,956,460 puzzles
- **Processing approach**: Single-threaded with progress monitoring

## Issues Encountered and Solutions

### 1. Progress Information Missing

**Issue**: When running `python create_full_dataset_cache.py --verbose`, progress information was not displayed despite tqdm being available and verbose mode being enabled.

**Root Cause**: Progress bars were only shown during chunk processing completion, but not during the initial chunk submission phase which could take significant time.

**Solution**: Added tqdm progress bars for chunk submission phase in `dataset.py`:
- Added progress bar for "Submitting chunks to workers" phase
- Added progress bar for "Processing chunks" phase  
- Added memory monitoring with psutil

### 2. BrokenProcessPool Errors

**Issue**: Despite having 377GB RAM available, worker processes were being terminated abruptly with `BrokenProcessPool` errors.

**Root Cause**: The issue was not memory-related but rather multiprocessing-related on this container environment. With 99,130 small chunks (chunk_size=50) being submitted to a single worker, the process queue became overwhelmed.

**Solutions Applied**:

1. **Increased chunk size**: From 50 to 500 FENs per chunk (reduced chunk count from 99,130 to 9,913)
2. **Implemented batched processing**: Limited pending futures to 100 at a time in low_memory mode
3. **Added single-threaded fallback**: Implemented `num_workers=0` mode for complete avoidance of multiprocessing

### 3. Single-Threaded Processing Implementation

**Final Solution**: Single-threaded processing (`num_workers=0`) completely bypasses multiprocessing issues.

**Key Changes**:
- Added `_sequential_batch_conversion()` method in `dataset.py`
- Modified parameter handling to properly respect `num_workers=0` (was previously falling back to `max(1, cpu_count()-1)`)
- Added memory monitoring and progress reporting for sequential processing
- Modified `create_full_dataset_cache.py` to force `num_workers=0`

## Performance Results

With single-threaded processing:
- **Processing rate**: ~20 chunks/second (~10,000 FENs/second)
- **Memory usage**: Only 7% of available 377GB RAM (23.3GB used)
- **Progress display**: Real-time tqdm progress bars showing percentage, ETA, and processing rate
- **Stability**: No process crashes or multiprocessing errors

## Code Changes

### dataset.py
1. Added tqdm progress bars for chunk submission phase
2. Added memory monitoring with psutil
3. Implemented `_sequential_batch_conversion()` method
4. Fixed parameter handling for `num_workers=0`
5. Added batched processing to limit concurrent futures

### create_full_dataset_cache.py  
1. Added `num_workers: 0` parameter to force single-threaded mode

## Usage

To run with full progress information and stability on the container:

```bash
python create_full_dataset_cache.py --verbose
```

This will:
- Use single-threaded processing (`num_workers=0`)
- Display detailed progress bars
- Monitor memory usage
- Process ~10K FENs per second
- Avoid all multiprocessing-related issues

## Memory Usage

The container environment has abundant memory (377GB total, 353GB available), so memory constraints are not an issue. The single-threaded approach uses only ~7% of available RAM while maintaining good performance.

## Lessons Learned

1. **High memory ≠ multiprocessing friendly**: Despite having 10x more RAM than smaller systems, this container environment had multiprocessing issues
2. **Single-threaded can be sufficient**: Processing 10K FENs/second single-threaded is adequate for this dataset size
3. **Progress monitoring is crucial**: For long-running operations, detailed progress information improves user experience
4. **Robust parameter handling**: Need to properly handle `num_workers=0` as a valid explicit value, not just falsy