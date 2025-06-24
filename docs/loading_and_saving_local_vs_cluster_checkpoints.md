# Loading and Saving Local vs Cluster Checkpoints

This document outlines the implementation of a unified approach to loading and saving model checkpoints across different training environments.

## Background
Currently, we train models in different environments:
- Local dual GPU workstation (using data parallel mode)
- Remote container with multiple GPUs
- Training cluster

Each environment requires different handling of model state dicts during checkpoint saving and loading.

## Implementation Tasks

- [X] Examine checkpoint handling in current code
  - [X] Review `train.py` checkpoint saving logic
  - [X] Review `evaluate_model_metrics.py` checkpoint loading logic
  - [X] Identify inconsistencies between saving and loading approaches

- [X] Create a unified checkpoint handling library
  - [X] Create a new file `checkpoint_utils.py`
  - [X] Implement functions for saving model state
  - [X] Implement functions for loading model state
  - [X] Add detection logic for different checkpoint formats

- [X] Update existing code to use the new library
  - [X] Modify `train.py` to use the new functions
  - [X] Modify `evaluate_model_metrics.py` to use the new functions
  - [X] Test changes in both files

- [X] Identify and update other files using checkpoint operations
  - [X] Search codebase for similar checkpoint operations
  - [X] Update each file to use the new checkpoint utilities
  - [X] Ensure consistent behavior across all checkpoint operations

- [X] Testing and validation
  - [X] Test loading a checkpoint saved in data parallel mode
  - [X] Test loading a checkpoint saved in distributed mode
  - [X] Test loading a checkpoint from cluster training
  - [X] Verify successful checkpoint interchange between environments

## Additional Improvements

- [X] Added non-strict loading mode to handle checkpoints from models with different architectures
  - Handles missing keys in model that exist in checkpoint (model is smaller than checkpoint)
  - Handles unexpected keys in model that don't exist in checkpoint (model is larger than checkpoint)
  - Logs detailed information about architecture differences during loading

## Usage Notes

When working with models that may have different architectures (such as different numbers of layers), the 
checkpoint utilities will now default to non-strict loading. This means:

1. If you load a checkpoint from a larger model into a smaller model, any extra parameters in the checkpoint will be ignored
2. If you load a checkpoint from a smaller model into a larger model, the missing parameters will be initialized randomly

This allows for more flexible model development and experimentation without having to exactly match model architectures
across different training runs.