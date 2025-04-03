# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Training (distributed): `torchrun --nproc_per_node=NUM_GPUS train.py`
- Training (single GPU): `python train_locally_single_gpu.py`
- Testing: `python test.py` (runs on 20,000 samples by default)
- Running a specific test: Modify test sample size with `--num_samples` flag

## Style Guidelines
- Formatting: 4-space indentation
- Imports: stdlib → third-party → local modules
- Naming: snake_case for variables/functions, CamelCase for classes
- Documentation: Use docstrings for functions and classes
- Types: Add type hints for function parameters and return values
- Error handling: Use try/except blocks for expected exceptions

## Project Structure
This is a PyTorch-based chess puzzle classifier that predicts themes from chess board positions using a CNN architecture for multi-label classification.