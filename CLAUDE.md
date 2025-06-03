# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Training (distributed): `torchrun --nproc_per_node=NUM_GPUS train.py`
- Training with W&B: `torchrun --nproc_per_node=NUM_GPUS train.py --wandb`
- Training (single GPU): `python train.py --single_gpu`
- Low memory mode: `python train.py --low_memory` (reduces memory usage, enabled by default)
- Using Adam optimizer: `python train.py --optimizer adam`
- Custom batch size: `python train.py --batch_size 8`
- Custom epochs: `python train.py --epochs 5`
- Legacy single GPU training: `python train_locally_single_gpu.py` (deprecated, use `--single_gpu` flag instead)
- Testing: `python test.py` (runs on 20,000 samples by default)
- Running a specific test: Modify test sample size with `--num_samples` flag
- Unit testing: `python -m pytest test_model_unittest.py -v`

## Dependencies
- Dependencies are managed via requirements.txt
- When adding new dependencies, add them to requirements.txt
- The project uses a Python virtual environment located at `.chess-theme-classifier/`
- Always activate the virtual environment before running any commands: `source .chess-theme-classifier/bin/activate`
- If new dependencies are needed, add them to requirements.txt and run `pip install -r requirements.txt`
- Required third-party packages include: PyTorch, torchvision, pandas, numpy, scikit-learn, matplotlib, etc. (see requirements.txt)
- Verify that installation succeeds before committing changes

## Security and Credentials
- NEVER store API keys, passwords, or other credentials in source code
- Use environment variables for sensitive information:
  - For Weights & Biases: Set WANDB_API_KEY or use `wandb login` command
  - Store credentials in .env files (add to .gitignore)
  - Document required environment variables in README.md
- When writing code that requires credentials, always use environment variables or config files
- For dev environments, add instructions for setting up credentials

## Style Guidelines
- Formatting: 4-space indentation
- Imports: stdlib → third-party → local modules
- Naming: snake_case for variables/functions, CamelCase for classes
- Documentation: Use docstrings for functions and classes
- Types: Add type hints for function parameters and return values
- Error handling: Use try/except blocks for expected exceptions

## Project Structure
This is a PyTorch-based chess puzzle classifier that predicts themes from chess board positions using a CNN architecture for multi-label classification.

## Dataset Files
- Full dataset: `lichess_db_puzzle.csv` (main dataset for training)
- Test dataset: `lichess_db_puzzle_test.csv` (smaller dataset for testing)
- Small sample: `lichess_db_puzzle_small.csv` (tiny dataset for quick tests)

## Dataset Optimization
- The ChessPuzzleDataset class in dataset.py uses tensor caching for improved performance
- First access creates a cache file (.tensors.pt) that's used in subsequent runs
- Cache validation checks the CSV modification time to ensure data consistency
- Use the --optimized flag with test.py and other scripts to ensure all optimizations are enabled
- Typical cache-enabled speedups are 2-3x for dataset access

## Testing
- Unit tests in test_model_unittest.py verify model functionality
- Test architecture includes:
  - Model initialization tests
  - Forward pass tests (with and without debug mode)
  - Backward pass tests
  - Mini training loop tests
  - Parameter gradient flow tests
- These tests use a small subset of data to quickly verify code changes
- Run regularly to catch model architecture bugs or training regressions

## Logging and Monitoring
- TensorBoard is used for basic metrics logging
- Weights & Biases (wandb) is available for more advanced experiment tracking
- To use wandb: set WANDB_API_KEY environment variable or run `wandb login`
- Enable wandb with `--wandb` flag when running train.py

## Documenting Progress through Milestones in Markdown Documentation
- When implementing changes listed in Markdown Documentation (in /docs) do not strikethrough completed work and subtasks; instead use [ ] and [x] before each item.