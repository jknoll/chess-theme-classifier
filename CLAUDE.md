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
- Custom save frequency: `python train.py --save-steps 500` (checkpoints saved every N steps)
- Legacy single GPU training: `python train_locally_single_gpu.py` (deprecated, use `--single_gpu` flag instead)

### Model Evaluation
- Standard evaluation (recommended): `python evaluate_model_classification.py` (with adaptive thresholding)
- Evaluation with specific sample size: `python evaluate_model_classification.py --num_samples 1000`
- Evaluation with fixed threshold: `python evaluate_model_classification.py --threshold 0.3`
- Verbose evaluation output: `python evaluate_model_classification.py --verbose`
- Minimized output for token efficiency: `python evaluate_model_classification.py --quiet`
- Using cached tensor files: `python evaluate_model_classification.py --use_cache`
- Alternative evaluation method: `python evaluate_model_fixed.py` (with adaptive thresholding)
- Simple evaluation focused on key themes: `python evaluate_model_simple.py`
- Direct cached tensor evaluation: `python evaluate_model_cache.py`

### Testing
- Unit testing: `python -m pytest test_model_unittest.py -v`

## Dependencies
- Dependencies are managed via requirements.txt
- When adding new dependencies, add them to requirements.txt
- The project uses a Python virtual environment located at `.chess-theme-classifier/`
- Always activate the virtual environment before running any commands: `source .chess-theme-classifier/bin/activate`
- If new dependencies are needed, add them to requirements.txt and run then run `pip install -r requirements.txt`
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

## Progress Reporting
- 
When performing operations over a large dataset, whether testing, training, evaluating, or generating dataset derivatives like cache tensor files, provide `tqdm` progress and estimated time of completion output for the user. Be sure to put this `tqdm` progress output behind a verbose flag as described in the verbose output section below, so that it can be omitted from output when you call scripts, where you should generally prefer not to pass `--verbose` for token efficiency. 

## Verbose output
- Add a `--verbose` flag option when you create new scripts. Be sure to put particularly long output behind the verbose flag. `tqdm` output is one example that should be put behind a verbose flag, which you do not invoke unnecessarily for purposes of token efficiency. 

## Emojis
Do NOT use Emojis in codeDo or documentation, or especially in logs where they may not be decoded properaly by terminals.