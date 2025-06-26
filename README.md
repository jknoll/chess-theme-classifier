

![CI](https://github.com/jknoll/chess-theme-classifier/actions/workflows/test.yml/badge.svg)

# Introduction

This project implements a deep convolutional neural network to perform multi-label classification on board positions sourced from the lichess puzzles dataset. Each board position is labeled with applicable themes (for example, _back rank mate_, _zugzwang_, _advanced pawn_, etc.) as well as openings, if relevant (for example, _Sicilian Defense_, _The English_, etc.) Instructions are included for training and evaluating the mode, along with a trained checkpoint.

The lichess puzzle database contains about 5M labeled boards as of 2025-06-24:
`4956460 processed_lichess_puzzle_files/lichess_db_puzzle.csv`

## Get Started
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
git clone git@github.com:jknoll/chess-theme-classifier.git
cd chess-theme-classifier
```

## Create Virtualenv
```bash
python -m venv .chess-theme-classifier
source .chess-theme-classifier/bin/activate
```

Note: on a non-clean system (i.e. one which already has other dependencies installed) this results in `python not found`, but `python3` is available. Then attempting the `venv create` line above with `python3` results in an error suggesting `apt install python3.10-venv`.

## Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Get Lichess Chess Puzzles Dataset

### Option 1: Download and Process Raw Dataset
```bash
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
sudo apt install -y zstd
unzstd lichess_db_puzzle.csv.zst
```

### Option 2: Download Pre-processed Dataset from S3 (Recommended)
The pre-processed dataset includes cached tensors and other derived files which significantly speed up training by avoiding redundant preprocessing.

You'll need to set up AWS credentials with access to the S3 bucket. You can do this in several ways:

1. Using environment variables:
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

2. Using the AWS CLI (if installed):
```bash
pip install awscli
aws configure
```

3. Creating a credentials file at `~/.aws/credentials`:
```
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key
```

#### Download the Dataset
Run the provided download script:
```bash
python download_dataset.py
```

This will download all processed dataset files to the `processed_lichess_puzzle_files` directory. You can specify a different output directory:
```bash
python download_dataset.py --output-dir custom_directory
```

Additional options:
```
--threads N     Use N threads for parallel downloads (default: 4)
--verify        Verify that all critical files were downloaded successfully
```

After downloading, the training scripts will automatically detect and use these pre-processed files.

## Verify Training
Test the training loop with a small test dataset
```bash
python train.py --local --test_mode
```

## Training Notes
### Distributed Training
To train in DistributedDataParallel mode on a multi-GPU system:

```bash 
torchrun --nproc_per_node=[NUM_GPUs] train.py
```

### Local Training (default 10 epochs)
To train on a single machine (will auto-detect if being run with distributed tools or not):
```bash 
python train.py
```

### Force Training Mode
You can force a specific training mode regardless of environment:
```bash
# Force local mode (even if run with torchrun)
python train.py --local

# Force distributed mode (will fail if no GPUs available)
python train.py --distributed
```

### Additional Training Arguments
```
--test_mode      Run with a smaller dataset for testing
--wandb          Enable Weights & Biases logging
--project        Weights & Biases project name (default: chess-theme-classifier)
--name           Weights & Biases run name
--checkpoint_steps  Number of steps between saving checkpoints (default: 50000)
```

The test mode dataset is not specially constructed in any way. It is merely the first _n_ lines of the full dataset. 

## Testing Performance
Generate a co-occurrence matrix for testing with:
```bash
$ python3 test.py
```

Optional parameters:
```
--num_samples Number of samples to test (default: 1000)
--threshold Prediction threshold for classification (default: 0.3)'
--checkpoint Checkpoint file to use for testing
```

## Tensorized Dataset
The original dataset is a lichess puzzle CSV file. The training script and dataset class will parse this file and generate a set of board tensors and other dataset cache files. For example, the list of all classes, that is, themes and openings found in the input dataset as separate cache files. If the CSV file is not found, these cache files are found by default in `./processed_chess_puzzle_files`. Then training will run with these as input. 

## Class Imbalance and Corrected Dataset
The dataset is class-imbalanced by default. There is a long-tail distribution of examples of particular openings (especially specialized branches of rarer openings) and of particular themes. We have generated 

The class balanced version is represented by the file with _conditional suffix: `lichess_db_puzzle_test.csv.tensors.pt_conditional`.

  This file contains the result of applying class-conditional augmentation to address class imbalance in the chess theme classification dataset. The augmentation process selectively applies
   horizontal flipping only to underrepresented theme combinations, as documented in the class_imbalance_work_breakdown.md file.

  The augmented indices are tracked in the file lichess_db_puzzle_test.csv.tensors.pt_conditional.augmented_indices.json.

  Here's a complete set of commands to run with the class-balanced dataset and weighted loss:

## Complete Loop with Corrected Dataset

### First, activate the virtual environment
```bash
source .chess-theme-classifier/bin/activate
```
### Generate the class conditional augmentation for the test dataset. This will create the class-balanced tensor cache.
```bash
python -c "from dataset import ChessPuzzleDataset; ChessPuzzleDataset('lichess_db_puzzle_test.csv', class_conditional_augmentation=True)"
```

### Run training with the class-balanced test dataset and weighted loss enabled
```bash
python train_locally_single_gpu.py --test_mode --weighted_loss
```

When running with both the class-balanced dataset and weighted loss, we see very unstable training. For example, Jaccard similarity will drop to zero and then spike up to very high values repeatedly.

### To view the co-occurrence matrices for the class-balanced dataset
```bash
python -c 'import json; import pprint; with open("lichess_db_puzzle_test.csv.cooccurrence.json", "r") as f: 
pprint.pprint(json.load(f))'
```

This sequence will first generate the conditional augmentation for the small test dataset, then run the training with both class balancing (through the
conditional augmentation) and cost-sensitive learning (via weighted loss), and finally display the co-occurrence data for analysis.

### Class-Imbalance-Considerate Metrics

 Micro Averaging


  - Calculation: Aggregates all true positives, false positives, and false negatives across all classes before calculating metrics
  - Emphasis: Gives equal weight to each sample-class pair, favoring performance on common themes
  - When to use: Best when you want to assess overall effectiveness across all predictions
  - Example: If your classifier is great at detecting common themes like "mate" but struggles with rare ones, micro metrics will look good

  Macro Averaging

  - Calculation: Calculates metrics for each class independently, then takes the unweighted average
  - Emphasis: Each chess theme contributes equally regardless of frequency
  - When to use: When performance on rare themes is as important as common ones
  - Example: Lower macro than micro scores indicate your model performs worse on rare chess themes

  Weighted Averaging

  - Calculation: Takes a weighted average of per-class metrics, with weights proportional to class frequency
  - Emphasis: Balances between micro and macro, giving more influence to common themes
  - When to use: When you want a balanced view that still reflects dataset distribution
  - Example: Similar weighted and micro scores but lower macro scores suggest your model performs well overall but struggles with some rare themes

  These averages apply to precision (correct predictions/total predictions), recall (correct predictions/actual positives), and F1 (harmonic mean
  of precision and recall). In your multi-label chess theme context, they help evaluate how well your model identifies all relevant themes for each 
  position.

### train.py vs. train-isc.py
There are currently two separate scripts for training locally versus on the strong Compute ISC. We have undertaken to deduplicate them, and currently `train.py` can be referenced in `chessVision.isc`. Training completes successfully. The train-isc.py script should be considered deprecated. 

### Test Automation
See ['tests/README.md']('./tests/README.md') for details.

Tests inside /tests run on every push and pull request via github actions, as defined in ['.github/workflows/test.yml']('.github/workflows/test.yml') There are some other tests located in the project root directory, which are preserved for historical purposes. Only those tests within `/tests` should be considered maintained. 

```bash
python -m pytest /tests
```

### Model Evaluation

We have several scripts for model evaluation:

#### evaluate_model_metrics.py (Recommended)

This will calculate per class and global adaptive thresholds and save related CSVs in `analysis/f1`. 

```bash
python evaluate_model_metrics.py
```

You can follow Up when this run is complete and generate precision-recall curves which will be output in `analysis/pr-curves`
```bash
python evaluate_model_metrics_pr_curves.py
```


#### evaluate_model_classification.py

This is the primary evaluation script with improved adaptive thresholding, optimized performance, and better token efficiency.

```bash
# With adaptive thresholding (default)
python evaluate_model_classification.py --num_samples=100

# With fixed threshold
python evaluate_model_classification.py --num_samples=100 --threshold=0.3

# With detailed verbose output
python evaluate_model_classification.py --num_samples=50 --verbose

# Minimize output for token efficiency
python evaluate_model_classification.py --num_samples=100 --quiet

# Use cached tensor files directly instead of test CSV
python evaluate_model_classification.py --use_cache

# With specific checkpoint
python evaluate_model_classification.py --checkpoint=checkpoints/my_checkpoint.pth
```

#### evaluate_model_fixed.py

This script properly maps between training and test dataset indices and supports adaptive thresholding.

```bash
# With adaptive thresholding
python evaluate_model_fixed.py --num_samples=50

# With fixed threshold
python evaluate_model_fixed.py --num_samples=50 --threshold=0.3
```

#### evaluate_model_simple.py

A simplified evaluation script focused only on key chess themes.

```bash
python evaluate_model_simple.py --num_samples=20 --threshold=0.3
```

#### evaluate_model_cache.py

This script uses cached tensor files directly, bypassing the test CSV completely.

```bash
python evaluate_model_cache.py --num_samples=1000
```

See [docs/model_evaluation.md](docs/model_evaluation.md) for detailed information about each evaluation script.
