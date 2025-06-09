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
pip install -r requirements.txt
```

## Get Lichess Chess Puzzles Dataset
```bash
wget https://database.lichess.org/lichess_db_puzzle.csv.zst
apt install -y zstd
unzstd lichess_db_puzzle.csv.zst
```

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

```bash
python -m pytest