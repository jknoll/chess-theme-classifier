## Get Started
```bash
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano
git clone git@github.com:jknoll/chess-theme-classifier.git
cd chess-theme-classifier
```

## Create Virtualenv
```bash
python -m venv create .chess-theme-classifier
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

More notes to come.