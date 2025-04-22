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

## Training Notes
For running in DistributedDataParallel mode

```bash 
torchrun --nproc_per_node=[NUM_GPUs] train.py
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