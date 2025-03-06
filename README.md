Get started:
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano

wget https://database.lichess.org/lichess_db_puzzle.csv.zst
apt install -y zstd
unzstd lichess_db_puzzle.csv.zst

For running in DistributedDataParallel mode, 
`torchrun --nproc_per_node=NUM_GPUS train.py`

For example: `torchrun --nproc_per_node=4 train.py`