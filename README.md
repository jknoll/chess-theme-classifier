Get started:
apt update && apt install -y python3-dev python3-pip python3-virtualenv git nano

wget https://database.lichess.org/lichess_db_puzzle.csv.zst
apt install -y zstd
unzstd lichess_db_puzzle.csv.zst