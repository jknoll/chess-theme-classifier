# Downloading the Processed Lichess Puzzle Dataset from S3

The processed Lichess puzzle files directory should contain a set of cache files which are generated and derived from the lichess_db_puzzle.csv file. The master storage location for these files is an Amazon S3 bucket with the information below.

## S3 Bucket Information
- **Bucket Name**: com-justinknoll-lichess-puzzles-dataset-processed
- **Region**: us-east-1

## Using the Download Script

We've created a script `download_dataset.py` in the project root that securely handles AWS credentials and downloads all the necessary files from the S3 bucket.

### Prerequisites

1. Install the required Python packages:
   ```bash
   pip install boto3
   ```

2. Configure AWS credentials with access to the S3 bucket. You can do this in several ways:

   a. Using environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID="your_access_key"
   export AWS_SECRET_ACCESS_KEY="your_secret_key"
   ```

   b. Using the AWS CLI (if installed):
   ```bash
   pip install awscli
   aws configure
   ```

   c. Creating a credentials file at `~/.aws/credentials`:
   ```
   [default]
   aws_access_key_id = your_access_key
   aws_secret_access_key = your_secret_key
   ```

### Running the Script

Basic usage:
```bash
python download_dataset.py
```

This will download all files from the S3 bucket to the `processed_lichess_puzzle_files` directory.

### Additional Options

- Specify a custom output directory:
  ```bash
  python download_dataset.py --output-dir custom_directory
  ```

- Control the number of download threads:
  ```bash
  python download_dataset.py --threads 8
  ```

- Verify critical files after download:
  ```bash
  python download_dataset.py --verify
  ```

### Verifying the Download

After downloading, you should have these critical files (among others):
- lichess_db_puzzle.csv.themes.json
- lichess_db_puzzle.csv.openings.json
- lichess_db_puzzle.csv.tensors.pt_conditional
- lichess_db_puzzle.csv.cooccurrence.json
- lichess_db_puzzle.csv.class_weights.pt

### Security Considerations

The download script:
- Does not hardcode any credentials
- Uses secure methods to fetch credentials from standard AWS locations
- Supports environment variables for CI/CD workflows
- Uses HTTPS connections to S3 by default

### Automatic Dataset Detection

The training scripts will automatically detect and use these pre-processed files, avoiding the need to regenerate them from the raw CSV file.