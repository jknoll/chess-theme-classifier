# Uploading the Processed Lichess Puzzle Dataset to S3

Refer to docs/dataset-download-to-s3 for the equivalent download operation.
Create a upload-dataset-to-s3.py script, similar to download-dataset-from-s3.py which 

When a new file has been generated locally, we will need to upload it to S3. Currently, these files need to be uploaded:
chess-theme-classifier/processed_lichess_puzzle_files/lichess_db_puzzle.csv.tensors.pt
chess-theme-classifier/processed_lichess_puzzle_files/lichess_db_puzzle.csv.tensors.pt_conditional_full
chess-theme-classifier/processed_lichess_puzzle_files/lichess_db_puzzle.csv.tensors.pt_conditional_full.augmented_indices.json

For now, the upload script can upload these files. It should output to the user the set up files being uploaded and encourage the user to check and modify the script if other files need to be uploaded. Later we can create smarter logic for uploading just the diff.

The processed Lichess puzzle files directory should contain a set of cache files which are generated and derived from the lichess_db_puzzle.csv file. The master storage location for these files is an Amazon S3 bucket with the information below.

## S3 Bucket Information
- **Bucket Name**: com-justinknoll-lichess-puzzles-dataset-processed
- **Region**: us-east-1

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