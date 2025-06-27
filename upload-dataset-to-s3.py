#!/usr/bin/env python3
"""
Upload processed Lichess puzzle dataset files to Amazon S3 bucket.

This script securely uploads dataset files from the processed_lichess_puzzle_files 
directory to the S3 bucket for backup and sharing.
"""

import os
import sys
import boto3
import logging
import argparse
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# S3 bucket information
S3_BUCKET = "com-justinknoll-lichess-puzzles-dataset-processed"
S3_REGION = "us-east-1"

# Local directory containing the dataset files
DEFAULT_INPUT_DIR = "processed_lichess_puzzle_files"

# Lock for thread-safe console output
console_lock = threading.Lock()

# Files that are typically uploaded (user should verify and modify as needed)
TYPICAL_UPLOAD_FILES = [
    "lichess_db_puzzle.csv.tensors.pt",
    "lichess_db_puzzle.csv.tensors.pt_conditional_full", 
    "lichess_db_puzzle.csv.tensors.pt_conditional_full.augmented_indices.json",
    "lichess_db_puzzle.csv.themes.json",
    "lichess_db_puzzle.csv.openings.json",
    "lichess_db_puzzle.csv.tensors.pt_conditional",
    "lichess_db_puzzle.csv.cooccurrence.json",
    "lichess_db_puzzle.csv.class_weights.pt"
]

def check_aws_credentials():
    """Verify AWS credentials are properly configured."""
    # Check for environment variables
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        logger.info("Using AWS credentials from environment variables")
        return True
    
    # Check for AWS credentials file
    home_dir = str(Path.home())
    aws_credentials_file = os.path.join(home_dir, ".aws", "credentials")
    aws_config_file = os.path.join(home_dir, ".aws", "config")
    
    if os.path.exists(aws_credentials_file):
        logger.info(f"Found AWS credentials file at {aws_credentials_file}")
        return True
    
    if os.path.exists(aws_config_file):
        logger.info(f"Found AWS config file at {aws_config_file}")
        return True
    
    return False

def initialize_s3_client():
    """Initialize and return an S3 client."""
    try:
        s3_client = boto3.client('s3', region_name=S3_REGION)
        logger.info(f"Initialized S3 client for region {S3_REGION}")
        return s3_client
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        sys.exit(1)

def get_file_md5(file_path):
    """Calculate MD5 hash of a file for comparison."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_s3_object_etag(s3_client, object_key):
    """Get the ETag of an S3 object if it exists."""
    try:
        response = s3_client.head_object(Bucket=S3_BUCKET, Key=object_key)
        # Remove quotes from ETag if present
        etag = response['ETag'].strip('"')
        return etag
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return None  # Object doesn't exist
        else:
            raise e

def upload_file(s3_client, local_file_path, object_key, force_upload=False):
    """Upload a single file to S3 bucket."""
    try:
        with console_lock:
            logger.info(f"Preparing to upload {object_key}")
        
        # Check file size
        file_size = os.path.getsize(local_file_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Check if file already exists in S3 and compare
        if not force_upload:
            try:
                s3_etag = get_s3_object_etag(s3_client, object_key)
                if s3_etag:
                    local_md5 = get_file_md5(local_file_path)
                    if s3_etag == local_md5:
                        with console_lock:
                            logger.info(f"File {object_key} already exists with same content, skipping upload")
                        return True
                    else:
                        with console_lock:
                            logger.info(f"File {object_key} exists but content differs, uploading new version")
            except Exception as e:
                with console_lock:
                    logger.warning(f"Could not check existing file {object_key}: {e}")
        
        # Upload the file
        start_time = time.time()
        
        with console_lock:
            logger.info(f"Uploading {object_key} ({file_size_mb:.2f} MB)")
        
        # Use multipart upload for large files (>100MB)
        if file_size > 100 * 1024 * 1024:
            s3_client.upload_file(
                local_file_path, 
                S3_BUCKET, 
                object_key,
                Config=boto3.s3.transfer.TransferConfig(
                    multipart_threshold=1024 * 25,  # 25MB
                    max_concurrency=10,
                    multipart_chunksize=1024 * 25,
                    use_threads=True
                )
            )
        else:
            s3_client.upload_file(local_file_path, S3_BUCKET, object_key)
        
        upload_time = time.time() - start_time
        upload_speed = file_size_mb / upload_time if upload_time > 0 else 0
        
        with console_lock:
            logger.info(f"Uploaded {object_key} in {upload_time:.2f} seconds ({upload_speed:.2f} MB/s)")
        return True
        
    except ClientError as e:
        with console_lock:
            logger.error(f"Failed to upload {object_key}: {e}")
        return False
    except Exception as e:
        with console_lock:
            logger.error(f"Unexpected error uploading {object_key}: {e}")
        return False

def find_files_to_upload(input_dir, file_patterns=None):
    """Find files in the input directory that should be uploaded."""
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        return []
    
    files_to_upload = []
    
    if file_patterns:
        # Upload specific files/patterns
        for pattern in file_patterns:
            file_path = os.path.join(input_dir, pattern)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                files_to_upload.append(pattern)
            else:
                logger.warning(f"File not found: {file_path}")
    else:
        # Default: upload all files in the directory
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                files_to_upload.append(filename)
    
    return files_to_upload

def upload_all_files(input_dir, file_patterns=None, threads=4, force_upload=False):
    """Upload files from the input directory to the S3 bucket."""
    if not check_aws_credentials():
        logger.error("AWS credentials not found. Please configure your AWS credentials.")
        logger.info("You can set up AWS credentials by:")
        logger.info("1. Setting AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        logger.info("2. Running 'aws configure' if you have the AWS CLI installed") 
        logger.info("3. Creating credentials file at ~/.aws/credentials")
        sys.exit(1)
    
    s3_client = initialize_s3_client()
    files_to_upload = find_files_to_upload(input_dir, file_patterns)
    
    if not files_to_upload:
        logger.error("No files found to upload")
        return False
    
    total_files = len(files_to_upload)
    total_size = sum(os.path.getsize(os.path.join(input_dir, f)) for f in files_to_upload)
    total_size_gb = total_size / (1024**3)
    
    logger.info(f"Found {total_files} files to upload (total size: {total_size_gb:.2f} GB)")
    logger.info("Files to upload:")
    for filename in files_to_upload:
        file_path = os.path.join(input_dir, filename)
        file_size_mb = os.path.getsize(file_path) / (1024**2)
        logger.info(f"  - {filename} ({file_size_mb:.2f} MB)")
    
    # Upload files in parallel
    successful_uploads = 0
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                upload_file, 
                s3_client, 
                os.path.join(input_dir, filename), 
                filename,
                force_upload
            ) 
            for filename in files_to_upload
        ]
        
        for future in futures:
            if future.result():
                successful_uploads += 1
    
    logger.info(f"Upload complete: {successful_uploads}/{total_files} files uploaded successfully")
    
    if successful_uploads == total_files:
        logger.info("All files were uploaded successfully!")
    else:
        logger.warning(f"Some files failed to upload ({total_files - successful_uploads} failures)")
    
    return successful_uploads == total_files

def main():
    parser = argparse.ArgumentParser(
        description="Upload Lichess puzzle dataset files to S3",
        epilog="""
IMPORTANT: Please review and modify this script if you need to upload different files.
The default files being uploaded are:
  - lichess_db_puzzle.csv.tensors.pt
  - lichess_db_puzzle.csv.tensors.pt_conditional_full
  - lichess_db_puzzle.csv.tensors.pt_conditional_full.augmented_indices.json
  - (and other typical dataset files)

Use --files to specify exact files to upload, or --all to upload everything in the directory.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir", 
        type=str, 
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing files to upload (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--files", 
        nargs="+",
        help="Specific files to upload (relative to input-dir)"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Upload all files in the input directory"
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=2,
        help="Number of upload threads to use (default: 2, keep low to avoid rate limits)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force upload even if files already exist with same content"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    
    args = parser.parse_args()
    
    # Determine which files to upload
    if args.all:
        file_patterns = None  # Upload all files
    elif args.files:
        file_patterns = args.files
    else:
        # Default: upload typical dataset files that exist
        file_patterns = TYPICAL_UPLOAD_FILES
        logger.info("Using default file list. Use --files to specify different files or --all for everything.")
    
    logger.info(f"Starting upload of processed Lichess puzzle dataset")
    logger.info(f"S3 bucket: {S3_BUCKET}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Using {args.threads} upload threads")
    
    if args.dry_run:
        logger.info("DRY RUN: No files will actually be uploaded")
        files_to_upload = find_files_to_upload(args.input_dir, file_patterns)
        if files_to_upload:
            logger.info("Would upload these files:")
            for filename in files_to_upload:
                file_path = os.path.join(args.input_dir, filename)
                file_size_mb = os.path.getsize(file_path) / (1024**2)
                logger.info(f"  - {filename} ({file_size_mb:.2f} MB)")
        else:
            logger.info("No files found to upload")
        return
    
    success = upload_all_files(args.input_dir, file_patterns, args.threads, args.force)
    
    if success:
        logger.info("✅ All uploads completed successfully!")
    else:
        logger.error("❌ Some uploads failed. Check the logs above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()