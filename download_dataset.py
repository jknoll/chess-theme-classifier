#!/usr/bin/env python3
"""
Download processed Lichess puzzle dataset from Amazon S3 bucket.

This script securely downloads the processed dataset files from the S3 bucket
and places them in the processed_lichess_puzzle_files directory.
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

# Local directory for storing the dataset
DEFAULT_OUTPUT_DIR = "processed_lichess_puzzle_files"

# Lock for thread-safe console output
console_lock = threading.Lock()

# Critical files that should be present after download
CRITICAL_FILES = [
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

def list_bucket_objects(s3_client):
    """List all objects in the S3 bucket."""
    try:
        logger.info(f"Listing objects in bucket {S3_BUCKET}...")
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET)
        
        if 'Contents' not in response:
            logger.error(f"No objects found in bucket {S3_BUCKET}")
            return []
        
        return response['Contents']
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            logger.error(f"Bucket {S3_BUCKET} does not exist")
        elif e.response['Error']['Code'] == 'AccessDenied':
            logger.error(f"Access denied to bucket {S3_BUCKET}. Check your credentials.")
        else:
            logger.error(f"Failed to list objects in bucket {S3_BUCKET}: {e}")
        return []

def download_file(s3_client, object_key, output_dir, total_objects):
    """Download a single file from S3 bucket."""
    local_file_path = os.path.join(output_dir, object_key)
    
    # Create any necessary subdirectories
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    
    try:
        with console_lock:
            logger.info(f"Downloading {object_key} to {local_file_path}")
        
        # Check if file already exists and has size > 0
        if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
            with console_lock:
                logger.info(f"File {object_key} already exists, skipping download")
            return True
        
        # Download with progress indication for large files
        start_time = time.time()
        s3_client.download_file(S3_BUCKET, object_key, local_file_path)
        download_time = time.time() - start_time
        
        # Get file size for reporting
        file_size = os.path.getsize(local_file_path) / (1024 * 1024)  # Size in MB
        
        with console_lock:
            logger.info(f"Downloaded {object_key} ({file_size:.2f} MB) in {download_time:.2f} seconds")
        return True
    except ClientError as e:
        with console_lock:
            logger.error(f"Failed to download {object_key}: {e}")
        return False

def download_all_files(output_dir, threads=4):
    """Download all files from the S3 bucket to the specified directory."""
    if not check_aws_credentials():
        logger.error("AWS credentials not found. Please configure your AWS credentials.")
        logger.info("You can set up AWS credentials by:")
        logger.info("1. Setting AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        logger.info("2. Running 'aws configure' if you have the AWS CLI installed")
        logger.info("3. Creating credentials file at ~/.aws/credentials")
        sys.exit(1)
    
    s3_client = initialize_s3_client()
    objects = list_bucket_objects(s3_client)
    
    if not objects:
        logger.error("No files to download. Please check your AWS credentials and bucket name.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    total_objects = len(objects)
    logger.info(f"Found {total_objects} objects in the S3 bucket")
    
    # Download files in parallel
    successful_downloads = 0
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(download_file, s3_client, obj['Key'], output_dir, total_objects) 
            for obj in objects
        ]
        
        for future in futures:
            if future.result():
                successful_downloads += 1
    
    logger.info(f"Download complete: {successful_downloads}/{total_objects} files downloaded successfully")
    
    if successful_downloads == total_objects:
        logger.info("All files were downloaded successfully!")
    else:
        logger.warning(f"Some files failed to download ({total_objects - successful_downloads} failures)")
    
    return successful_downloads == total_objects

def verify_dataset(output_dir):
    """Verify that critical dataset files exist after download."""
    missing_files = []
    for file in CRITICAL_FILES:
        file_path = os.path.join(output_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        elif os.path.getsize(file_path) == 0:
            missing_files.append(f"{file} (empty file)")
    
    if missing_files:
        logger.warning("Some critical dataset files are missing:")
        for file in missing_files:
            logger.warning(f"  - {file}")
        return False
    
    logger.info("All critical dataset files were successfully downloaded")
    
    # Also check the total size of the downloaded files
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) 
                     if os.path.isfile(os.path.join(output_dir, f)))
    
    total_size_mb = total_size / (1024 * 1024)
    logger.info(f"Total dataset size: {total_size_mb:.2f} MB")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Download Lichess puzzle dataset from S3")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for downloaded files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=4,
        help="Number of download threads to use (default: 4)"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify that all critical files were downloaded successfully"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting download of processed Lichess puzzle dataset")
    logger.info(f"S3 bucket: {S3_BUCKET}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using {args.threads} download threads")
    
    success = download_all_files(args.output_dir, args.threads)
    
    if args.verify or not success:
        verify_dataset(args.output_dir)

if __name__ == "__main__":
    main()