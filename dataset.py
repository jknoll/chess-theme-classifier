import torch
from torch.utils.data import Dataset
import pandas as pd
from chess import Board, Piece, SQUARES
import numpy as np
import os
import json
import time
import concurrent.futures
from functools import lru_cache
import multiprocessing
try:
    from torch.serialization import add_safe_globals
except ImportError:
    # Fallback for older PyTorch versions
    add_safe_globals = None

# Limit OpenMP threads to avoid resource exhaustion
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads

class ChessPuzzleDataset(Dataset):
    def __init__(self, csv_file, cache_size=10000, num_workers=None, augment_with_reflections=False, 
                 class_conditional_augmentation=False, rarity_threshold=None, low_memory=False,
                 use_cache=False, verbose_progress=False):
        """
        Args:
            csv_file (str): Path to the CSV file with chess puzzles
            cache_size (int): Size of the LRU cache for tensor storage in memory
            num_workers (int): Number of workers for parallel processing. If None, uses CPU count.
            augment_with_reflections (bool): If True, augment all boards with horizontal reflections.
            class_conditional_augmentation (bool): If True, only augment underrepresented label combinations.
            rarity_threshold (int): Optional threshold below which a label combination is considered rare.
                If None, will be automatically determined based on distribution.
            low_memory (bool): If True, use fewer workers and smaller chunks to reduce memory usage.
            use_cache (bool): If True, use cache files even if CSV exists and is newer than cache.
            verbose_progress (bool): If True, show detailed progress bars during processing.
        """
        self.csv_file = csv_file
        self.cache_size = cache_size
        
        # Set num_workers based on low_memory flag
        if low_memory:
            self.num_workers = 1  # Use just 1 worker in low memory mode
        else:
            self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
            
        self.low_memory = low_memory
        self.augment_with_reflections = augment_with_reflections
        self.class_conditional_augmentation = class_conditional_augmentation
        self.use_cache = use_cache  # Store the use_cache parameter
        self.verbose_progress = verbose_progress  # Store verbose progress flag
        self.augmented_indices = set()  # Track which indices were augmented
        
        # Cache file paths
        csv_basename = os.path.basename(csv_file)
        
        # Use TENSOR_CACHE_DIR environment variable if set, otherwise use csv_file directory
        cache_dir = os.environ.get('TENSOR_CACHE_DIR', os.path.dirname(csv_file))
        print(f"Using cache directory: {cache_dir}")
        
        self.themes_cache_file = os.path.join(cache_dir, f"{csv_basename}.themes.json")
        self.openings_cache_file = os.path.join(cache_dir, f"{csv_basename}.openings.json")
        self.tensors_cache_file = os.path.join(cache_dir, f"{csv_basename}.tensors.pt")
        self.label_cooccurrence_file = os.path.join(cache_dir, f"{csv_basename}.cooccurrence.json")
        
        # Check if the CSV file actually exists
        self.csv_exists = os.path.exists(csv_file)
        
        # Check if the path contains processed_lichess_puzzle_files
        # This means we're using the cache files directly from processed directory
        using_processed_dir = 'processed_lichess_puzzle_files' in csv_file
        
        # If we're using processed directory AND CSV doesn't exist, use cache-only mode
        # If CSV exists, we can create cache files normally
        if using_processed_dir and not self.csv_exists:
            # Extract the directory from csv_file
            cache_dir = os.path.dirname(csv_file)
            csv_basename = os.path.basename(csv_file)
            
            # Update cache file paths to use the processed directory
            self.themes_cache_file = os.path.join(cache_dir, f"{csv_basename}.themes.json")
            self.openings_cache_file = os.path.join(cache_dir, f"{csv_basename}.openings.json")
            self.tensors_cache_file = os.path.join(cache_dir, f"{csv_basename}.tensors.pt")
            self.label_cooccurrence_file = os.path.join(cache_dir, f"{csv_basename}.cooccurrence.json")
            
            # Check which cache files exist and which are needed
            essential_cache_files = []
            
            # Always need themes and openings
            essential_cache_files.append(self.themes_cache_file)
            essential_cache_files.append(self.openings_cache_file)
            
            # Add files based on the training mode
            if self.class_conditional_augmentation:
                essential_cache_files.append(self.label_cooccurrence_file)
                essential_cache_files.append(f"{self.tensors_cache_file}_conditional")
                essential_cache_files.append(f"{self.tensors_cache_file}_conditional.augmented_indices.json")
            else:
                # If not using class conditional augmentation, just need basic tensors
                essential_cache_files.append(self.tensors_cache_file)
            
            # Check if essential cache files exist
            essential_files_exist = all(os.path.exists(f) for f in essential_cache_files)
            
            if not essential_files_exist:
                # Get missing essential files for better error message
                missing_files = [f for f in essential_cache_files if not os.path.exists(f)]
                missing_str = '\n  - '.join(missing_files)
                raise FileNotFoundError(
                    f"Essential cache files are missing from {cache_dir}:\n  - {missing_str}\n"
                    "Please ensure all required cache files are present for the current training mode."
                )
                
            # We're in cache-only mode since CSV doesn't exist
            self.csv_exists = False
            print(f"Using cache files from {cache_dir}")
        elif using_processed_dir and self.csv_exists:
            # We have CSV in processed directory, update cache paths to use the same directory
            cache_dir = os.path.dirname(csv_file)
            csv_basename = os.path.basename(csv_file)
            
            # Update cache file paths to use the processed directory
            self.themes_cache_file = os.path.join(cache_dir, f"{csv_basename}.themes.json")
            self.openings_cache_file = os.path.join(cache_dir, f"{csv_basename}.openings.json")
            self.tensors_cache_file = os.path.join(cache_dir, f"{csv_basename}.tensors.pt")
            self.label_cooccurrence_file = os.path.join(cache_dir, f"{csv_basename}.cooccurrence.json")
            print(f"CSV found in processed directory, will create cache files there: {cache_dir}")
            
            # Load the CSV data
            print(f"Loading CSV data from {csv_file}...")
            self.puzzle_data = pd.read_csv(csv_file)
            
        # Handle normal case when not using processed directory and CSV doesn't exist
        elif not self.csv_exists:
            # CSV doesn't exist and we're not using the processed directory
            # Check which cache files are essential based on the training mode
            essential_cache_files = []
            
            # Always need themes and openings
            essential_cache_files.append(self.themes_cache_file)
            essential_cache_files.append(self.openings_cache_file)
            
            # Add files based on the training mode
            if self.class_conditional_augmentation:
                essential_cache_files.append(self.label_cooccurrence_file)
                essential_cache_files.append(f"{self.tensors_cache_file}_conditional")
                essential_cache_files.append(f"{self.tensors_cache_file}_conditional.augmented_indices.json")
            else:
                # If not using class conditional augmentation, just need basic tensors
                essential_cache_files.append(self.tensors_cache_file)
            
            # Check if essential cache files exist
            essential_files_exist = all(os.path.exists(f) for f in essential_cache_files)
            
            if not essential_files_exist:
                # Get missing essential files for better error message
                missing_files = [f for f in essential_cache_files if not os.path.exists(f)]
                missing_str = '\n  - '.join(missing_files)
                raise FileNotFoundError(
                    f"CSV file {csv_file} not found and essential cache files are missing:\n  - {missing_str}\n"
                    "Please provide either the original CSV file or the essential pre-processed cache files for the current training mode."
                )
                
            print(f"CSV file {csv_file} not found, but essential cache files exist. Proceeding with cache-only mode.")
            # Create a minimal DataFrame with enough structure for other methods
            self.puzzle_data = pd.DataFrame(columns=['FEN', 'Themes', 'OpeningTags'])
        else:
            # Read CSV data as usual (for normal paths outside processed directory)
            self.puzzle_data = pd.read_csv(csv_file)
        
        # Load or create theme and opening tag caches
        self._load_or_create_caches()
        
        # Combine and sort all labels
        self.all_labels = sorted(list(self.all_themes) + list(self.all_opening_tags))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        
        # Analyze label co-occurrence if doing class-conditional augmentation
        if class_conditional_augmentation:
            self.label_combinations, self.rare_combinations = self._analyze_label_cooccurrence(rarity_threshold)
            print(f"Found {len(self.label_combinations)} unique label combinations")
            print(f"Identified {len(self.rare_combinations)} rare combinations below threshold")
        
        # Set up LRU cache for tensor conversion
        # We can't directly apply lru_cache to a static method, so create a wrapper function
        self._get_cached_tensor = lru_cache(maxsize=cache_size)(
            lambda fen: ChessPuzzleDataset._optimized_fen_to_tensor(fen)
        )
        
        # Create or load tensor cache
        if class_conditional_augmentation:
            self._load_or_create_tensor_cache_conditional()
        else:
            self._load_or_create_tensor_cache(augment_with_reflections)
    
    def _load_or_create_caches(self):
        """Load themes and opening tags from cache files if they exist and are newer than the CSV,
        otherwise extract them from the CSV and save to cache files."""
        
        # If CSV doesn't exist, we must use the cache files
        if not self.csv_exists:
            missing_files = []
            if not os.path.exists(self.themes_cache_file):
                missing_files.append(self.themes_cache_file)
            if not os.path.exists(self.openings_cache_file):
                missing_files.append(self.openings_cache_file)
                
            if not missing_files:
                print(f"CSV file not found. Loading themes and openings from cache files")
                self._load_themes_and_openings_from_cache()
            else:
                # Try creating minimal empty caches if possible
                if missing_files:
                    print(f"Warning: The following theme/opening cache files are missing:")
                    for f in missing_files:
                        print(f"  - {f}")
                    print(f"Attempting to create minimal empty caches...")
                    
                    # Create minimal cache files with empty data
                    try:
                        if self.themes_cache_file in missing_files:
                            with open(self.themes_cache_file, 'w') as f:
                                json.dump([], f)
                            print(f"Created empty themes cache file")
                            
                        if self.openings_cache_file in missing_files:
                            with open(self.openings_cache_file, 'w') as f:
                                json.dump([], f)
                            print(f"Created empty openings cache file")
                            
                        # Initialize with empty sets
                        self.all_themes = set()
                        self.all_opening_tags = set()
                        print(f"Initialized with empty themes and openings")
                    except Exception as e:
                        # If we can't create the files, raise an error
                        raise FileNotFoundError(
                            f"CSV file not found and unable to create theme/opening cache files:\n"
                            f"  - {', '.join(missing_files)}\n"
                            f"Error: {str(e)}\n"
                            f"Cannot proceed without either the CSV file or the cache files."
                        )
                else:
                    # Load from existing cache files
                    self._load_themes_and_openings_from_cache()
            return
            
        # Normal flow when CSV exists
        csv_mtime = os.path.getmtime(self.csv_file)
        cache_is_valid = (
            os.path.exists(self.themes_cache_file) and
            os.path.exists(self.openings_cache_file) and
            os.path.getmtime(self.themes_cache_file) > csv_mtime and
            os.path.getmtime(self.openings_cache_file) > csv_mtime
        )
        
        if cache_is_valid:
            # Load from cache files
            print(f"Loading themes and openings from cache files")
            self._load_themes_and_openings_from_cache()
        else:
            # Extract from CSV and save to cache
            print(f"Extracting themes and openings from CSV and creating cache files")
            self.all_themes = set()
            self.all_opening_tags = set()
            
            # Process themes
            for themes_str in self.puzzle_data['Themes']:
                themes_list = themes_str.split()
                self.all_themes.update(themes_list)
                
            # Process opening tags
            for tags_str in self.puzzle_data['OpeningTags']:
                if pd.notna(tags_str):  # Handle potential NaN values
                    tags_list = tags_str.split()
                    self.all_opening_tags.update(tags_list)
            
            # Save to cache files
            with open(self.themes_cache_file, 'w') as f:
                json.dump(sorted(list(self.all_themes)), f)
            with open(self.openings_cache_file, 'w') as f:
                json.dump(sorted(list(self.all_opening_tags)), f)
                
    def _load_themes_and_openings_from_cache(self):
        """Helper method to load themes and openings from cache files."""
        with open(self.themes_cache_file, 'r') as f:
            themes_data = json.load(f)
            # Check if the format is a list of [name, count] lists or just a list of names
            if themes_data and isinstance(themes_data[0], list):
                # Format is [[name, count], [name, count], ...]
                self.all_themes = set(item[0] for item in themes_data)
            else:
                # Format is [name, name, ...]
                self.all_themes = set(themes_data)
                
        with open(self.openings_cache_file, 'r') as f:
            openings_data = json.load(f)
            # Check if the format is a list of [name, count] lists or just a list of names
            if openings_data and isinstance(openings_data[0], list):
                # Format is [[name, count], [name, count], ...]
                self.all_opening_tags = set(item[0] for item in openings_data)
            else:
                # Format is [name, name, ...]
                self.all_opening_tags = set(openings_data)
        
    def __len__(self):
        if self.augment_with_reflections or self.class_conditional_augmentation:
            return len(self.tensor_cache)  # Will include both original and any augmented boards
        else:
            return len(self.puzzle_data)
    
    def _load_or_create_tensor_cache(self, augment_with_reflections=False):
        """Load tensor cache if it exists and is newer than the CSV,
        otherwise create it by batch processing all FENs.
        
        Args:
            augment_with_reflections (bool): If True, augment the dataset with horizontally 
                                             reflected boards.
        """
        
        cache_suffix = '_reflected' if augment_with_reflections else ''
        tensors_cache_file = f"{self.tensors_cache_file}{cache_suffix}"
        
        # If CSV doesn't exist, try to use the cache files
        if not self.csv_exists:
            if os.path.exists(tensors_cache_file):
                print(f"CSV file not found. Loading tensor cache from: {tensors_cache_file}")
                try:
                    self.tensor_cache = torch.load(tensors_cache_file, map_location='cpu', weights_only=False)
                    print(f"Loaded {len(self.tensor_cache)} tensors from cache")
                except Exception as e:
                    print(f"Warning: Error loading tensor cache: {e}")
                    # If we're using class conditional augmentation, check if that cache exists instead
                    conditional_cache_file = f"{self.tensors_cache_file}_conditional"
                    if os.path.exists(conditional_cache_file):
                        print(f"Attempting to use class conditional tensor cache instead: {conditional_cache_file}")
                        try:
                            self.tensor_cache = torch.load(conditional_cache_file, map_location='cpu', weights_only=False)
                            print(f"Successfully loaded {len(self.tensor_cache)} tensors from conditional cache")
                            # Create an empty list if we're successful with conditional cache
                            return
                        except Exception as e2:
                            print(f"Error loading conditional tensor cache: {e2}")
                    
                    # If we can't load either cache, create an empty tensor cache
                    print(f"Creating empty tensor cache as fallback")
                    self.tensor_cache = []
            else:
                # Try to see if conditional cache exists instead
                conditional_cache_file = f"{self.tensors_cache_file}_conditional"
                if os.path.exists(conditional_cache_file):
                    print(f"Regular tensor cache not found, but conditional cache exists: {conditional_cache_file}")
                    print(f"Attempting to use class conditional tensor cache instead")
                    try:
                        self.tensor_cache = torch.load(conditional_cache_file, map_location='cpu', weights_only=False)
                        print(f"Successfully loaded {len(self.tensor_cache)} tensors from conditional cache")
                        return
                    except Exception as e:
                        print(f"Error loading conditional tensor cache: {e}")
                
                print(f"Warning: CSV file not found and tensor cache file missing: {tensors_cache_file}")
                print(f"Creating empty tensor cache as fallback")
                self.tensor_cache = []
            return
            
        # Normal flow when CSV exists
        csv_mtime = os.path.getmtime(self.csv_file)
        
        # Check if cache exists
        tensor_cache_exists = os.path.exists(tensors_cache_file)
        
        # If use_cache is True, we'll use the cache file if it exists, regardless of modification time
        # Otherwise, we'll only use it if it's newer than the CSV
        tensor_cache_is_valid = tensor_cache_exists and (
            self.use_cache or os.path.getmtime(tensors_cache_file) > csv_mtime
        )
        
        if tensor_cache_is_valid:
            print(f"Loading tensors from cache file: {tensors_cache_file}")
            # Use memory mapping for efficient loading of large tensor files
            try:
                self.tensor_cache = torch.load(tensors_cache_file, map_location='cpu', weights_only=False)
                print(f"Loaded {len(self.tensor_cache)} tensors from cache")
            except Exception as e:
                print(f"Error loading tensor cache: {e}")
                print("Regenerating tensor cache...")
                self._create_tensor_cache(augment_with_reflections)
        else:
            self._create_tensor_cache(augment_with_reflections)
    
    def _create_tensor_cache(self, augment_with_reflections=False):
        """Create the tensor cache by batch processing all FENs.
        
        Args:
            augment_with_reflections (bool): If True, augment the dataset with horizontally 
                                             reflected boards.
        """
        print(f"Creating tensor cache for {len(self.puzzle_data):,} puzzles...")
        start_time = time.time()
        
        # Get all FENs
        all_fens = self.puzzle_data['FEN'].tolist()
        
        # Create tensors in parallel
        self.tensor_cache = self._parallel_batch_conversion(all_fens, augment_with_reflections)
        
        # Save to disk
        cache_suffix = '_reflected' if augment_with_reflections else ''
        tensors_cache_file = f"{self.tensors_cache_file}{cache_suffix}"
        print(f"Saving tensor cache to: {tensors_cache_file}")
        torch.save(self.tensor_cache, tensors_cache_file)
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Created and saved tensor cache in {elapsed:.2f} seconds")
        base_conversion_rate = len(all_fens) / elapsed
        print(f"Conversion rate: {base_conversion_rate:.2f} puzzles/second")
        if augment_with_reflections:
            print(f"Total number of tensors (including reflections): {len(self.tensor_cache)}")
    
    @staticmethod
    def reflect_tensor_horizontally(tensor):
        """Reflect a chess board tensor horizontally (along vertical axis).
        
        Args:
            tensor (torch.Tensor): A tensor representing a chess board.
            
        Returns:
            torch.Tensor: The horizontally reflected board tensor.
        """
        # Flip the tensor along the horizontal axis (dim=1 for second dimension)
        return torch.flip(tensor.clone(), dims=[1])
    
    # Helper function for parallel processing - needs to be a module-level function to be picklable
    @staticmethod
    def _process_chunk(chunk):
        """Process a chunk of FENs and convert them to tensors"""
        # Our tensor representation uses these indices
        piece_to_idx = {
            'K': 0,   # white king
            'Q': 1,   # white queen
            'R': 2,   # white rook
            'B': 3,   # white bishop
            'N': 4,   # white knight
            'P': 5,   # white pawn
            'p': 7,   # black pawn
            'n': 8,   # black knight
            'b': 9,   # black bishop
            'r': 10,  # black rook
            'q': 11,  # black queen
            'k': 12,  # black king
        }
        
        results = []
        for fen in chunk:
            # Initialize tensor with empty squares (6)
            tensor = torch.full((8, 8), 6, dtype=torch.int8)
            
            # Get just the board part of the FEN (before the first space)
            board_fen = fen.split(' ')[0]
            
            # Parse the FEN board representation directly
            rank = 0
            file = 0
            
            for char in board_fen:
                if char == '/':
                    # Move to the next rank, reset file
                    rank += 1
                    file = 0
                elif char.isdigit():
                    # Skip empty squares
                    file += int(char)
                else:
                    # Place a piece
                    if char in piece_to_idx:
                        tensor[rank, file] = piece_to_idx[char]
                    file += 1
            
            results.append(tensor)
        
        return results
        
    # A static helper for the multiprocessing executor to avoid lambda pickling issues
    @staticmethod
    def _process_chunk_with_args_wrapper(args):
        """Wrapper to unpack arguments for _process_chunk_with_args to avoid pickling issues"""
        chunk, augment_with_reflections = args
        return ChessPuzzleDataset._process_chunk_with_args(chunk, augment_with_reflections)
    
    def _parallel_batch_conversion(self, fen_list, augment_with_reflections=False):
        """Use multiple CPU cores for FEN-to-tensor conversion
        
        Args:
            fen_list (list): List of FEN strings to convert.
            augment_with_reflections (bool): If True, augment the dataset with horizontally 
                                             reflected boards.
        
        Returns:
            list: List of tensors representing the boards.
        """
        # Process the full dataset - no artificial limits
        # Note: For very large datasets, we'll use memory-efficient processing instead of limiting samples
        print(f"Processing full dataset: {len(fen_list):,} FENs")
        
        # Split the FEN list into chunks - optimize chunk size based on dataset size
        if self.low_memory:
            chunk_size = 50  # Very small chunks for low memory mode
        elif len(fen_list) > 1000000:  # For very large datasets (>1M), use smaller chunks
            chunk_size = 1000  # Process 1000 FENs per chunk to manage memory
        else:
            chunk_size = max(1, len(fen_list) // self.num_workers)
            
        # Ensure chunk size is reasonable for large datasets
        chunk_size = min(chunk_size, 10000)  # Cap at 10K per chunk
            
        chunks = [fen_list[i:i+chunk_size] for i in range(0, len(fen_list), chunk_size)]
        
        print(f"Processing {len(fen_list):,} FENs in {len(chunks)} chunks using {self.num_workers} workers (low_memory={self.low_memory})")
        
        # Process chunks in a more memory-efficient way
        all_tensors = []
        
        # Use ProcessPoolExecutor with a timeout to prevent hanging
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Process each chunk, with info about whether to include reflections
            if augment_with_reflections:
                # Create a list of tuples with (chunk, augment_flag) for each chunk
                chunk_args = [(chunk, True) for chunk in chunks]
                # Use the wrapper function to avoid lambda pickling issues
                try:
                    from tqdm import tqdm
                    tqdm_available = True
                except ImportError:
                    tqdm_available = False
                
                # Process chunks with a timeout to prevent hanging
                futures = []
                for chunk_arg in chunk_args:
                    future = executor.submit(self._process_chunk_with_args_wrapper, chunk_arg)
                    futures.append(future)
                
                # Use tqdm to show progress while waiting for results
                results = []
                failed_chunks = []
                
                if tqdm_available and self.verbose_progress:
                    progress_desc = "Converting FENs to tensors (with reflections)"
                    iterator = tqdm(concurrent.futures.as_completed(futures), 
                                   total=len(futures), desc=progress_desc)
                elif self.verbose_progress:
                    print("Converting FENs to tensors (with reflections)...")
                    iterator = concurrent.futures.as_completed(futures)
                else:
                    iterator = concurrent.futures.as_completed(futures)
                
                for i, future in enumerate(iterator):
                        try:
                            result = future.result(timeout=120)  # Increased timeout to 120 seconds
                            results.append(result)
                        except concurrent.futures.TimeoutError:
                            print(f"⚠️ Warning: A worker process timed out. Will retry this chunk.")
                            failed_chunks.append(chunk_args[i])
                        except Exception as e:
                            print(f"⚠️ Error processing chunk: {e}. Will retry this chunk.")
                            failed_chunks.append(chunk_args[i])
                            
                # This code block is now handled above with conditional tqdm
                
                # Retry failed chunks sequentially to avoid further issues
                if failed_chunks:
                    retry_msg = f"Retrying {len(failed_chunks)} failed chunks sequentially..."
                    print(retry_msg)
                    
                    if self.verbose_progress:
                        try:
                            from tqdm import tqdm
                            retry_iterator = tqdm(failed_chunks, desc="Retrying failed chunks")
                        except ImportError:
                            retry_iterator = failed_chunks
                    else:
                        retry_iterator = failed_chunks
                        
                    for chunk_arg in retry_iterator:
                        try:
                            result = self._process_chunk_with_args_wrapper(chunk_arg)
                            results.append(result)
                            if self.verbose_progress:
                                print(f"✅ Successfully retried chunk with {len(chunk_arg[0])} items")
                        except Exception as e:
                            if self.verbose_progress:
                                print(f"❌ Failed to retry chunk even sequentially: {e}")
                            # As last resort, process individual FENs
                            chunk, augment_flag = chunk_arg
                            individual_count = 0
                            for fen in chunk:
                                try:
                                    if augment_flag:
                                        individual_result = self._process_chunk_with_args([fen], True)
                                    else:
                                        individual_result = self._process_chunk([fen])
                                    results.append(individual_result)
                                    individual_count += 1
                                except Exception as fen_error:
                                    if self.verbose_progress:
                                        print(f"❌ Failed to process individual FEN: {fen_error}")
                                    continue
                            if self.verbose_progress and individual_count > 0:
                                print(f"✅ Processed {individual_count}/{len(chunk)} individual FENs")
            else:
                # No reflections, use the original method but with improved error handling
                try:
                    from tqdm import tqdm
                    tqdm_available = True
                except ImportError:
                    tqdm_available = False
                
                # Process chunks with a timeout to prevent hanging
                futures = []
                for chunk in chunks:
                    future = executor.submit(self._process_chunk, chunk)
                    futures.append(future)
                
                # Use tqdm to show progress while waiting for results
                results = []
                failed_chunks = []
                
                if tqdm_available and self.verbose_progress:
                    progress_desc = "Converting FENs to tensors"
                    iterator = tqdm(concurrent.futures.as_completed(futures), 
                                   total=len(futures), desc=progress_desc)
                elif self.verbose_progress:
                    print("Converting FENs to tensors...")
                    iterator = concurrent.futures.as_completed(futures)
                else:
                    iterator = concurrent.futures.as_completed(futures)
                
                for i, future in enumerate(iterator):
                        try:
                            result = future.result(timeout=120)  # Increased timeout to 120 seconds
                            results.append(result)
                        except concurrent.futures.TimeoutError:
                            print(f"⚠️ Warning: A worker process timed out. Will retry this chunk.")
                            failed_chunks.append(chunks[i])
                        except Exception as e:
                            print(f"⚠️ Error processing chunk: {e}. Will retry this chunk.")
                            failed_chunks.append(chunks[i])
                            
                # This code block is now handled above with conditional tqdm
                
                # Retry failed chunks sequentially for non-reflection path
                if failed_chunks:
                    retry_msg = f"Retrying {len(failed_chunks)} failed chunks sequentially..."
                    print(retry_msg)
                    
                    if self.verbose_progress:
                        try:
                            from tqdm import tqdm
                            retry_iterator = tqdm(failed_chunks, desc="Retrying failed chunks")
                        except ImportError:
                            retry_iterator = failed_chunks
                    else:
                        retry_iterator = failed_chunks
                        
                    for chunk in retry_iterator:
                        try:
                            result = self._process_chunk(chunk)
                            results.append(result)
                            if self.verbose_progress:
                                print(f"✅ Successfully retried chunk with {len(chunk)} items")
                        except Exception as e:
                            if self.verbose_progress:
                                print(f"❌ Failed to retry chunk even sequentially: {e}")
                            # As last resort, process individual FENs
                            individual_count = 0
                            for fen in chunk:
                                try:
                                    individual_result = self._process_chunk([fen])
                                    results.append(individual_result)
                                    individual_count += 1
                                except Exception as fen_error:
                                    if self.verbose_progress:
                                        print(f"❌ Failed to process individual FEN: {fen_error}")
                                    continue
                            if self.verbose_progress and individual_count > 0:
                                print(f"✅ Processed {individual_count}/{len(chunk)} individual FENs")
        
        # Combine results
        for result in results:
            all_tensors.extend(result)
        
        # Detailed reporting
        expected_count = len(fen_list)
        actual_count = len(all_tensors)
        success_rate = (actual_count / expected_count) * 100 if expected_count > 0 else 0
        
        print(f"📊 Processing Summary:")
        print(f"   Expected: {expected_count:,} FENs")
        print(f"   Processed: {actual_count:,} tensors")
        print(f"   Success rate: {success_rate:.2f}%")
        
        if actual_count < expected_count:
            missing_count = expected_count - actual_count
            print(f"⚠️  Missing {missing_count:,} tensors ({(missing_count/expected_count)*100:.2f}%)")
        else:
            print(f"✅ Successfully processed all tensors!")
            
        return all_tensors
    
    @staticmethod
    def _process_chunk_with_args(chunk, augment_with_reflections=False):
        """Process a chunk of FENs and convert them to tensors, with option for reflections
        
        Args:
            chunk (list): List of FEN strings to convert.
            augment_with_reflections (bool): If True, add horizontal reflections.
            
        Returns:
            list: List of tensors, potentially including reflections.
        """
        # Get the base tensors first
        base_tensors = ChessPuzzleDataset._process_chunk(chunk)
        
        if not augment_with_reflections:
            return base_tensors
            
        # If augmenting with reflections, create a new list with originals and reflections
        augmented_tensors = []
        for tensor in base_tensors:
            # Add the original tensor
            augmented_tensors.append(tensor)
            # Add the horizontally reflected tensor
            reflected_tensor = ChessPuzzleDataset.reflect_tensor_horizontally(tensor)
            augmented_tensors.append(reflected_tensor)
            
        return augmented_tensors
    
    @staticmethod
    def _optimized_fen_to_tensor(fen):
        """
        A faster FEN to tensor conversion function that parses the FEN directly
        without creating a full Board object
        """
        # Our tensor representation uses these indices
        piece_to_idx = {
            'K': 0,   # white king
            'Q': 1,   # white queen
            'R': 2,   # white rook
            'B': 3,   # white bishop
            'N': 4,   # white knight
            'P': 5,   # white pawn
            'p': 7,   # black pawn
            'n': 8,   # black knight
            'b': 9,   # black bishop
            'r': 10,  # black rook
            'q': 11,  # black queen
            'k': 12,  # black king
        }
        
        # Initialize tensor with empty squares (6)
        tensor = torch.full((8, 8), 6, dtype=torch.int8)
        
        # Get just the board part of the FEN (before the first space)
        board_fen = fen.split(' ')[0]
        
        # Parse the FEN board representation directly
        rank = 0
        file = 0
        
        for char in board_fen:
            if char == '/':
                # Move to the next rank, reset file
                rank += 1
                file = 0
            elif char.isdigit():
                # Skip empty squares
                file += int(char)
            else:
                # Place a piece
                if char in piece_to_idx:
                    tensor[rank, file] = piece_to_idx[char]
                file += 1
        
        return tensor
    
    def _load_or_create_tensor_cache_conditional(self):
        """
        Load tensor cache if it exists and is newer than the CSV and co-occurrence cache,
        otherwise create it by selectively augmenting only the rare label combinations.
        Handles both old (tensors only) and new (tensors and labels) cache formats.
        """
        # Define cache file with conditional suffix
        conditional_suffix = '_conditional'
        tensors_cache_file = f"{self.tensors_cache_file}{conditional_suffix}"
        
        # If CSV doesn't exist, try to use the cache files
        if not self.csv_exists:
            if os.path.exists(tensors_cache_file):
                print(f"CSV file not found. Loading tensor cache from: {tensors_cache_file}")
                try:
                    # Load the cache file
                    cache_data = torch.load(tensors_cache_file, map_location='cpu', weights_only=False)
                    
                    # Check if this is the new format (dictionary with tensors and labels)
                    if isinstance(cache_data, dict) and 'tensors' in cache_data and 'labels' in cache_data:
                        print(f"Found new format cache with tensors and labels (version {cache_data.get('format_version', 1)})")
                        self.tensor_cache = cache_data['tensors']
                        self.label_cache = cache_data['labels']
                        has_labels = True
                    else:
                        # Old format (just tensors)
                        print(f"Found old format cache (tensors only)")
                        self.tensor_cache = cache_data
                        # Create an empty label cache
                        self.label_cache = None
                        has_labels = False
                    
                    # Load the augmented indices
                    augmented_indices_file = f"{tensors_cache_file}.augmented_indices.json"
                    if os.path.exists(augmented_indices_file):
                        with open(augmented_indices_file, 'r') as f:
                            self.augmented_indices = set(json.load(f))
                        print(f"Loaded {len(self.tensor_cache)} tensors from cache")
                        print(f"Including {len(self.augmented_indices)} augmented (reflected) rare samples")
                        if has_labels:
                            print(f"Cache includes label data for use in cache-only mode")
                    else:
                        print(f"Warning: Augmented indices file not found: {augmented_indices_file}")
                        print(f"Continuing with empty augmented indices")
                        self.augmented_indices = set()
                except Exception as e:
                    print(f"Warning: Error loading conditional tensor cache: {e}")
                    # Try loading non-conditional cache as a fallback
                    regular_cache = self.tensors_cache_file
                    if os.path.exists(regular_cache):
                        print(f"Attempting to use regular tensor cache instead: {regular_cache}")
                        try:
                            cache_data = torch.load(regular_cache, map_location='cpu', weights_only=False)
                            if isinstance(cache_data, dict) and 'tensors' in cache_data:
                                self.tensor_cache = cache_data['tensors']
                                self.label_cache = cache_data.get('labels')
                            else:
                                self.tensor_cache = cache_data
                                self.label_cache = None
                            print(f"Successfully loaded {len(self.tensor_cache)} tensors from regular cache")
                            self.augmented_indices = set()  # No augmentation in regular cache
                            return
                        except Exception as e2:
                            print(f"Error loading regular tensor cache: {e2}")
                    
                    # If we can't load either cache, create an empty tensor cache
                    print(f"Creating empty tensor cache as fallback")
                    self.tensor_cache = []
                    self.label_cache = None
                    self.augmented_indices = set()
            else:
                # Try to see if non-conditional cache exists as a fallback
                regular_cache = self.tensors_cache_file
                if os.path.exists(regular_cache):
                    print(f"Conditional tensor cache not found, but regular cache exists: {regular_cache}")
                    print(f"Attempting to use regular tensor cache instead")
                    try:
                        cache_data = torch.load(regular_cache, map_location='cpu', weights_only=False)
                        if isinstance(cache_data, dict) and 'tensors' in cache_data:
                            self.tensor_cache = cache_data['tensors']
                            self.label_cache = cache_data.get('labels')
                        else:
                            self.tensor_cache = cache_data
                            self.label_cache = None
                        print(f"Successfully loaded {len(self.tensor_cache)} tensors from regular cache")
                        self.augmented_indices = set()  # No augmentation in regular cache
                        return
                    except Exception as e:
                        print(f"Error loading regular tensor cache: {e}")
                
                print(f"Warning: CSV file not found and all tensor cache files missing.")
                print(f"Creating empty tensor cache as fallback")
                self.tensor_cache = []
                self.label_cache = None
                self.augmented_indices = set()
            return
            
        # Normal flow when CSV exists
        csv_mtime = os.path.getmtime(self.csv_file)
        cooc_mtime = os.path.getmtime(self.label_cooccurrence_file) if os.path.exists(self.label_cooccurrence_file) else 0
        latest_mtime = max(csv_mtime, cooc_mtime)
        
        # Check if cache exists
        tensor_cache_exists = os.path.exists(tensors_cache_file)
        
        # If use_cache is True, we'll use the cache file if it exists, regardless of modification time
        # Otherwise, we'll only use it if it's newer than the latest modification time of prerequisites
        tensor_cache_is_valid = tensor_cache_exists and (
            self.use_cache or os.path.getmtime(tensors_cache_file) > latest_mtime
        )
        
        if tensor_cache_is_valid:
            print(f"Loading conditionally augmented tensors from cache file: {tensors_cache_file}")
            # Use memory mapping for efficient loading of large tensor files
            try:
                cache_data = torch.load(tensors_cache_file, map_location='cpu', weights_only=False)
                
                # Check if this is the new format (dictionary with tensors and labels)
                if isinstance(cache_data, dict) and 'tensors' in cache_data and 'labels' in cache_data:
                    print(f"Found new format cache with tensors and labels (version {cache_data.get('format_version', 1)})")
                    self.tensor_cache = cache_data['tensors']
                    self.label_cache = cache_data['labels']
                else:
                    # Old format (just tensors)
                    print(f"Found old format cache (tensors only)")
                    self.tensor_cache = cache_data
                    self.label_cache = None
                
                # Load the augmented indices
                augmented_indices_file = f"{tensors_cache_file}.augmented_indices.json"
                if os.path.exists(augmented_indices_file):
                    with open(augmented_indices_file, 'r') as f:
                        self.augmented_indices = set(json.load(f))
                print(f"Loaded {len(self.tensor_cache)} tensors from cache")
                print(f"Including {len(self.augmented_indices)} augmented (reflected) rare samples")
                if self.label_cache is not None:
                    print(f"Cache includes label data for use in cache-only mode")
            except Exception as e:
                print(f"Error loading tensor cache: {e}")
                print("Regenerating tensor cache...")
                self._create_tensor_cache_conditional()
        else:
            self._create_tensor_cache_conditional()
    
    def _create_tensor_cache_conditional(self):
        """
        Create a tensor cache with conditional augmentation based on label rarity.
        Only rare label combinations will be augmented with horizontal reflections.
        Also includes labels in the cache for use in cache-only mode.
        """
        print(f"Creating conditional tensor cache for {len(self.puzzle_data):,} puzzles...")
        start_time = time.time()
        
        # Get all FENs
        all_fens = self.puzzle_data['FEN'].tolist()
        
        # Create basic tensors first (no augmentation)
        basic_tensors = self._parallel_batch_conversion(all_fens, augment_with_reflections=False)
        
        # Check if any tensors were successfully created
        if not basic_tensors:
            print("⚠️ No basic tensors were created. Using an empty tensor cache.")
            self.tensor_cache = []
            self.augmented_indices = set()
            return
            
        # Make sure we only process as many indices as we have tensors
        valid_indices = min(len(self.puzzle_data), len(basic_tensors))
        if valid_indices < len(self.puzzle_data):
            print(f"⚠️ Warning: Only processed {valid_indices} out of {len(self.puzzle_data)} puzzles.")
            
        # Now identify which indices should be augmented (those with rare theme combinations)
        self.augmented_indices = set()
        
        # Store both tensors and labels in the cache
        cache_entries = []
        
        # Map from original index to tensor cache index
        original_to_cache_idx = {}
        next_tensor_idx = 0
        
        # Add progress bar for augmentation process
        if self.verbose_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(valid_indices), desc="Creating augmented dataset")
            except ImportError:
                print("Processing tensors for augmentation...")
                iterator = range(valid_indices)
        else:
            iterator = range(valid_indices)
        
        for idx in iterator:
            # Create label vector for this item
            label_vector = torch.zeros(len(self.all_labels), dtype=torch.float32)
            
            # Get themes and opening tags
            themes = self.puzzle_data.iloc[idx]['Themes'].split()
            opening_tags = []
            if pd.notna(self.puzzle_data.iloc[idx]['OpeningTags']):
                opening_tags = self.puzzle_data.iloc[idx]['OpeningTags'].split()
            
            # Add themes to label vector
            for theme in themes:
                if theme in self.label_to_idx:  # Ensure theme exists in our index
                    theme_idx = self.label_to_idx[theme]
                    label_vector[theme_idx] = 1
            
            # Add opening tags to label vector
            for tag in opening_tags:
                if tag in self.label_to_idx:
                    tag_idx = self.label_to_idx[tag]
                    label_vector[tag_idx] = 1
            
            # Add the original tensor and label vector
            cache_entries.append({
                'tensor': basic_tensors[idx],
                'labels': label_vector,
                'is_reflection': False
            })
            original_to_cache_idx[idx] = next_tensor_idx
            next_tensor_idx += 1
            
            # Check if this puzzle has rare theme combinations that should be augmented
            # Use only theme labels as the label set
            theme_labels = frozenset(themes)
            
            # If this theme combination is rare, add a reflection
            if theme_labels in self.rare_combinations:
                # Reflect the tensor horizontally
                reflected_tensor = self.reflect_tensor_horizontally(basic_tensors[idx])
                
                # Create label vector for reflection (only include themes, not opening tags)
                reflection_label_vector = torch.zeros(len(self.all_labels), dtype=torch.float32)
                for theme in themes:
                    if theme in self.label_to_idx:
                        theme_idx = self.label_to_idx[theme]
                        reflection_label_vector[theme_idx] = 1
                
                # Add the reflected tensor and label vector
                cache_entries.append({
                    'tensor': reflected_tensor,
                    'labels': reflection_label_vector,
                    'is_reflection': True
                })
                
                # Track that this index was augmented
                self.augmented_indices.add(idx)
                next_tensor_idx += 1
        
        # Set the tensor cache (just the tensors for backward compatibility)
        self.tensor_cache = [entry['tensor'] for entry in cache_entries]
        self.label_cache = [entry['labels'] for entry in cache_entries]
        
        # Save to disk
        tensors_cache_file = f"{self.tensors_cache_file}_conditional"
        print(f"Saving conditional tensor cache with labels to: {tensors_cache_file}")
        torch.save({
            'tensors': self.tensor_cache,
            'labels': self.label_cache,
            'format_version': 2  # Add version to indicate the new format
        }, tensors_cache_file)
        
        # Also save the augmented indices for future reference
        augmented_indices_file = f"{tensors_cache_file}.augmented_indices.json"
        with open(augmented_indices_file, 'w') as f:
            json.dump(list(self.augmented_indices), f)
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Created and saved conditional tensor cache in {elapsed:.2f} seconds")
        print(f"Original dataset size: {len(self.puzzle_data)}")
        print(f"Augmented dataset size: {len(self.tensor_cache)}")
        print(f"Number of rare combinations augmented: {len(self.augmented_indices)}")
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Handle the case where we're running with cache files only (no CSV)
        if not self.csv_exists:
            # In this case, we can only access the tensor data and not the original FEN or labels
            # These will be derived from the tensor_cache and augmented_indices
            is_reflection = False
            original_idx = idx
            themes = []
            opening_tags = []
            fen = "cache_only_mode"  # Placeholder since we don't have the original FEN
            
            # We still need to determine if this is a reflection based on augmented_indices
            if self.class_conditional_augmentation and hasattr(self, 'augmented_indices'):
                # This logic attempts to infer if the current index is an augmented (reflected) entry
                # Note: Without the CSV, this is an approximation based on the cached augmented_indices
                augmented_count = 0
                for i in range(len(self.tensor_cache) // 2):
                    if i in self.augmented_indices:
                        augmented_count += 1
                        if idx == i + augmented_count:
                            is_reflection = True
                            original_idx = i
                            break
        
        # Handle the augmented dataset case (both full and conditional)
        elif self.augment_with_reflections or self.class_conditional_augmentation:
            # For class-conditional augmentation, the mapping is more complex
            if self.class_conditional_augmentation:
                # Get the original index and whether this is a reflection
                is_reflection = False
                original_idx = idx
                
                # Determine if this is an augmented (reflected) sample
                # This requires traversing until we reach the right index
                remaining_idx = idx
                for i in range(len(self.puzzle_data)):
                    # Original tensor takes 1 slot
                    remaining_idx -= 1
                    
                    # If this was augmented, the reflection takes another slot
                    if i in self.augmented_indices:
                        if remaining_idx < 0:
                            # This is the reflection of the previous index
                            original_idx = i
                            is_reflection = True
                            break
                        remaining_idx -= 1
                    
                    if remaining_idx < 0:
                        # This is the original tensor
                        original_idx = i
                        break
            else:
                # Simple case for full augmentation: even indices are originals, odd are reflections
                original_idx = idx // 2
                is_reflection = idx % 2 == 1
            
            # Get FEN and themes from the original puzzle
            fen = self.puzzle_data.iloc[original_idx]['FEN']
            themes = self.puzzle_data.iloc[original_idx]['Themes'].split()
            
            # For reflections, we keep the theme labels but strip opening tags
            # For non-reflections, we include both themes and opening tags
            opening_tags = []
            if not is_reflection and pd.notna(self.puzzle_data.iloc[original_idx]['OpeningTags']):
                opening_tags = self.puzzle_data.iloc[original_idx]['OpeningTags'].split()
        else:
            # Normal dataset without reflections
            fen = self.puzzle_data.iloc[idx]['FEN']
            themes = self.puzzle_data.iloc[idx]['Themes'].split()
            opening_tags = []
            if pd.notna(self.puzzle_data.iloc[idx]['OpeningTags']):
                opening_tags = self.puzzle_data.iloc[idx]['OpeningTags'].split()
            is_reflection = False
        
        # Create one-hot encoding for labels
        label_vector = torch.zeros(len(self.all_labels), dtype=torch.float32)
        
        # In cache-only mode, we won't have labels from the CSV
        # Instead, we'll rely on the labels embedded in the cache files
        if not self.csv_exists:
            # In cache-only mode, we must have cached labels
            if hasattr(self, 'label_cache') and self.label_cache is not None:
                # Use the cached labels for this index
                label_vector = self.label_cache[idx]
            else:
                # If no cached labels, raise an error - we can't train or predict without labels
                raise RuntimeError(
                    "Running in cache-only mode but no labels are available in the cache. "
                    "Please regenerate the cache with the latest version of the code that stores labels."
                )
        else:
            # Always include theme labels
            for theme in themes:
                if theme in self.label_to_idx:  # Ensure theme exists in our index
                    theme_idx = self.label_to_idx[theme]
                    label_vector[theme_idx] = 1
            
            # Only include opening tags for non-reflections
            if not is_reflection:
                for tag in opening_tags:
                    if tag in self.label_to_idx:
                        tag_idx = self.label_to_idx[tag]
                        label_vector[tag_idx] = 1
        
        # Get board tensor - use cached tensor if available
        board_tensor = self.tensor_cache[idx].to(dtype=torch.float32)
        
        return {
            'board': board_tensor,
            'themes': label_vector,
            'fen': fen,  # Include original FEN for reference
            'is_reflection': is_reflection,  # Flag if this is a reflection
            'original_idx': original_idx if is_reflection else idx,  # Track original index for tracing
            'only_themes': is_reflection  # Flag that this item only has theme labels (no openings)
        }
    
    def _board_to_tensor(self, board):
        """Convert a chess board to a tensor representation.
        Returns an 8x8 tensor where:
        White pieces:
            king = 0
            queen = 1
            rook = 2
            bishop = 3
            knight = 4
            pawn = 5
        Empty square = 6
        Black pieces:
            pawn = 7
            knight = 8
            bishop = 9
            rook = 10
            queen = 11
            king = 12
        """
        # Map piece symbols to indices
        piece_to_idx = {
            'K': 0,   # white king
            'Q': 1,   # white queen
            'R': 2,   # white rook
            'B': 3,   # white bishop
            'N': 4,   # white knight
            'P': 5,   # white pawn
            'p': 7,   # black pawn
            'n': 8,   # black knight
            'b': 9,   # black bishop
            'r': 10,  # black rook
            'q': 11,  # black queen
            'k': 12,  # black king
        }
        
        tensor = torch.full((8, 8), 6)  # Initialize with empty squares (6)
        
        for i in range(64):
            rank, file = i // 8, i % 8
            piece = board.piece_at(i)
            if piece is not None:
                tensor[rank, file] = piece_to_idx[piece.symbol()]
                
        return tensor
    
    def get_theme_names(self):
        """Return the list of all possible labels (themes and opening tags)."""
        return self.all_labels
        
    def get_themes_only(self):
        """Return only the puzzle themes (without opening tags)."""
        return sorted(list(self.all_themes))
        
    def get_openings_only(self):
        """Return only the opening tags."""
        return sorted(list(self.all_opening_tags))
        
    def is_opening_tag(self, label):
        """Check if a label is an opening tag."""
        return label in self.all_opening_tags
        
    def is_theme(self, label):
        """Check if a label is a puzzle theme."""
        return label in self.all_themes
    
    def get_theme_frequencies(self):
        """Return a list of theme frequencies sorted by count."""
        theme_counts = {}
        for idx in range(len(self.puzzle_data)):
            themes = self.puzzle_data.iloc[idx]['Themes'].split()
            for theme in themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Convert to list of (theme, count) tuples and sort by count in descending order
        freq_list = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return freq_list


    def get_opening_frequencies(self):
        """Return a list of opening tag frequencies sorted by count."""
        opening_counts = {}
        for idx in range(len(self.puzzle_data)):
            if pd.notna(self.puzzle_data.iloc[idx]['OpeningTags']):
                openings = self.puzzle_data.iloc[idx]['OpeningTags'].split()
                for opening in openings:
                    opening_counts[opening] = opening_counts.get(opening, 0) + 1
        
        # Convert to list of (opening, count) tuples and sort by count in descending order
        freq_list = sorted(opening_counts.items(), key=lambda x: x[1], reverse=True)
        return freq_list
    
    def _analyze_label_cooccurrence(self, rarity_threshold=None):
        """
        Analyze the co-occurrence patterns of theme labels in the dataset.
        Based on the algorithm in multilabel_reflection_algorithm.md
        
        Args:
            rarity_threshold (int, optional): Count below which a label combination is considered rare.
                If None, it's automatically determined based on distribution statistics.
                
        Returns:
            tuple: (label_combinations, rare_combinations)
                label_combinations: Dictionary mapping frozensets of theme labels to their frequency
                rare_combinations: Set of frozensets containing theme label combinations below threshold
        """
        # If CSV doesn't exist, try to use the cache file or create fallbacks
        if not self.csv_exists:
            if os.path.exists(self.label_cooccurrence_file):
                print(f"CSV file not found. Loading theme co-occurrence data from cache: {self.label_cooccurrence_file}")
                with open(self.label_cooccurrence_file, 'r') as f:
                    try:
                        cache_data = json.load(f)
                        
                        # Convert string representations back to frozensets
                        label_combinations = {
                            frozenset(eval(k)): v for k, v in cache_data['combinations'].items()
                        }
                        
                        # Get threshold from cache or compute from data
                        stored_threshold = cache_data.get('rarity_threshold', None)
                        if rarity_threshold is None:
                            rarity_threshold = stored_threshold
                        
                        # If we still don't have a threshold, compute one
                        if rarity_threshold is None:
                            rarity_threshold = self._compute_rarity_threshold(label_combinations)
                        
                        # Get rare combinations
                        rare_combinations = {
                            frozenset(eval(combo)) for combo in cache_data.get('rare_combinations', [])
                        }
                        
                        # Use the precomputed rare combinations
                        if not rare_combinations:
                            print(f"Computing rare combinations with threshold: {rarity_threshold}")
                            rare_combinations = self._identify_rare_combinations(label_combinations, rarity_threshold)
                        
                        return label_combinations, rare_combinations
                    except Exception as e:
                        print(f"Warning: Error loading co-occurrence data: {e}")
                        # Continue to fallback
            else:
                print(f"Warning: CSV file not found and co-occurrence cache file missing: {self.label_cooccurrence_file}")
            
            # Create fallback minimal co-occurrence data
            print("Creating minimal co-occurrence data as fallback")
            
            # If we have themes from cache, use them to create minimal combinations
            if hasattr(self, 'all_themes') and self.all_themes:
                # Create a single combination with one label from each theme for demonstration
                sample_themes = list(self.all_themes)[:min(5, len(self.all_themes))]
                label_combinations = {frozenset([theme]): 1 for theme in sample_themes}
                # Add one combination with multiple themes
                if len(sample_themes) >= 2:
                    label_combinations[frozenset(sample_themes[:2])] = 1
            else:
                # Create an empty placeholder
                label_combinations = {frozenset(['placeholder']): 1}
            
            # Set a default threshold
            if rarity_threshold is None:
                rarity_threshold = 1
                
            # Create empty rare combinations
            rare_combinations = set()
            
            # Save this fallback data to the cache file
            try:
                cache_dir = os.path.dirname(self.label_cooccurrence_file)
                os.makedirs(cache_dir, exist_ok=True)
                
                cache_data = {
                    'combinations': {str(list(k)): v for k, v in label_combinations.items()},
                    'rare_combinations': [],
                    'rarity_threshold': rarity_threshold
                }
                
                with open(self.label_cooccurrence_file, 'w') as f:
                    json.dump(cache_data, f)
                print(f"Saved minimal co-occurrence data to cache: {self.label_cooccurrence_file}")
            except Exception as e:
                print(f"Warning: Unable to save fallback co-occurrence data: {e}")
            
            return label_combinations, rare_combinations
                
        # Normal flow when CSV exists
        # Check if cache file exists and is newer than the CSV
        csv_mtime = os.path.getmtime(self.csv_file)
        cache_is_valid = (
            os.path.exists(self.label_cooccurrence_file) and
            os.path.getmtime(self.label_cooccurrence_file) > csv_mtime
        )
        
        if cache_is_valid:
            # Load from cache
            print(f"Loading theme co-occurrence data from cache: {self.label_cooccurrence_file}")
            with open(self.label_cooccurrence_file, 'r') as f:
                cache_data = json.load(f)
                
                # Convert string representations back to frozensets
                label_combinations = {
                    frozenset(eval(k)): v for k, v in cache_data['combinations'].items()
                }
                
                # If a threshold was provided in the cache, use it, otherwise compute from data
                stored_threshold = cache_data.get('rarity_threshold', None)
                if rarity_threshold is None:
                    rarity_threshold = stored_threshold
                
                # If we still don't have a threshold, compute one
                if rarity_threshold is None:
                    rarity_threshold = self._compute_rarity_threshold(label_combinations)
                
                # Get rare combinations
                rare_combinations = {
                    frozenset(eval(combo)) for combo in cache_data.get('rare_combinations', [])
                }
                
                # If rare_combinations is not in cache or threshold changed, recompute them
                if not rare_combinations or stored_threshold != rarity_threshold:
                    if stored_threshold != rarity_threshold:
                        print(f"Recomputing rare combinations with new threshold: {rarity_threshold}")
                    rare_combinations = self._identify_rare_combinations(label_combinations, rarity_threshold)
                
                return label_combinations, rare_combinations
        
        # Analyze co-occurrence patterns
        # Sample up to MAX_SAMPLES puzzles for co-occurrence analysis to avoid memory issues
        MAX_SAMPLES = 100000
        
        if len(self.puzzle_data) > MAX_SAMPLES:
            print(f"⚠️ Large dataset detected. Sampling {MAX_SAMPLES:,} out of {len(self.puzzle_data):,} puzzles for co-occurrence analysis.")
            # Use random sampling without replacement for a representative subset
            sample_indices = np.random.choice(len(self.puzzle_data), size=MAX_SAMPLES, replace=False)
            puzzle_subset = self.puzzle_data.iloc[sample_indices]
        else:
            puzzle_subset = self.puzzle_data
            
        print(f"Analyzing theme co-occurrence patterns for {len(puzzle_subset):,} puzzles...")
        label_combinations = {}
        
        # Process each puzzle with tqdm progress bar
        if self.verbose_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(len(puzzle_subset)), desc="Analyzing theme co-occurrences")
            except ImportError:
                print("Analyzing theme co-occurrences...")
                iterator = range(len(puzzle_subset))
        else:
            iterator = range(len(puzzle_subset))
            
        for idx in iterator:
            # Get ONLY themes (ignoring opening tags as per updated requirements)
            # Use puzzle_subset instead of self.puzzle_data
            themes = puzzle_subset.iloc[idx]['Themes'].split()
            
            # Use only theme labels as the label set
            theme_labels = frozenset(themes)
            
            # Count this combination
            label_combinations[theme_labels] = label_combinations.get(theme_labels, 0) + 1
        
        # Compute rarity threshold if not provided
        if rarity_threshold is None:
            rarity_threshold = self._compute_rarity_threshold(label_combinations)
        
        # Identify rare combinations using the method from multilabel_reflection_algorithm.md
        rare_combinations = self._identify_rare_combinations(label_combinations, rarity_threshold)
        
        # Save to cache for future use
        cache_data = {
            'combinations': {str(list(k)): v for k, v in label_combinations.items()},
            'rare_combinations': [str(list(combo)) for combo in rare_combinations],
            'rarity_threshold': rarity_threshold
        }
        
        with open(self.label_cooccurrence_file, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"Saved theme co-occurrence data to cache: {self.label_cooccurrence_file}")
        
        return label_combinations, rare_combinations
        
    def _identify_rare_combinations(self, label_combinations, threshold):
        """
        Identify rare label combinations using the algorithm from multilabel_reflection_algorithm.md
        
        Args:
            label_combinations: Dictionary mapping frozensets of theme labels to their frequency
            threshold: Count threshold below which a combination is initially considered
            
        Returns:
            Set of frozensets containing theme label combinations to augment
        """
        # Step 1: Compute total samples and unique combinations (already done in label_combinations)
        total_samples = sum(label_combinations.values())
        unique_combos = len(label_combinations)
        
        # Step 2: Compute ideal uniform count
        ideal_count = total_samples / unique_combos
        print(f"Total samples: {total_samples}, Unique combinations: {unique_combos}")
        print(f"Ideal count per combination: {ideal_count:.2f}")
        
        # Step 3: Calculate gain from reflection (improved balance metric)
        # For each label set Y, compute the L2 distance reduction gain from adding 1 more example
        # gain(Y) = (f_Y - ideal)^2 - (f_Y + 1 - ideal)^2
        gains = {}
        
        # Add progress bar for gain calculation
        if self.verbose_progress:
            try:
                from tqdm import tqdm
                items = tqdm(label_combinations.items(), desc="Calculating augmentation gains")
            except ImportError:
                print("Calculating augmentation gains...")
                items = label_combinations.items()
        else:
            items = label_combinations.items()
            
        for label_set, count in items:
            # Only consider combinations below threshold for potential augmentation
            if count <= threshold:
                # Calculate L2 distance reduction (how much adding 1 more improves balance)
                current_distance = (count - ideal_count) ** 2
                new_distance = (count + 1 - ideal_count) ** 2
                gains[label_set] = current_distance - new_distance
        
        # Sort label sets by gain (highest gain first)
        sorted_by_gain = sorted(gains.items(), key=lambda x: x[1], reverse=True)
        
        # If we have too many rare combinations, take the ones with highest gain
        rare_combinations = set(combo for combo, _ in sorted_by_gain)
        print(f"Identified {len(rare_combinations)} rare theme combinations to augment")
        
        return rare_combinations
    
    def _compute_rarity_threshold(self, label_combinations):
        """
        Compute a rarity threshold for label combinations based on distribution statistics.
        
        Args:
            label_combinations (dict): Dictionary mapping frozensets of labels to their frequency
            
        Returns:
            int: Computed rarity threshold
        """
        # Get the counts of all label combinations
        counts = list(label_combinations.values())
        
        # Compute statistics
        min_count = min(counts)
        max_count = max(counts)
        median_count = sorted(counts)[len(counts) // 2]
        mean_count = sum(counts) / len(counts)
        
        # Using 25th percentile as a reasonable threshold for "rare"
        sorted_counts = sorted(counts)
        percentile_25 = sorted_counts[int(len(sorted_counts) * 0.25)]
        
        # Log the statistics
        print(f"Label combination frequency statistics:")
        print(f"  Total unique combinations: {len(counts)}")
        print(f"  Min count: {min_count}")
        print(f"  25th percentile: {percentile_25}")
        print(f"  Median count: {median_count}")
        print(f"  Mean count: {mean_count:.2f}")
        print(f"  Max count: {max_count}")
        
        # Use the 25th percentile as our threshold
        rarity_threshold = percentile_25
        print(f"Using rarity threshold: {rarity_threshold}")
        
        return rarity_threshold
    
    def create_reflected_boards(self, item):
        """
        Creates reflected versions of a chess board and validates them.
        
        Args:
            item (dict): A dataset item containing 'board' tensor and 'fen' string
            
        Returns:
            dict: Contains valid reflected boards with keys:
                'horizontal': horizontally reflected board or None if invalid
                'vertical': vertically reflected board or None if invalid
                'both': board reflected both horizontally and vertically or None if invalid
                Each valid board is represented as a tuple of (board_tensor, fen_string)
        """
        results = {
            'horizontal': None,
            'vertical': None,
            'both': None
        }
        
        # Get original board state
        original_board_tensor = item['board']
        original_fen = item['fen']
        
        # Create horizontal reflection (flip along vertical axis)
        h_board_tensor = torch.flip(original_board_tensor.clone(), dims=[1])
        
        # Create vertical reflection (flip along horizontal axis)
        v_board_tensor = torch.flip(original_board_tensor.clone(), dims=[0])
        
        # Create both horizontal and vertical reflection
        hv_board_tensor = torch.flip(original_board_tensor.clone(), dims=[0, 1])
        
        # Try to convert tensors back to valid board positions
        for reflection_type, board_tensor in [
            ('horizontal', h_board_tensor),
            ('vertical', v_board_tensor),
            ('both', hv_board_tensor)
        ]:
            try:
                # Convert tensor to FEN
                reflected_board = Board()
                # Clear the board first
                reflected_board.clear()
                
                # Map indices back to chess pieces
                idx_to_piece = {
                    0: 'K',   # white king
                    1: 'Q',   # white queen
                    2: 'R',   # white rook
                    3: 'B',   # white bishop
                    4: 'N',   # white knight
                    5: 'P',   # white pawn
                    7: 'p',   # black pawn
                    8: 'n',   # black knight
                    9: 'b',   # black bishop
                    10: 'r',  # black rook
                    11: 'q',  # black queen
                    12: 'k',  # black king
                }
                
                # Place pieces on the board
                for rank in range(8):
                    for file in range(8):
                        piece_idx = int(board_tensor[rank, file].item())
                        if piece_idx != 6:  # Skip empty squares
                            piece_symbol = idx_to_piece[piece_idx]
                            # Convert back to chess.Square (0-63)
                            square = rank * 8 + file
                            # Convert symbol to chess.Piece
                            piece = Piece.from_symbol(piece_symbol)
                            reflected_board.set_piece_at(square, piece)
                
                # The python-chess library will validate the board when we construct 
                # a Board from the FEN string. If the position is illegal, it will 
                # raise a ValueError, which we'll catch.
                
                # Get FEN of the reflected board
                reflected_fen = reflected_board.fen()
                # Extract just the board position part (before the space)
                board_position = reflected_fen.split(' ')[0]
                # Use original FEN for turn, castling, etc.
                original_fen_parts = original_fen.split(' ')
                # Combine reflected board position with original game state
                complete_fen = board_position + ' ' + ' '.join(original_fen_parts[1:])
                
                # Try to validate the complete FEN
                try:
                    Board(complete_fen)  # Validate the FEN by creating a Board
                    results[reflection_type] = (board_tensor, complete_fen)
                except ValueError:
                    # Invalid FEN string
                    pass
            except Exception as e:
                # Any other error during the process
                print(f"Error validating {reflection_type} reflection: {e}")
        
        return results
    

# Add necessary classes to PyTorch's serialization allowlist to avoid FutureWarning
if add_safe_globals is not None:
    add_safe_globals([
        # Core modules and classes
        np, torch, pd, lru_cache, Board, Piece, SQUARES, ChessPuzzleDataset, 
        # Container types
        list, dict, set, frozenset, tuple, 
        # PyTorch tensor types
        torch.Tensor, torch.int8, torch.float32,
        # Basic Python types
        int, float, bool, str
    ])
