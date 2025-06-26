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

try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False

# Limit OpenMP threads to avoid resource exhaustion
os.environ['OMP_NUM_THREADS'] = '4'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '4'  # Limit MKL threads

class ChessPuzzleDataset(Dataset):
    def __init__(self, csv_file, cache_size=10000, num_workers=None, augment_with_reflections=False, 
                 class_conditional_augmentation=False, rarity_threshold=None, low_memory=False,
                 use_cache=False, verbose_progress=False, full_class_conditional=False, test_resume=None):
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
            full_class_conditional (bool): If True, create full class-conditional cache (Milestone 2).
        """
        self.csv_file = csv_file
        self.cache_size = cache_size
        
        # Set num_workers based on low_memory flag and explicit parameter
        if num_workers is not None:
            self.num_workers = num_workers  # Use explicitly provided value (including 0)
        elif low_memory:
            self.num_workers = 1  # Use just 1 worker in low memory mode
        else:
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)
            
        self.low_memory = low_memory
        self.augment_with_reflections = augment_with_reflections
        self.class_conditional_augmentation = class_conditional_augmentation
        self.use_cache = use_cache  # Store the use_cache parameter
        self.verbose_progress = verbose_progress  # Store verbose progress flag
        self.full_class_conditional = full_class_conditional  # Store full conditional flag
        self.test_resume = test_resume  # Store test resume parameter for testing
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
                if self.full_class_conditional:
                    # Full class-conditional mode (Milestone 2)
                    essential_cache_files.append(f"{self.tensors_cache_file}_conditional_full")
                    essential_cache_files.append(f"{self.tensors_cache_file}_conditional_full.augmented_indices.json")
                else:
                    # Original partial class-conditional mode
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
                if self.full_class_conditional:
                    # Full class-conditional mode (Milestone 2)
                    essential_cache_files.append(f"{self.tensors_cache_file}_conditional_full")
                    essential_cache_files.append(f"{self.tensors_cache_file}_conditional_full.augmented_indices.json")
                else:
                    # Original partial class-conditional mode
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
            chunk_size = 500  # Moderate chunks for low memory mode - balance memory vs efficiency
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
        # In low memory mode with many chunks, use batched processing to avoid overwhelming the process queue
        max_pending_futures = 100 if self.low_memory else 1000
        
        # If num_workers is 0, use single-threaded processing to avoid multiprocessing issues
        if self.num_workers == 0:
            if self.verbose_progress:
                print("Using single-threaded processing to avoid multiprocessing issues")
            return self._sequential_batch_conversion(fen_list, augment_with_reflections)
        
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
                
                # Process chunks in batches to avoid overwhelming the process queue
                results = []
                failed_chunks = []
                
                # Process chunks in batches
                total_chunks = len(chunk_args)
                processed_count = 0
                
                if tqdm_available and self.verbose_progress:
                    overall_progress = tqdm(total=total_chunks, desc="Processing chunks (with reflections)")
                elif self.verbose_progress:
                    print("Processing chunks (with reflections)...")
                
                for batch_start in range(0, total_chunks, max_pending_futures):
                    batch_end = min(batch_start + max_pending_futures, total_chunks)
                    batch_chunk_args = chunk_args[batch_start:batch_end]
                    
                    # Submit batch of futures
                    futures = []
                    for chunk_arg in batch_chunk_args:
                        future = executor.submit(self._process_chunk_with_args_wrapper, chunk_arg)
                        futures.append(future)
                    
                    # Process results from this batch
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            result = future.result(timeout=120)
                            results.append(result)
                            processed_count += 1
                            
                            if tqdm_available and self.verbose_progress:
                                overall_progress.update(1)
                            elif self.verbose_progress and processed_count % 1000 == 0:
                                print(f"Processed {processed_count:,}/{total_chunks:,} chunks...")
                                
                        except concurrent.futures.TimeoutError:
                            print(f"⚠️ Warning: A worker process timed out. Will retry this chunk.")
                            failed_chunks.append(batch_chunk_args[i])
                        except Exception as e:
                            print(f"⚠️ Error processing chunk: {e}. Will retry this chunk.")
                            failed_chunks.append(batch_chunk_args[i])
                
                if tqdm_available and self.verbose_progress:
                    overall_progress.close()
                            
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
                
                # Process chunks in batches to avoid overwhelming the process queue
                results = []
                failed_chunks = []
                
                # Process chunks in batches
                total_chunks = len(chunks)
                processed_count = 0
                
                if tqdm_available and self.verbose_progress:
                    overall_progress = tqdm(total=total_chunks, desc="Processing chunks")
                elif self.verbose_progress:
                    print("Processing chunks...")
                
                # Initial memory logging
                if self.verbose_progress and psutil_available:
                    mem = psutil.virtual_memory()
                    current_proc = psutil.Process()
                    child_count = len(current_proc.children(recursive=True))
                    print(f"Initial memory: {mem.percent:.1f}% used ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB)")
                    print(f"Initial process count: {child_count} child processes")
                
                for batch_start in range(0, total_chunks, max_pending_futures):
                    batch_end = min(batch_start + max_pending_futures, total_chunks)
                    batch_chunks = chunks[batch_start:batch_end]
                    
                    # Submit batch of futures
                    futures = []
                    for chunk in batch_chunks:
                        future = executor.submit(self._process_chunk, chunk)
                        futures.append(future)
                    
                    # Process results from this batch
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            result = future.result(timeout=120)
                            results.append(result)
                            processed_count += 1
                            
                            if tqdm_available and self.verbose_progress:
                                overall_progress.update(1)
                            elif self.verbose_progress and processed_count % 1000 == 0:
                                print(f"Processed {processed_count:,}/{total_chunks:,} chunks...")
                            
                            # Memory logging every 100 chunks
                            if self.verbose_progress and psutil_available and processed_count % 100 == 0:
                                mem = psutil.virtual_memory()
                                if not tqdm_available:  # Only print if not using tqdm to avoid cluttering
                                    print(f"Memory: {mem.percent:.1f}% used ({mem.used/1024**3:.1f}GB)")
                                # Check if memory usage is getting high
                                if mem.percent > 80:
                                    print(f"⚠️ High memory usage: {mem.percent:.1f}%")
                                
                        except concurrent.futures.TimeoutError:
                            print(f"⚠️ Warning: A worker process timed out. Will retry this chunk.")
                            failed_chunks.append(batch_chunks[i])
                        except Exception as e:
                            print(f"⚠️ Error processing chunk: {e}. Will retry this chunk.")
                            failed_chunks.append(batch_chunks[i])
                
                if tqdm_available and self.verbose_progress:
                    overall_progress.close()
                            
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
    
    def _sequential_batch_conversion(self, fen_list, augment_with_reflections):
        """Sequential (single-threaded) batch conversion to avoid multiprocessing issues."""
        print(f"Processing {len(fen_list):,} FENs sequentially")
        
        # Process chunks sequentially
        if self.low_memory:
            chunk_size = 500  # Same as parallel version
        elif len(fen_list) > 1000000:
            chunk_size = 1000
        else:
            chunk_size = 5000  # Larger chunks for sequential processing
            
        chunk_size = min(chunk_size, 10000)
        chunks = [fen_list[i:i+chunk_size] for i in range(0, len(fen_list), chunk_size)]
        
        print(f"Processing {len(fen_list):,} FENs in {len(chunks)} chunks sequentially")
        
        all_tensors = []
        
        # Memory monitoring setup
        if self.verbose_progress and psutil_available:
            mem = psutil.virtual_memory()
            print(f"Initial memory: {mem.percent:.1f}% used ({mem.used/1024**3:.1f}GB/{mem.total/1024**3:.1f}GB)")
        
        # Progress bar setup
        if self.verbose_progress:
            try:
                from tqdm import tqdm
                chunk_iterator = tqdm(chunks, desc="Processing chunks sequentially")
            except ImportError:
                chunk_iterator = chunks
                print("Processing chunks sequentially...")
        else:
            chunk_iterator = chunks
        
        processed_count = 0
        for chunk in chunk_iterator:
            try:
                if augment_with_reflections:
                    result = self._process_chunk_with_args(chunk, True)
                else:
                    result = self._process_chunk(chunk)
                
                if result is not None and len(result) > 0:
                    all_tensors.extend(result)
                
                processed_count += 1
                
                # Memory monitoring
                if self.verbose_progress and psutil_available and processed_count % 100 == 0:
                    mem = psutil.virtual_memory()
                    if mem.percent > 80:
                        print(f"⚠️ High memory usage: {mem.percent:.1f}%")
                        
            except Exception as e:
                print(f"⚠️ Error processing chunk: {e}")
                continue
        
        # Detailed reporting like parallel version
        expected_count = len(fen_list)
        actual_count = len(all_tensors)
        success_rate = (actual_count / expected_count) * 100 if expected_count > 0 else 0
        
        print(f"📊 Sequential Processing Summary:")
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
        # Define cache file with conditional suffix (full or regular)
        conditional_suffix = '_conditional_full' if self.full_class_conditional else '_conditional'
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
                        
                        # Check if cache is empty and force recreation in test mode
                        if len(self.tensor_cache) == 0 and hasattr(self, 'test_resume') and self.test_resume:
                            print("Cache is empty and test_resume mode is active, forcing recreation...")
                            # Remove empty cache files to force recreation
                            os.remove(tensors_cache_file)
                            if os.path.exists(augmented_indices_file):
                                os.remove(augmented_indices_file)
                            # Clear cache and proceed to creation
                            self.tensor_cache = None
                            self.label_cache = None
                            self.augmented_indices = set()
                        elif len(self.tensor_cache) == 0:
                            print("Warning: Cache file exists but is empty")
                            return
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
                
                # Check if cache is empty and force recreation
                if len(self.tensor_cache) == 0:
                    print("Cache is empty, forcing recreation...")
                    # Remove empty cache files to force recreation
                    os.remove(tensors_cache_file)
                    if os.path.exists(augmented_indices_file):
                        os.remove(augmented_indices_file)
                    raise Exception("Empty cache detected, forcing recreation")
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
        
        # Check for existing checkpoints/segments to resume from FIRST
        conditional_suffix = '_conditional_full' if self.full_class_conditional else '_conditional'
        tensors_cache_file = f"{self.tensors_cache_file}{conditional_suffix}"
        
        # Look for existing segment files to resume from
        import glob
        existing_segments = glob.glob(f"{tensors_cache_file}.segment_*")
        existing_chunks = glob.glob(f"/tmp/chess_cache_*/chunk_*.pt")
        
        # Filter out empty chunk files
        existing_chunks = [chunk for chunk in existing_chunks if os.path.exists(chunk) and os.path.getsize(chunk) > 0]
        
        if existing_segments:
            print(f"Found {len(existing_segments)} existing segments, resuming from segments...")
            # Skip chunk creation and go directly to segment combination
            segment_files = sorted(existing_segments)
            temp_dir = None  # No temp dir needed
            
            # Use streaming combination to avoid OOM
            total_entries = self._stream_combine_segments(segment_files, tensors_cache_file)
            
            # Save augmented indices (need to reconstruct for segments resume)
            augmented_indices_file = f"{tensors_cache_file}.augmented_indices.json"
            with open(augmented_indices_file, 'w') as f:
                json.dump(list(self.augmented_indices), f)
            
            print(f"Resume completed! Final cache has {total_entries:,} entries")
            return
        
        elif existing_chunks:
            print(f"Found {len(existing_chunks)} existing chunk files, resuming chunk processing...")
            # Find the temp directory and continue from where we left off
            temp_dirs = set(os.path.dirname(chunk) for chunk in existing_chunks)
            if len(temp_dirs) == 1:
                temp_dir = list(temp_dirs)[0]
                chunk_files = sorted(existing_chunks)
                print(f"Resuming with {len(chunk_files)} existing chunks in {temp_dir}")
                # Skip chunk creation and go to segment processing
                resume_from_chunks = True
            else:
                print("Multiple temp directories found, consolidating all chunks...")
                # Use all chunks from all directories
                chunk_files = sorted(existing_chunks)
                temp_dir = os.path.dirname(chunk_files[0]) if chunk_files else None
                print(f"Consolidating {len(chunk_files)} chunks from {len(temp_dirs)} directories")
                resume_from_chunks = True
        else:
            # Start fresh
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix='chess_cache_')
            chunk_files = []
            resume_from_chunks = False
        
        # Create basic tensors if not resuming from chunks/segments
        if not resume_from_chunks and not existing_segments:
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
        else:
            # When resuming from chunks, we don't need basic_tensors but we need valid_indices
            valid_indices = len(self.puzzle_data)
            basic_tensors = None  # Will not be used when resuming
        
        # Only proceed with chunk creation if we don't have existing chunks to resume from
        if not resume_from_chunks:
            # Map from original index to tensor cache index
            original_to_cache_idx = {}
            next_tensor_idx = 0
            
            # Process in very small chunks to manage memory
            chunk_size = 2000  # Process 2K entries at a time
            current_chunk = 0
            
            print("Starting chunk creation from scratch...")
            
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
            
            chunk_tensors = []
            chunk_labels = []
            processed_count = 0
            
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
                
                # Add the original tensor and label vector to chunk
                chunk_tensors.append(basic_tensors[idx])
                chunk_labels.append(label_vector)
                original_to_cache_idx[idx] = next_tensor_idx
                next_tensor_idx += 1
                processed_count += 1
                
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
                    
                    # Add the reflected tensor and label vector to chunk
                    chunk_tensors.append(reflected_tensor)
                    chunk_labels.append(reflection_label_vector)
                    
                    # Track that this index was augmented
                    self.augmented_indices.add(idx)
                    next_tensor_idx += 1
                    processed_count += 1
                
                # Save chunk to disk when it reaches the desired size
                if len(chunk_tensors) >= chunk_size:
                    # Save chunk to temporary file
                    chunk_file = os.path.join(temp_dir, f'chunk_{current_chunk:06d}.pt')
                    torch.save({
                        'tensors': chunk_tensors,
                        'labels': chunk_labels
                    }, chunk_file)
                    chunk_files.append(chunk_file)
                    current_chunk += 1
                    
                    # Clear chunk and force garbage collection
                    chunk_tensors.clear()
                    chunk_labels.clear()
                    import gc
                    gc.collect()
                    
                    if self.verbose_progress:
                        try:
                            import psutil
                            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                            print(f"Memory usage: {memory_usage:.1f} GB, saved chunk {current_chunk-1}, processed {processed_count:,} entries")
                        except ImportError:
                            print(f"Saved chunk {current_chunk-1}, processed {processed_count:,} entries")
                    
                    # Test resume: abort after creating some chunks
                    if self.test_resume == "chunks" and current_chunk >= 10:
                        print(f"TEST RESUME: Aborting after creating {current_chunk} chunks for testing")
                        print(f"Chunk files saved in: {temp_dir}")
                        print("Re-run the script to test chunk resume functionality")
                        import sys
                        sys.exit(0)  # Clean exit instead of exception
            
            # Save any remaining items in the final chunk
            if chunk_tensors:
                chunk_file = os.path.join(temp_dir, f'chunk_{current_chunk:06d}.pt')
                torch.save({
                    'tensors': chunk_tensors,
                    'labels': chunk_labels
                }, chunk_file)
                chunk_files.append(chunk_file)
                chunk_tensors.clear()
                chunk_labels.clear()
            
            # Clear the basic_tensors list to free memory
            if basic_tensors is not None:
                basic_tensors.clear()
        else:
            # If resuming from existing chunks, clear basic_tensors as well
            if basic_tensors is not None:
                basic_tensors.clear()
        
        # True streaming approach - build final file in segments to avoid memory accumulation
        print(f"Streaming {len(chunk_files)} chunks directly to final cache file...")
        
        # Initialize counters
        total_entries = 0
        segment_tensors = []
        segment_labels = []
        segment_files = []
        
        # Process chunks in very small segments that get saved immediately  
        segment_size = 10  # Save every 10 chunks (20K entries) to avoid memory buildup
        
        for i, chunk_file in enumerate(chunk_files):
            if self.verbose_progress:
                try:
                    import psutil
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                    print(f"Memory: {memory_usage:.1f} GB, processing chunk {i+1}/{len(chunk_files)}")
                except ImportError:
                    print(f"Processing chunk {i+1}/{len(chunk_files)}")
            
            # Load single chunk
            chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
            chunk_tensors = chunk_data['tensors']
            chunk_labels = chunk_data['labels']
            total_entries += len(chunk_tensors)
            
            # Add to current segment
            segment_tensors.extend(chunk_tensors)
            segment_labels.extend(chunk_labels)
            
            # Clear chunk data immediately
            del chunk_data, chunk_tensors, chunk_labels
            
            # Remove chunk file to save disk space
            os.remove(chunk_file)
            
            # Save segment every segment_size chunks or at the end
            if (i + 1) % segment_size == 0 or i == len(chunk_files) - 1:
                segment_num = (i // segment_size) + 1
                segment_file = f"{tensors_cache_file}.segment_{segment_num:04d}"
                
                if self.verbose_progress:
                    print(f"Saving segment {segment_num} with {len(segment_tensors):,} entries")
                
                torch.save({
                    'tensors': segment_tensors,
                    'labels': segment_labels,
                    'segment_num': segment_num,
                    'format_version': 2
                }, segment_file)
                
                segment_files.append(segment_file)
                
                # Clear segment data to free memory
                segment_tensors.clear()
                segment_labels.clear()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                if self.verbose_progress:
                    try:
                        import psutil
                        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                        print(f"Memory after segment save: {memory_usage:.1f} GB")
                    except ImportError:
                        print(f"Segment {segment_num} saved with {total_entries:,} total entries so far")
                
                # Test resume: abort after creating some segments
                if self.test_resume == "segments" and segment_num >= 3:
                    print(f"TEST RESUME: Aborting after creating {segment_num} segments for testing")
                    print(f"Segment files: {tensors_cache_file}.segment_*")
                    print("Re-run the script to test segment resume functionality")
                    import sys
                    sys.exit(0)  # Clean exit instead of exception
        
        # Use streaming combination to avoid OOM
        total_entries = self._stream_combine_segments(segment_files, tensors_cache_file)
        
        # Final memory report
        if self.verbose_progress:
            try:
                import psutil
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                print(f"Streaming combination complete: {final_memory:.1f} GB, {total_entries:,} total entries")
            except ImportError:
                print(f"Streaming combination complete: {total_entries:,} total entries")
        
        # Test resume: abort just before final save
        if self.test_resume == "final":
            print(f"TEST RESUME: Aborting just before saving final cache for testing")
            print(f"Final tensors ready: {total_entries:,} entries")
            print("Re-run the script to test final resume functionality")
            import sys
            sys.exit(0)  # Clean exit instead of exception
        
        # Clean up temporary directory
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Note: Final cache is already saved by _stream_combine_segments method
        
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
        # For full class-conditional mode, analyze the entire dataset for accurate rare combination identification
        if self.full_class_conditional:
            print(f"🔍 Full class-conditional mode: Analyzing complete dataset of {len(self.puzzle_data):,} puzzles for rare combinations.")
            puzzle_subset = self.puzzle_data
        else:
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
    
    def _stream_combine_segments(self, segment_files, output_file):
        """
        Stream-combine segment files into final cache without loading all data into memory.
        This avoids OOM issues by writing the final cache incrementally.
        
        Args:
            segment_files (list): List of segment file paths to combine
            output_file (str): Path where the final combined cache will be saved
            
        Returns:
            int: Total number of entries in the combined cache
        """
        print(f"Stream-combining {len(segment_files)} segments into final cache...")
        
        # Initialize counters
        total_entries = 0
        
        # Prepare the output data structure
        combined_data = {
            'tensors': [],
            'labels': [],
            'format_version': 2
        }
        
        # Process segments one by one with minimal memory footprint
        for i, segment_file in enumerate(segment_files):
            if self.verbose_progress:
                print(f"Processing segment {i+1}/{len(segment_files)}: {os.path.basename(segment_file)}")
            
            try:
                # Load segment with minimal memory footprint
                segment_data = torch.load(segment_file, map_location='cpu', weights_only=False)
                
                # Extend the combined data directly (still in memory but more controlled)
                combined_data['tensors'].extend(segment_data['tensors'])
                combined_data['labels'].extend(segment_data['labels'])
                
                # Update counter
                segment_size = len(segment_data['tensors'])
                total_entries += segment_size
                
                # Clear segment data immediately
                del segment_data
                
                # Remove segment file to free disk space
                os.remove(segment_file)
                
                # Force garbage collection every segment
                import gc
                gc.collect()
                
                # Memory monitoring
                if self.verbose_progress and i % 5 == 0:
                    try:
                        import psutil
                        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                        print(f"  Memory usage: {memory_usage:.1f} GB, processed {total_entries:,} entries so far")
                    except ImportError:
                        print(f"  Processed {total_entries:,} entries so far")
                
            except Exception as e:
                print(f"Error processing segment {segment_file}: {e}")
                continue
        
        # True streaming: Write in batches to avoid memory accumulation
        print(f"Writing {total_entries:,} entries in memory-efficient batches...")
        batch_size = 50000  # Write every 50K entries
        temp_files = []
        
        # Process accumulated data in batches
        for batch_start in range(0, len(combined_data['tensors']), batch_size):
            batch_end = min(batch_start + batch_size, len(combined_data['tensors']))
            batch_tensors = combined_data['tensors'][batch_start:batch_end]
            batch_labels = combined_data['labels'][batch_start:batch_end]
            
            # Convert batch to tensors and save to temp file
            temp_file = f"{output_file}.temp_{len(temp_files):04d}.pt"
            batch_data = {
                'tensors': torch.stack(batch_tensors),
                'labels': torch.stack(batch_labels),
                'format_version': 2
            }
            torch.save(batch_data, temp_file)
            temp_files.append(temp_file)
            
            if self.verbose_progress:
                print(f"  Wrote batch {len(temp_files)} with {len(batch_tensors):,} entries")
            
            # Clear batch data
            del batch_tensors, batch_labels, batch_data
            import gc
            gc.collect()
        
        # Clear the combined data to free memory
        del combined_data
        import gc
        gc.collect()
        
        # Now combine temp files using torch.cat for efficient concatenation
        print(f"Combining {len(temp_files)} temp batches into final cache...")
        all_tensors = []
        all_labels = []
        
        for i, temp_file in enumerate(temp_files):
            if self.verbose_progress and i % 5 == 0:
                try:
                    import psutil
                    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024
                    print(f"  Loading temp file {i+1}/{len(temp_files)}, memory: {memory_usage:.1f} GB")
                except ImportError:
                    print(f"  Loading temp file {i+1}/{len(temp_files)}")
            
            temp_data = torch.load(temp_file, map_location='cpu', weights_only=False)
            all_tensors.append(temp_data['tensors'])
            all_labels.append(temp_data['labels'])
            del temp_data
            os.remove(temp_file)  # Clean up immediately
        
        # Final concatenation
        print("Final tensor concatenation...")
        final_tensors = torch.cat(all_tensors, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        del all_tensors, all_labels
        
        # Save final cache
        print(f"Saving final cache to: {output_file}")
        final_data = {
            'tensors': final_tensors,
            'labels': final_labels,
            'format_version': 2
        }
        torch.save(final_data, output_file)
        
        # Set instance variables for compatibility
        self.tensor_cache = final_tensors
        self.label_cache = final_labels
        
        print(f"✅ Stream combination complete: {total_entries:,} entries saved")
        return total_entries
    
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
