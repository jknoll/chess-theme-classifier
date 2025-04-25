import torch
from torch.utils.data import Dataset
import pandas as pd
from chess import Board
import numpy as np
import os
import json
import time

class ChessPuzzleDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file with chess puzzles
        """
        self.csv_file = csv_file
        self.puzzle_data = pd.read_csv(csv_file)
        
        # Cache file paths
        # cache_dir = os.path.dirname(csv_file) or '.'
        cache_dir = '.' # /data/ is a read-only filesystem
        self.themes_cache_file = os.path.join(cache_dir, f"{os.path.basename(csv_file)}.themes.json")
        self.openings_cache_file = os.path.join(cache_dir, f"{os.path.basename(csv_file)}.openings.json")
        
        # Load or create theme and opening tag caches
        self._load_or_create_caches()
        
        # Combine and sort all labels
        self.all_labels = sorted(list(self.all_themes) + list(self.all_opening_tags))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
    
    def _load_or_create_caches(self):
        """Load themes and opening tags from cache files if they exist and are newer than the CSV,
        otherwise extract them from the CSV and save to cache files."""
        
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
            with open(self.themes_cache_file, 'r') as f:
                self.all_themes = set(json.load(f))
            with open(self.openings_cache_file, 'r') as f:
                self.all_opening_tags = set(json.load(f))
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
        
    def __len__(self):
        return len(self.puzzle_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get FEN, themes, and opening tags for the puzzle
        fen = self.puzzle_data.iloc[idx]['FEN']
        themes = self.puzzle_data.iloc[idx]['Themes'].split()
        opening_tags = []
        if pd.notna(self.puzzle_data.iloc[idx]['OpeningTags']):
            opening_tags = self.puzzle_data.iloc[idx]['OpeningTags'].split()
        
        # Create one-hot encoding for all labels (themes + opening tags)
        label_vector = torch.zeros(len(self.all_labels), dtype=torch.float32)
        for theme in themes:
            label_vector[self.label_to_idx[theme]] = 1
        for tag in opening_tags:
            label_vector[self.label_to_idx[tag]] = 1
            
        # Convert FEN to board 
        board = Board(fen)
        # Create an 8x8 tensor representation with piece values 0-12
        board_tensor = self._board_to_tensor(board).to(dtype=torch.float32)
        
        return {
            'board': board_tensor,
            'themes': label_vector,
            'fen': fen  # Include original FEN for reference
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

    
