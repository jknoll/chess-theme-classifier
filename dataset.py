import torch
from torch.utils.data import Dataset
import pandas as pd
from chess import Board
import numpy as np

class ChessPuzzleDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file with chess puzzles
        """
        self.puzzle_data = pd.read_csv(csv_file)
        # Get unique themes across all puzzles
        self.all_themes = set()
        for themes_str in self.puzzle_data['Themes']:
            themes_list = themes_str.split()
            self.all_themes.update(themes_list)
        self.all_themes = sorted(list(self.all_themes))
        self.theme_to_idx = {theme: idx for idx, theme in enumerate(self.all_themes)}
        
    def __len__(self):
        return len(self.puzzle_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get FEN and themes for the puzzle
        fen = self.puzzle_data.iloc[idx]['FEN']
        themes = self.puzzle_data.iloc[idx]['Themes'].split()
        
        # Create one-hot encoding for themes
        theme_vector = torch.zeros(len(self.all_themes), dtype=torch.float32)
        for theme in themes:
            theme_vector[self.theme_to_idx[theme]] = 1
            
        # Convert FEN to board representation
        board = Board(fen)
        # Create an 8x8 tensor representation with piece values 0-12
        board_tensor = self._board_to_tensor(board).to(dtype=torch.float32)
        
        return {
            'board': board_tensor,
            'themes': theme_vector,
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
        """Return the list of all possible themes."""
        return self.all_themes

    
