import pytest
import torch
import pandas as pd
import numpy as np
import os
from dataset import ChessPuzzleDataset
from chess import Board
from model import PIECE_CHARS

@pytest.fixture
def test_dataset():
    # Create a small test dataset
    data = {
        'FEN': ['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],  # Initial chess position
        'Themes': ['opening'],
        'OpeningTags': ['e4']
    }
    df = pd.DataFrame(data)
    csv_file = 'test_reflection.csv'
    df.to_csv(csv_file, index=False)
    
    # Create the dataset
    dataset = ChessPuzzleDataset(csv_file)
    
    yield dataset, csv_file
    
    # Clean up
    if os.path.exists(csv_file):
        os.remove(csv_file)
    if os.path.exists(f"{csv_file}.themes.json"):
        os.remove(f"{csv_file}.themes.json")
    if os.path.exists(f"{csv_file}.openings.json"):
        os.remove(f"{csv_file}.openings.json")

def board_to_unicode(fen):
    """Convert FEN to a Unicode chess board representation using PIECE_CHARS"""
    board = Board(fen)
    # Mapping from piece symbols to the Unicode characters in PIECE_CHARS
    symbol_to_unicode = {
        'K': PIECE_CHARS[0],  # white king
        'Q': PIECE_CHARS[1],  # white queen
        'R': PIECE_CHARS[2],  # white rook
        'B': PIECE_CHARS[3],  # white bishop
        'N': PIECE_CHARS[4],  # white knight
        'P': PIECE_CHARS[5],  # white pawn
        '.': PIECE_CHARS[6],  # empty
        'p': PIECE_CHARS[7],  # black pawn
        'n': PIECE_CHARS[8],  # black knight
        'b': PIECE_CHARS[9],  # black bishop
        'r': PIECE_CHARS[10], # black rook
        'q': PIECE_CHARS[11], # black queen
        'k': PIECE_CHARS[12]  # black king
    }
    
    # Create the Unicode representation
    unicode_board = ""
    ranks = str(board).split('\n')
    
    # Add rank numbers and board borders
    unicode_board += "  +-----------------+\n"
    
    for i, rank in enumerate(ranks):
        # Replace pieces with Unicode symbols
        row = f"{8-i} |"
        for char in rank:
            if char in symbol_to_unicode:
                row += " " + symbol_to_unicode[char]
            elif char == " ":
                # Skip spaces in the original representation
                continue
            else:
                row += " " + char
        unicode_board += row + " |\n"
    
    unicode_board += "  +-----------------+\n"
    unicode_board += "    a b c d e f g h"
    
    return unicode_board

def test_reflection_functions(test_dataset):
    dataset, _ = test_dataset
    
    # Get the first item
    item = dataset[0]
    
    # Test the reflection function
    reflections = dataset.create_reflected_boards(item)
    
    # Check that we have reflections dictionary
    assert reflections is not None
    assert isinstance(reflections, dict)
    
    # Print the original board and reflections using Unicode representation
    print("\nORIGINAL BOARD:")
    print(board_to_unicode(item['fen']))
    
    # Check that the expected reflection types are present in the dictionary
    expected_reflection_types = ['horizontal', 'vertical', 'both']
    for reflection_type in expected_reflection_types:
        assert reflection_type in reflections
    
    # For each valid reflection, check that the tensor and FEN are valid
    for reflection_type, result in reflections.items():
        if result is not None:  # Some reflections might be invalid
            board_tensor, fen = result
            
            # Check tensor shape
            assert isinstance(board_tensor, torch.Tensor)
            assert board_tensor.shape == item['board'].shape
            
            # Check FEN is a string
            assert isinstance(fen, str)
            
            # Print the reflected board using Unicode representation
            print(f"\n{reflection_type.upper()} REFLECTION:")
            print(board_to_unicode(fen))
            
            # For a standard chess position, all reflections should be valid
            if item['fen'] == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1':
                assert result is not None
    
    # Test specific properties of horizontal reflection
    h_result = reflections['horizontal']
    assert h_result is not None
    h_board, h_fen = h_result
    
    # In horizontal reflection of starting position, pieces should be mirrored left-to-right
    # e.g., king and queen should swap sides
    original_board = item['board']
    # Check that h_board[0,3] and h_board[0,4] are swapped compared to original
    assert h_board[7,3].item() != original_board[7,3].item()
    assert h_board[7,4].item() != original_board[7,4].item()
    
    # Test vertical reflection
    v_result = reflections['vertical']
    assert v_result is not None
    
    # Test both reflections
    both_result = reflections['both']
    assert both_result is not None