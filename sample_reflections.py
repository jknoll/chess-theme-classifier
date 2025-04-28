import torch
import pandas as pd
import numpy as np
import os
from dataset import ChessPuzzleDataset
from chess import Board
import random
from model import PIECE_CHARS

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

def main():
    # Create a small test dataset for demonstrating reflections
    data = {
        'FEN': [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',  # Initial position
            'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',  # Ruy Lopez position
            'r3k2r/ppp2ppp/2n1bn2/3pp3/q3P3/2N2N2/PPPB1PPP/R2QK2R w KQkq - 4 9'   # Complex middle game
        ],
        'Themes': ['opening opening opening'],
        'OpeningTags': ['e4 e4 e4']
    }
    
    # Fix the theme and opening format for the sample data
    data['Themes'] = ['opening'] * 3
    data['OpeningTags'] = ['e4'] * 3
    
    df = pd.DataFrame(data)
    csv_file = 'sample_reflections.csv'
    df.to_csv(csv_file, index=False)
    
    print("Loading dataset...")
    dataset = ChessPuzzleDataset(csv_file)
    
    # Process each position
    for idx in range(len(dataset)):
        item = dataset[idx]
        original_fen = item['fen']
        
        # Get reflections
        reflections = dataset.create_reflected_boards(item)
        
        # Display original board
        print("\n" + "="*50)
        print(f"Position {idx+1}")
        print("="*50)
        
        print("\nORIGINAL BOARD:")
        print("--------------")
        print(board_to_unicode(original_fen))
        print(f"FEN: {original_fen}")
        
        # Display reflections
        for reflection_type, result in reflections.items():
            if result is not None:
                reflected_board, reflected_fen = result
                print(f"\n{reflection_type.upper()} REFLECTION:")
                print("-" * (len(reflection_type) + 12))
                print(board_to_unicode(reflected_fen))
                print(f"FEN: {reflected_fen}")
            else:
                print(f"\n{reflection_type.upper()} REFLECTION: Invalid")
    
    # Clean up
    os.remove(csv_file)
    if os.path.exists(f"{csv_file}.themes.json"):
        os.remove(f"{csv_file}.themes.json")
    if os.path.exists(f"{csv_file}.openings.json"):
        os.remove(f"{csv_file}.openings.json")

if __name__ == "__main__":
    main()