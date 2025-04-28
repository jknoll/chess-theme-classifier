#!/usr/bin/env python3
"""
Chess Board Reflection Visualizer

This script loads the chess puzzle dataset, samples random entries,
performs board reflections (horizontal, vertical, and both),
and displays the results using Unicode chess piece characters.
"""

import random
import chess
from dataset import ChessPuzzleDataset
import torch
from model import PIECE_CHARS

def board_to_unicode(board):
    """
    Convert a chess.Board object to a Unicode string representation.
    
    Args:
        board: A chess.Board object or FEN string
    
    Returns:
        A string representation of the board with Unicode chess pieces
    """
    if isinstance(board, str):
        board = chess.Board(board)
        
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

def display_board_with_label(board, label):
    """
    Display a chess board with a label using Unicode chess pieces.
    
    Args:
        board: A chess.Board object
        label: A label to display above the board
    """
    board_str = board_to_unicode(board)
    
    # Make a horizontal divider line
    divider = "-" * 20
    
    # Add label heading with padding
    formatted_label = f"\n{divider}\n{label}\n{divider}\n"
    
    # Combine label with board
    return formatted_label + board_str


def display_reflection_set(original_board, reflections):
    """
    Display the original board and its reflections side by side.
    
    Args:
        original_board: The original chess.Board object
        reflections: Dictionary of reflection results from create_reflected_boards
    """
    results = []
    
    # Add the original board
    results.append(display_board_with_label(original_board, "Original"))
    
    # Add each reflection if valid
    for reflection_type in ['horizontal', 'vertical', 'both']:
        if reflections[reflection_type]:
            # Extract the FEN from the reflection tuple
            reflected_fen = reflections[reflection_type][1]
            # Create a new board from the FEN
            reflected_board = chess.Board(reflected_fen)
            # Add it to results with appropriate label
            label = f"Reflection: {reflection_type.capitalize()}"
            results.append(display_board_with_label(reflected_board, label))
    
    return "\n".join(results)


def main():
    """
    Main function that loads dataset, samples entries, performs reflections,
    and displays results.
    """
    # Use the test dataset for quicker loading
    dataset_path = "lichess_db_puzzle_test.csv"
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = ChessPuzzleDataset(dataset_path)
    
    # Check the dataset size and adjust sample size if needed
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} puzzles")
    
    # Sample random entries (up to 3, but not more than the dataset size)
    num_samples = min(3, dataset_size)
    print(f"Sampling {num_samples} puzzles...")
    
    # If dataset is empty, handle it gracefully
    if dataset_size == 0:
        print("Dataset is empty. Cannot sample entries.")
        return "Dataset is empty."
        
    sample_indices = random.sample(range(dataset_size), num_samples)
    
    all_results = []
    
    # Process each sample
    for i, idx in enumerate(sample_indices):
        print(f"Processing sample {i+1}/{num_samples}...")
        
        # Get the sample from the dataset
        sample = dataset[idx]
        
        # Create the original chess board from FEN
        original_board = chess.Board(sample['fen'])
        
        # Create reflections
        reflections = dataset.create_reflected_boards(sample)
        
        # Display the original and reflections
        results = display_reflection_set(original_board, reflections)
        all_results.append(f"\n\nSAMPLE {i+1}/{num_samples}\n{results}\n")
    
    # Join all results and print them
    final_output = "\n".join(all_results)
    print("\nRESULTS:\n")
    print(final_output)
    
    return final_output


if __name__ == "__main__":
    main()