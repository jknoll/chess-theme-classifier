import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import ChessPuzzleDataset
from model import PIECE_CHARS

def visualize_board(board_tensor, title):
    """Visualize a chess board tensor with Unicode chess pieces"""
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a light/dark checkerboard pattern
    board_colors = np.zeros((8, 8, 3))
    light_color = np.array([0.9, 0.9, 0.8])  # Light squares
    dark_color = np.array([0.5, 0.6, 0.4])   # Dark squares
    
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                board_colors[i, j] = light_color
            else:
                board_colors[i, j] = dark_color
    
    # Display the board
    ax.imshow(board_colors)
    
    # Map piece indices to Unicode characters
    piece_map = PIECE_CHARS
    
    # Place pieces on the board
    for row in range(8):
        for col in range(8):
            piece_idx = int(board_tensor[row, col].item())
            if piece_idx != 6:  # If not an empty square
                ax.text(col, row, piece_map[piece_idx], 
                       fontsize=32, ha='center', va='center')
    
    # Add grid lines
    for i in range(9):
        ax.axhline(i - 0.5, color='black', linewidth=1)
        ax.axvline(i - 0.5, color='black', linewidth=1)
    
    # Add labels
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticklabels([str(8-i) for i in range(8)])
    
    # Set title
    ax.set_title(title)
    
    return fig

if __name__ == "__main__":
    # Load the small dataset
    dataset = ChessPuzzleDataset('lichess_db_puzzle_small.csv', augment_with_reflections=True)
    
    # Get the original and reflected board
    original = dataset[0]
    reflected = dataset[1]
    
    # Check that they are what we expect
    assert not original['is_reflection']
    assert reflected['is_reflection']
    
    # Print FEN of original board
    print(f"Original Board FEN: {original['fen']}")
    
    # Visualize original board
    fig1 = visualize_board(original['board'], "Original Board")
    fig1.savefig('original_board.png')
    
    # Visualize reflected board
    fig2 = visualize_board(reflected['board'], "Horizontally Reflected Board")
    fig2.savefig('reflected_board.png')
    
    print("Board visualizations saved to original_board.png and reflected_board.png")
    
    # Verify the reflection by comparing specific positions
    print("\nChecking specific positions between original and reflected:")
    
    # Get original and reflected tensors
    orig_tensor = original['board']
    refl_tensor = reflected['board']
    
    # Check some key squares (for the start position)
    key_positions = [
        ((0, 0), (0, 7)), # Rooks
        ((0, 1), (0, 6)), # Knights
        ((0, 2), (0, 5)), # Bishops
        ((0, 3), (0, 4)), # Queen/King
        ((7, 3), (7, 4)), # Queen/King
    ]
    
    for (r1, c1), (r2, c2) in key_positions:
        orig_piece = int(orig_tensor[r1, c1].item())
        refl_piece = int(refl_tensor[r1, c2].item())
        print(f"Original at [{r1},{c1}]: {PIECE_CHARS[orig_piece]} <-> Reflected at [{r1},{c2}]: {PIECE_CHARS[refl_piece]}")