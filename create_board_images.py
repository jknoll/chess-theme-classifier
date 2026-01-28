#!/usr/bin/env python3
"""Create aesthetic chessboard visualizations similar to lichess.org style."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Unicode chess pieces
PIECE_SYMBOLS = {
    'K': '\u2654', 'Q': '\u2655', 'R': '\u2656', 'B': '\u2657', 'N': '\u2658', 'P': '\u2659',
    'k': '\u265A', 'q': '\u265B', 'r': '\u265C', 'b': '\u265D', 'n': '\u265E', 'p': '\u265F',
}

# Lichess-style colors
LIGHT_SQUARE = '#f0d9b5'  # Lichess light square
DARK_SQUARE = '#b58863'   # Lichess dark square


def fen_to_board(fen):
    """Convert FEN string to 8x8 board array."""
    board = []
    rows = fen.split()[0].split('/')
    for row in rows:
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend(['.'] * int(char))
            else:
                board_row.append(char)
        board.append(board_row)
    return board


def draw_board(board, output_path, title=None):
    """Draw a chess board with pieces in lichess style."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw squares
    for row in range(8):
        for col in range(8):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            rect = patches.Rectangle((col, 7 - row), 1, 1,
                                     linewidth=0, facecolor=color)
            ax.add_patch(rect)

    # Draw pieces
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != '.':
                symbol = PIECE_SYMBOLS.get(piece, '')
                # Use black color for all pieces (the unicode chars have built-in fill)
                ax.text(col + 0.5, 7 - row + 0.5, symbol,
                       fontsize=42, ha='center', va='center',
                       fontfamily='DejaVu Sans')

    # Add coordinates (lichess style)
    for i in range(8):
        # File labels (a-h)
        ax.text(i + 0.5, -0.3, chr(ord('a') + i),
               fontsize=12, ha='center', va='center', color='#666666')
        # Rank labels (1-8)
        ax.text(-0.3, i + 0.5, str(i + 1),
               fontsize=12, ha='center', va='center', color='#666666')

    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white',
                pad_inches=0.1)
    plt.close()
    print(f"Saved: {output_path}")


def mirror_board(board):
    """Mirror a board horizontally (flip files a<->h)."""
    return [row[::-1] for row in board]


def main():
    # Example position: A tactical puzzle position
    # Using a position with a back rank mate theme
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR"

    board = fen_to_board(fen)
    mirrored = mirror_board(board)

    draw_board(board, 'original_board.png', 'Original Position')
    draw_board(mirrored, 'reflected_board.png', 'Horizontally Reflected')


if __name__ == '__main__':
    main()
