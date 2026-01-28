#!/usr/bin/env python3
"""Create a professional model architecture diagram similar to transformer visualizations."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import matplotlib.patheffects as path_effects
import numpy as np

def create_model_diagram(output_path):
    """Create a clean, professional architecture diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_facecolor('white')

    # Professional color palette
    colors = {
        'input': '#E3F2FD',      # Light blue
        'embed': '#C8E6C9',      # Light green
        'residual': '#FFF3E0',   # Light orange
        'attention': '#F3E5F5',  # Light purple
        'fc': '#FFEBEE',         # Light red
        'output': '#E8F5E9',     # Light green
        'border': '#37474F',     # Dark gray
        'arrow': '#546E7A',      # Medium gray
        'text': '#212121',       # Near black
    }

    def draw_block(x, y, w, h, text, color, fontsize=10, subtext=None):
        """Draw a rounded rectangle block with text."""
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color,
            edgecolor=colors['border'],
            linewidth=1.5
        )
        ax.add_patch(rect)

        if subtext:
            ax.text(x, y + 0.15, text, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color=colors['text'])
            ax.text(x, y - 0.25, subtext, ha='center', va='center',
                   fontsize=fontsize-2, color='#666666', style='italic')
        else:
            ax.text(x, y, text, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color=colors['text'])

    def draw_arrow(x1, y1, x2, y2):
        """Draw an arrow between points."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=colors['arrow'],
                                  lw=1.5, shrinkA=3, shrinkB=3))

    def draw_plus(x, y):
        """Draw a plus symbol for residual connection."""
        circle = plt.Circle((x, y), 0.2, fill=True, facecolor='white',
                           edgecolor=colors['border'], linewidth=1.5)
        ax.add_patch(circle)
        ax.text(x, y, '+', ha='center', va='center', fontsize=14,
               fontweight='bold', color=colors['border'])

    # Title
    ax.text(5, 13.5, 'Chess Theme Classifier Architecture', ha='center', va='center',
           fontsize=16, fontweight='bold', color=colors['text'])

    # Input block
    draw_block(5, 12.5, 3, 0.7, 'Input', colors['input'], 11, '8x8 board (0-12)')

    # Embedding
    draw_arrow(5, 12.1, 5, 11.6)
    draw_block(5, 11.2, 3, 0.7, 'Embedding', colors['embed'], 11, '13 -> 64 dim')

    # Reshape
    draw_arrow(5, 10.8, 5, 10.3)
    draw_block(5, 9.9, 3, 0.7, 'Permute', colors['input'], 11, '(B,H,W,C) -> (B,C,H,W)')

    # Repeated blocks container
    rect = FancyBboxPatch(
        (1.5, 4.5), 7, 5,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor='#FAFAFA',
        edgecolor='#BDBDBD',
        linewidth=1,
        linestyle='--'
    )
    ax.add_patch(rect)
    ax.text(5, 9.3, 'x10 layers', ha='center', va='center',
           fontsize=10, style='italic', color='#757575')

    # Residual Block
    draw_arrow(5, 9.5, 5, 8.9)
    draw_block(5, 8.5, 4, 1.0, 'Residual Block', colors['residual'], 11)
    ax.text(5, 8.1, 'Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN', ha='center', va='center',
           fontsize=8, color='#666666')

    # Skip connection visualization
    ax.annotate('', xy=(7.2, 7.5), xytext=(7.2, 8.5),
               arrowprops=dict(arrowstyle='-', color=colors['arrow'],
                              lw=1, connectionstyle='arc3,rad=0'))
    ax.annotate('', xy=(5.3, 7.5), xytext=(7.2, 7.5),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1))
    ax.text(7.5, 8.0, 'skip', ha='left', va='center', fontsize=8, color='#888888')

    # Plus for residual
    draw_arrow(5, 8.0, 5, 7.7)
    draw_plus(5, 7.5)

    # Attention
    draw_arrow(5, 7.3, 5, 6.8)
    draw_block(5, 6.4, 4, 0.8, 'Self-Attention', colors['attention'], 11)
    ax.text(5, 6.0, 'Q, K, V projections (dim=64)', ha='center', va='center',
           fontsize=8, color='#666666')

    # Dilation note
    ax.text(8.0, 7.0, 'dilation = 2^i', ha='left', va='center',
           fontsize=9, color='#888888', style='italic')

    draw_arrow(5, 5.6, 5, 5.1)

    # Outside repeated block
    # Accumulator
    draw_block(5, 4.0, 3.5, 0.7, 'Accumulator', colors['input'], 11, 'Conv 8x8 -> 1x1')

    draw_arrow(5, 3.6, 5, 3.1)

    # FC layers
    draw_block(5, 2.7, 3, 0.6, 'FC + Dropout', colors['fc'], 10, '64 -> 512')
    draw_arrow(5, 2.4, 5, 2.0)
    draw_block(5, 1.7, 3, 0.6, 'FC + Dropout', colors['fc'], 10, '512 -> 256')
    draw_arrow(5, 1.4, 5, 1.0)
    draw_block(5, 0.7, 3, 0.6, 'FC', colors['fc'], 10, '256 -> 1616')

    draw_arrow(5, 0.4, 5, 0.0)

    # Output
    draw_block(5, -0.4, 3, 0.6, 'Sigmoid', colors['output'], 11, '1616 probabilities')

    # Parameter summary on the side
    params = [
        'Parameters:',
        'embed_dim: 64',
        'inner_dim: 320',
        'attention_dim: 64',
        'dropout: 0.5',
        '',
        '~3.2M trainable'
    ]
    for i, p in enumerate(params):
        weight = 'bold' if i == 0 or i == 6 else 'normal'
        ax.text(0.3, 4.0 - i*0.4, p, ha='left', va='center',
               fontsize=9, fontweight=weight, color='#666666')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white',
               pad_inches=0.2)
    plt.close()
    print(f"Model diagram saved to {output_path}")


if __name__ == '__main__':
    create_model_diagram('docs/model_architecture_diagram.png')
