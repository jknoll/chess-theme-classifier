#!/usr/bin/env python3
"""Create a model architecture diagram for the README."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_model_diagram(output_path):
    """Create a visual diagram of the model architecture."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    input_color = '#e8f4f8'
    embed_color = '#d4edda'
    conv_color = '#fff3cd'
    attn_color = '#cce5ff'
    fc_color = '#f8d7da'
    output_color = '#e2d5f1'

    # Helper function to draw a box with text
    def draw_box(x, y, width, height, text, color, fontsize=10, text_color='black'):
        box = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight='bold', wrap=True)

    # Helper function to draw an arrow
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

    # Title
    ax.text(7, 9.5, 'Chess Theme Classifier - Model Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # Input layer
    draw_box(0.5, 7.5, 2.5, 1.2, 'Input\n8x8 Board\n(0-12 pieces)', input_color, 9)

    # Embedding
    draw_box(0.5, 5.5, 2.5, 1.2, 'Embedding\n13 -> 64 dim', embed_color, 10)
    draw_arrow(1.75, 7.5, 1.75, 6.7)

    # Permute
    draw_box(0.5, 3.8, 2.5, 1.0, 'Permute\n(B,H,W,C)->(B,C,H,W)', '#f0f0f0', 8)
    draw_arrow(1.75, 5.5, 1.75, 4.8)

    # Residual blocks section
    draw_box(4, 3.5, 5.5, 5.5, '', '#fafafa')
    ax.text(6.75, 8.5, 'Repeated 10x (with increasing dilation: 1,2,4,8,...)',
            ha='center', va='center', fontsize=9, style='italic')

    # Residual block
    draw_box(4.3, 6.5, 4.9, 2.0,
             'Residual Block\nConv1x1 -> BN -> ReLU -> Conv3x3 -> BN\n+ Skip Connection (1x1 conv)',
             conv_color, 9)
    draw_arrow(3, 4.3, 4.3, 7.5)

    # Attention
    draw_box(4.3, 4.0, 4.9, 1.8,
             'Self-Attention\nQ,K,V projections -> Softmax -> Output',
             attn_color, 9)
    draw_arrow(6.75, 6.5, 6.75, 5.8)

    # Dilation label
    ax.text(9.7, 6.0, 'Dilation\n2^i', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Accumulator
    draw_box(10.5, 5.5, 3, 1.2, 'Accumulator\nConv 8x8 -> 1x1', '#e0e0e0', 9)
    draw_arrow(9.5, 4.9, 10.5, 6.1)

    # FC layers
    draw_box(10.5, 3.5, 3, 0.8, 'FC1: 64 -> 512\n+ Dropout', fc_color, 9)
    draw_arrow(12, 5.5, 12, 4.3)

    draw_box(10.5, 2.2, 3, 0.8, 'FC2: 512 -> 256\n+ Dropout', fc_color, 9)
    draw_arrow(12, 3.5, 12, 3.0)

    draw_box(10.5, 0.9, 3, 0.8, 'FC3: 256 -> 1616', fc_color, 9)
    draw_arrow(12, 2.2, 12, 1.7)

    # Output
    draw_box(10.5, -0.5, 3, 0.9, 'Sigmoid\n1616 probabilities', output_color, 9)
    draw_arrow(12, 0.9, 12, 0.4)

    # Parameter counts on the side
    params_text = """Key Parameters:
    - nlayers: 10
    - embed_dim: 64
    - inner_dim: 320
    - attention_dim: 64
    - dropout: 0.5

Total: ~3.2M params"""
    ax.text(0.5, 1.5, params_text, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9),
            family='monospace')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=input_color, edgecolor='gray', label='Input'),
        mpatches.Patch(facecolor=embed_color, edgecolor='gray', label='Embedding'),
        mpatches.Patch(facecolor=conv_color, edgecolor='gray', label='Convolution'),
        mpatches.Patch(facecolor=attn_color, edgecolor='gray', label='Attention'),
        mpatches.Patch(facecolor=fc_color, edgecolor='gray', label='Fully Connected'),
        mpatches.Patch(facecolor=output_color, edgecolor='gray', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Model diagram saved to {output_path}")

if __name__ == '__main__':
    create_model_diagram('analysis/model_architecture_diagram.png')
