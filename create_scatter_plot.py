#!/usr/bin/env python3
"""Create an aesthetically improved F1 vs Support scatter plot."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def create_scatter_plot(pr_curves_csv, output_path):
    """Create a polished F1 vs Support scatter plot.

    Args:
        pr_curves_csv: Path to per_class_pr_curves.csv
        output_path: Output path for the scatter plot
    """
    # Load data
    df = pd.read_csv(pr_curves_csv)

    # Get max F1 and support for each class
    # First get max F1 per class
    max_f1_df = df.groupby('class_name').agg({
        'f1': 'max',
        'tp': 'max',  # Use as proxy for support
        'class_index': 'first'
    }).reset_index()

    # Load per_class_thresholds.csv for actual support values
    thresholds_df = pd.read_csv('analysis/f1/per_class_thresholds.csv')

    # Merge to get support
    merged = max_f1_df.merge(
        thresholds_df[['class_name', 'num_positive_examples']],
        on='class_name',
        how='left'
    )

    f1_scores = merged['f1'].values
    support = merged['num_positive_examples'].fillna(0).values

    # Theme names for highlighting
    theme_names = [
        'advancedPawn', 'advantage', 'anastasiaMate', 'arabianMate', 'attackingF2F7',
        'attraction', 'backRankMate', 'bishopEndgame', 'bodenMate', 'capturingDefender',
        'castling', 'clearance', 'crushing', 'defensiveMove', 'deflection',
        'discoveredAttack', 'doubleBishopMate', 'doubleCheck', 'dovetailMate', 'endgame',
        'enPassant', 'exposedKing', 'fork', 'hangingPiece', 'hookMate',
        'interference', 'intermezzo', 'kingsideAttack', 'knightEndgame', 'long',
        'master', 'mate', 'mateIn1', 'mateIn2', 'mateIn3',
        'mateIn4', 'mateIn5', 'middlegame', 'oneMove', 'opening',
        'pawnEndgame', 'pin', 'promotion', 'queenEndgame', 'queenRookEndgame',
        'queensideAttack', 'quietMove', 'rookEndgame', 'sacrifice', 'short',
        'skewer', 'smotheredMate', 'superGM', 'trappedPiece', 'underPromotion',
        'veryLong', 'xRayAttack', 'zugzwang'
    ]

    is_theme = merged['class_name'].isin(theme_names)

    # Create figure with professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Custom colormap for points
    colors_themes = plt.cm.viridis(f1_scores[is_theme])
    colors_openings = plt.cm.plasma(f1_scores[~is_theme])

    # Plot openings (background, smaller, more transparent)
    scatter_openings = ax.scatter(
        support[~is_theme],
        f1_scores[~is_theme],
        c=f1_scores[~is_theme],
        cmap='Blues',
        s=30,
        alpha=0.4,
        edgecolors='none',
        label='Openings'
    )

    # Plot themes (foreground, larger, more prominent)
    scatter_themes = ax.scatter(
        support[is_theme],
        f1_scores[is_theme],
        c=f1_scores[is_theme],
        cmap='RdYlGn',
        s=100,
        alpha=0.85,
        edgecolors='white',
        linewidths=0.5,
        label='Themes',
        zorder=5
    )

    # Add colorbar
    cbar = plt.colorbar(scatter_themes, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('F1 Score', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)

    # Annotate top performers
    top_performers = merged.nlargest(8, 'f1')
    for _, row in top_performers.iterrows():
        if row['num_positive_examples'] > 0:
            ax.annotate(
                row['class_name'],
                (row['num_positive_examples'], row['f1']),
                xytext=(8, 4),
                textcoords='offset points',
                fontsize=8,
                color='#333333',
                fontweight='bold',
                alpha=0.9
            )

    # Styling
    ax.set_xlabel('Support (number of positive samples)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score vs Support by Label Type', fontsize=14, fontweight='bold', pad=15)

    # Use log scale for x-axis due to wide range
    ax.set_xscale('log')
    ax.set_xlim(0.8, support.max() * 1.5)
    ax.set_ylim(-0.02, 1.05)

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    legend = ax.legend(
        loc='lower right',
        fontsize=10,
        framealpha=0.95,
        edgecolor='#cccccc'
    )

    # Add statistics annotation
    theme_f1 = f1_scores[is_theme]
    opening_f1 = f1_scores[~is_theme]
    stats_text = (
        f'Themes: median F1 = {np.median(theme_f1):.2f}\n'
        f'Openings: median F1 = {np.median(opening_f1[opening_f1 > 0]):.2f}'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#dddddd')
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Scatter plot saved to {output_path}")


if __name__ == '__main__':
    create_scatter_plot('analysis/f1/per_class_pr_curves.csv', 'analysis/scatter/f1_vs_support_scatter.png')
