#!/usr/bin/env python3
"""Create a styled theme performance chart with color mapping by F1 score."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_theme_chart(pr_curves_csv, output_dir):
    """Create theme performance chart with color gradient based on F1 score.

    Args:
        pr_curves_csv: Path to per_class_pr_curves.csv
        output_dir: Directory for output chart
    """
    # Load data
    df = pd.read_csv(pr_curves_csv)

    # Get max F1 for each class
    max_f1_df = df.groupby('class_name').agg({
        'f1': 'max',
        'class_index': 'first'
    }).reset_index()

    # Theme names (camelCase, typically no underscores)
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

    # Filter to themes only
    themes_df = max_f1_df[max_f1_df['class_name'].isin(theme_names)].copy()

    # Sort by F1 score descending
    themes_df = themes_df.sort_values('f1', ascending=False)

    print(f"Found {len(themes_df)} themes")
    print(f"Top 5 themes by F1:")
    print(themes_df.head())

    os.makedirs(output_dir, exist_ok=True)

    # Create chart with color gradient
    plt.figure(figsize=(16, 8))
    x = np.arange(len(themes_df))
    f1_scores = themes_df['f1'].values
    labels = themes_df['class_name'].values

    # Use color gradient based on F1 score (same as openings chart)
    colors = plt.cm.RdYlGn(f1_scores / max(f1_scores) if max(f1_scores) > 0 else f1_scores)

    bars = plt.bar(x, f1_scores, color=colors, width=0.8, edgecolor='none')

    plt.xlabel('Theme', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'Theme Classification Performance (All {len(themes_df)} Themes, Sorted by F1)', fontsize=14)

    # Rotate labels for readability
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=9)
    plt.ylim(0, max(f1_scores) * 1.1 if max(f1_scores) > 0 else 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add annotation for key statistics
    stats_text = f'Max F1: {max(f1_scores):.2f}\nMedian F1: {np.median(f1_scores):.2f}\nMin F1: {min(f1_scores):.2f}'
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'themes_f1_color_gradient_chart.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    pr_curves_csv = 'analysis/f1/per_class_pr_curves.csv'
    output_dir = 'analysis/f1'
    create_theme_chart(pr_curves_csv, output_dir)
