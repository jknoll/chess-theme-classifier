#!/usr/bin/env python3
"""Create improved opening performance charts for README."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_opening_charts(pr_curves_csv, output_dir):
    """Create two versions of opening performance charts.

    1. Top 20 performers with legible labels
    2. All openings without labels (distribution view)

    Args:
        pr_curves_csv: Path to per_class_pr_curves.csv
        output_dir: Directory for output charts
    """
    # Load data - this has detailed metrics at each threshold
    df = pd.read_csv(pr_curves_csv)

    # Get max F1 for each class
    max_f1_df = df.groupby('class_name').agg({
        'f1': 'max',
        'class_index': 'first'
    }).reset_index()

    # Theme names (camelCase, no underscores typically)
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

    # Filter to openings (not in theme_names)
    openings_df = max_f1_df[~max_f1_df['class_name'].isin(theme_names)].copy()

    # Sort by F1 score
    openings_df = openings_df.sort_values('f1', ascending=False)

    print(f"Found {len(openings_df)} openings")
    print(f"Top 5 openings by F1:")
    print(openings_df.head())

    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Top 20 with legible labels
    top20 = openings_df.head(20)

    plt.figure(figsize=(14, 8))
    x = np.arange(len(top20))

    # Clean up label names (replace underscores with spaces)
    labels = [l.replace('_', ' ') for l in top20['class_name']]
    f1_scores = top20['f1'].values

    bars = plt.bar(x, f1_scores, color='#2ecc71', edgecolor='#27ae60', linewidth=0.5)

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Opening', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Opening Classification Performance (Top 20)', fontsize=14)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=10)
    plt.ylim(0, max(f1_scores) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_path1 = os.path.join(output_dir, 'openings_top20_chart.png')
    plt.savefig(output_path1, dpi=300)
    plt.close()
    print(f"Saved: {output_path1}")

    # Chart 2: All openings without labels (distribution view)
    plt.figure(figsize=(16, 6))
    x = np.arange(len(openings_df))
    f1_all = openings_df['f1'].values

    # Use color gradient based on F1 score
    colors = plt.cm.RdYlGn(f1_all / max(f1_all) if max(f1_all) > 0 else f1_all)

    plt.bar(x, f1_all, color=colors, width=1.0, edgecolor='none')

    plt.xlabel('Opening Index (sorted by F1 score)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'Opening Classification Performance Distribution (All {len(openings_df)} Openings)', fontsize=14)

    # Remove x-axis tick labels but keep the axis
    plt.xticks([])
    plt.ylim(0, max(f1_all) * 1.1 if max(f1_all) > 0 else 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add annotation for key statistics
    nonzero_f1 = f1_all[f1_all > 0.01]
    stats_text = f'Non-zero F1: {len(nonzero_f1)}/{len(f1_all)} openings\nMax F1: {max(f1_all):.2f}\nMedian (non-zero): {np.median(nonzero_f1):.2f}'
    plt.text(0.98, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    output_path2 = os.path.join(output_dir, 'openings_all_no_labels_chart.png')
    plt.savefig(output_path2, dpi=300)
    plt.close()
    print(f"Saved: {output_path2}")

if __name__ == '__main__':
    pr_curves_csv = 'analysis/f1/per_class_pr_curves.csv'
    output_dir = 'analysis/f1'
    create_opening_charts(pr_curves_csv, output_dir)
