#!/usr/bin/env python3
"""
Standalone script to generate F1 vs Support scatter plot visualization.
Reads from analysis/f1/per_class_pr_curves.csv and per_class_thresholds.csv
and outputs visualization to analysis/scatter/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

def is_theme_class(class_name):
    """
    Determine if a class is a theme or opening based on naming conventions.
    Themes typically have simple names, openings have more complex names with underscores.
    """
    # Common theme patterns (single words or simple combinations)
    common_themes = {
        'advantage', 'attacking', 'backRankMate', 'bishopEndgame', 'capturingDefender',
        'clearance', 'crushing', 'defensiveMove', 'deflection', 'discoveredAttack',
        'doubleBishopMate', 'doubleCheck', 'dovetailMate', 'endgame', 'enPassant',
        'equality', 'escape', 'fork', 'hangingPiece', 'hookMate', 'interference',
        'intermezzo', 'kingsideAttack', 'knightEndgame', 'long', 'mate', 'mateIn1',
        'mateIn2', 'mateIn3', 'mateIn4', 'mateIn5', 'middlegame', 'opening',
        'pawnEndgame', 'pin', 'promotion', 'queenEndgame', 'queenRookEndgame',
        'queensideAttack', 'quietMove', 'rookEndgame', 'sacrifice', 'short',
        'skewer', 'smotheredMate', 'superGM', 'trappedPiece', 'underPromotion',
        'veryLong', 'xRayAttack', 'zugzwang'
    }
    
    # If it's in our known themes list, it's definitely a theme
    if class_name in common_themes:
        return True
    
    # If it contains underscores and has "Defense", "Opening", "Game", "Attack", etc., it's likely an opening
    opening_indicators = ['_Defense', '_Opening', '_Game', '_Attack', '_Variation', '_Gambit', '_System']
    if any(indicator in class_name for indicator in opening_indicators):
        return False
    
    # If it has multiple underscores, it's likely an opening
    if class_name.count('_') >= 2:
        return False
    
    # Default to theme for simple names
    return True

def create_f1_vs_support_plot(pr_data, threshold_data, output_dir):
    """
    Create F1 vs Support scatter plot distinguishing themes from openings.
    
    Args:
        pr_data: DataFrame containing per-class PR curve data
        threshold_data: DataFrame containing per-class threshold and support data
        output_dir: Directory to save the plot
    """
    # Calculate max F1 score for each class
    max_f1_per_class = pr_data.groupby('class_name')['f1'].max().reset_index()
    
    # Merge with threshold data to get support information
    plot_data = pd.merge(max_f1_per_class, threshold_data, on='class_name', how='inner')
    
    # Classify as theme or opening
    plot_data['is_theme'] = plot_data['class_name'].apply(is_theme_class)
    
    # Separate themes and openings
    themes = plot_data[plot_data['is_theme'] == True]
    openings = plot_data[plot_data['is_theme'] == False]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot themes in blue
    if len(themes) > 0:
        ax.scatter(themes['num_positive_examples'], themes['f1'], 
                  c='blue', alpha=0.7, s=60, label=f'Themes ({len(themes)})', edgecolors='navy', linewidth=0.5)
    
    # Plot openings in red
    if len(openings) > 0:
        ax.scatter(openings['num_positive_examples'], openings['f1'], 
                  c='red', alpha=0.7, s=60, label=f'Openings ({len(openings)})', edgecolors='darkred', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Support (Number of Positive Examples)', fontsize=12)
    ax.set_ylabel('Maximum F1 Score', fontsize=12)
    ax.set_title('F1 Score vs Support: Chess Themes and Openings', fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(0, max(plot_data['num_positive_examples']) * 1.05)
    ax.set_ylim(0, 1.0)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Add some statistics as text
    stats_text = f"Total Classes: {len(plot_data)}\n"
    stats_text += f"Mean F1 (Themes): {themes['f1'].mean():.3f}\n" if len(themes) > 0 else "Mean F1 (Themes): N/A\n"
    stats_text += f"Mean F1 (Openings): {openings['f1'].mean():.3f}\n" if len(openings) > 0 else "Mean F1 (Openings): N/A\n"
    stats_text += f"Mean Support (Themes): {themes['num_positive_examples'].mean():.1f}\n" if len(themes) > 0 else "Mean Support (Themes): N/A\n"
    stats_text += f"Mean Support (Openings): {openings['num_positive_examples'].mean():.1f}" if len(openings) > 0 else "Mean Support (Openings): N/A"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'f1_vs_support_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path, len(themes), len(openings)

def main():
    parser = argparse.ArgumentParser(description='Generate F1 vs Support scatter plot visualization')
    parser.add_argument('--pr_curves_file', type=str, 
                       default='analysis/f1/per_class_pr_curves.csv',
                       help='Path to the per-class PR curves CSV file')
    parser.add_argument('--thresholds_file', type=str,
                       default='analysis/f1/per_class_thresholds.csv',
                       help='Path to the per-class thresholds CSV file')
    parser.add_argument('--output_dir', type=str, 
                       default='analysis/scatter',
                       help='Directory to save scatter plot')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.pr_curves_file):
        print(f"Error: PR curves file {args.pr_curves_file} not found.")
        print("Please run evaluate_model_metrics.py first to generate the PR curve data.")
        return 1
        
    if not os.path.exists(args.thresholds_file):
        print(f"Error: Thresholds file {args.thresholds_file} not found.")
        print("Please run evaluate_model_metrics.py first to generate the threshold data.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    if args.verbose:
        print(f"Loading PR curve data from {args.pr_curves_file}...")
        print(f"Loading threshold data from {args.thresholds_file}...")
    
    try:
        pr_data = pd.read_csv(args.pr_curves_file)
        threshold_data = pd.read_csv(args.thresholds_file)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return 1
    
    if args.verbose:
        print(f"Loaded PR data for {pr_data['class_name'].nunique()} unique classes")
        print(f"Loaded threshold data for {len(threshold_data)} classes")
    
    try:
        output_path, num_themes, num_openings = create_f1_vs_support_plot(pr_data, threshold_data, args.output_dir)
        
        # Summary
        print(f"=== F1 vs Support Scatter Plot Generation Summary ===")
        print(f"Successfully generated scatter plot: {output_path}")
        print(f"Themes plotted: {num_themes}")
        print(f"Openings plotted: {num_openings}")
        print(f"Total classes: {num_themes + num_openings}")
        print(f"Output directory: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"Error generating scatter plot: {e}")
        return 1

if __name__ == "__main__":
    exit(main())