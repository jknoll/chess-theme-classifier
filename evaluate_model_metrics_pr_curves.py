#!/usr/bin/env python3
"""
Standalone script to generate precision-recall curve visualizations from per-class PR curve data.
Reads from analysis/f1/per_class_pr_curves.csv and outputs visualizations to analysis/pr-curves/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

def create_pr_curve_plot(class_data, class_name, output_dir):
    """
    Create a precision-recall curve plot for a single class.
    
    Args:
        class_data: DataFrame containing PR curve data for one class
        class_name: Name of the class
        output_dir: Directory to save the plot
    """
    # Sort by recall for proper curve plotting
    class_data = class_data.sort_values('recall')
    
    # Extract data
    precision = class_data['precision'].values
    recall = class_data['recall'].values
    f1_scores = class_data['f1'].values
    thresholds = class_data['threshold'].values
    
    # Find the point with maximum F1 score
    max_f1_idx = np.argmax(f1_scores)
    max_f1_score = f1_scores[max_f1_idx]
    max_f1_precision = precision[max_f1_idx]
    max_f1_recall = recall[max_f1_idx]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot with color mapping based on threshold values (like reference image)
    # For overlapping points, increase size and add edge to make them visible
    scatter = ax.scatter(recall, precision, c=thresholds, cmap='viridis', 
                        s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold', rotation=270, labelpad=15)
    
    # Highlight the maximum F1 score point
    ax.scatter(max_f1_recall, max_f1_precision, color='red', s=100, 
              edgecolors='darkred', linewidth=2, zorder=5,
              label=f'Max F-score ({max_f1_score:.2f})')
    
    # Set labels and title
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{class_name} Precision-Recall Curve', fontsize=14, fontweight='bold')
    
    # Set axis limits from 0.0 to 1.0 for both axes
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='lower left')
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot with F1 score prefix
    safe_class_name = class_name.replace('/', '_').replace(' ', '_')
    output_path = os.path.join(output_dir, f'{max_f1_score:.2f}_{safe_class_name}_pr_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate precision-recall curve visualizations')
    parser.add_argument('--input_file', type=str, 
                       default='analysis/f1/per_class_pr_curves.csv',
                       help='Path to the per-class PR curves CSV file')
    parser.add_argument('--output_dir', type=str, 
                       default='analysis/pr-curves',
                       help='Directory to save PR curve plots')
    parser.add_argument('--min_positive_examples', type=int, default=1,
                       help='Minimum number of positive examples to generate PR curve')
    parser.add_argument('--max_classes', type=int, default=None,
                       help='Maximum number of classes to process (default: None for all classes, set to specific number to limit)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        print("Please run evaluate_model_metrics.py first to generate the PR curve data.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the PR curve data
    if args.verbose:
        print(f"Loading PR curve data from {args.input_file}...")
    
    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return 1
    
    if args.verbose:
        print(f"Loaded data for {df['class_name'].nunique()} unique classes")
        print(f"Total data points: {len(df)}")
    
    # Group by class and calculate max F1 score for each class
    class_groups = df.groupby('class_name')
    
    # Calculate max F1 score for each class and sort by it
    class_max_f1 = []
    for class_name, class_data in class_groups:
        # Skip classes with no positive examples
        has_positive_examples = (class_data['tp'] > 0).any()
        if has_positive_examples:
            max_f1 = class_data['f1'].max()
            class_max_f1.append((class_name, max_f1, class_data))
    
    # Sort by max F1 score in descending order
    class_max_f1.sort(key=lambda x: x[1], reverse=True)
    
    if args.verbose:
        print(f"Found {len(class_max_f1)} classes with positive examples")
        print("Top 10 classes by max F1 score:")
        for i, (class_name, max_f1, _) in enumerate(class_max_f1[:10]):
            print(f"  {i+1:2d}. {class_name} (F1: {max_f1:.3f})")
        if len(class_max_f1) > 10:
            print(f"  ... and {len(class_max_f1) - 10} more classes")
    
    generated_plots = []
    skipped_classes = []
    
    # Process classes in order of decreasing F1 score
    for class_name, max_f1, class_data in class_max_f1:
        
        # Check if we've reached the max classes limit
        if args.max_classes is not None and len(generated_plots) >= args.max_classes:
            if args.verbose:
                print(f"Reached maximum classes limit ({args.max_classes})")
            break
        
        try:
            output_path = create_pr_curve_plot(class_data, class_name, args.output_dir)
            generated_plots.append(output_path)
            
            if args.verbose:
                max_f1 = class_data['f1'].max()
                print(f"Generated PR curve for '{class_name}' (max F1: {max_f1:.3f}) -> {output_path}")
                
        except Exception as e:
            print(f"Error generating PR curve for '{class_name}': {e}")
            continue
    
    # Summary
    print(f"\n=== PR Curve Generation Summary ===")
    print(f"Successfully generated: {len(generated_plots)} PR curves")
    print(f"Skipped classes (no positive examples): {len(skipped_classes)}")
    print(f"Output directory: {args.output_dir}")
    
    if args.verbose and skipped_classes:
        print(f"\nSkipped classes: {skipped_classes[:10]}")  # Show first 10
        if len(skipped_classes) > 10:
            print(f"... and {len(skipped_classes) - 10} more")
    
    if generated_plots:
        print(f"\nExample generated files:")
        for path in generated_plots[:5]:  # Show first 5
            print(f"  {path}")
        if len(generated_plots) > 5:
            print(f"  ... and {len(generated_plots) - 5} more")
    
    return 0

if __name__ == "__main__":
    exit(main())