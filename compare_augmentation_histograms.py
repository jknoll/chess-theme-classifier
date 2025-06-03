#!/usr/bin/env python3
"""
Compare theme distributions between regular and class-conditionally augmented datasets.
This script generates histograms showing how the theme distribution changes
with selective augmentation of rare theme combinations.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import ChessPuzzleDataset
import argparse
from collections import Counter

def get_theme_distribution(dataset, only_themes=False):
    """
    Calculate theme distribution from the dataset.
    
    Args:
        dataset: ChessPuzzleDataset instance
        only_themes: If True, analyze only theme labels (not openings)
    
    Returns:
        Counter of theme frequencies
    """
    theme_counts = Counter()
    all_labels = dataset.all_labels
    theme_only_indices = set(i for i, label in enumerate(all_labels) 
                            if label in dataset.all_themes)
    
    # Process each sample in the dataset
    for i in range(len(dataset)):
        item = dataset[i]
        themes_tensor = item['themes']
        
        # Iterate through each position in the one-hot vector
        for j in range(len(themes_tensor)):
            if themes_tensor[j] > 0:
                # If only considering themes, check that this is a theme label
                if not only_themes or j in theme_only_indices:
                    label = all_labels[j]
                    theme_counts[label] += 1
    
    return theme_counts

def plot_combined_histogram(regular_dist, augmented_dist, title, filename, rarity_threshold=None):
    """
    Plot a combined histogram showing before/after augmentation, sorted by frequency.
    
    Args:
        regular_dist: Counter with distribution before augmentation
        augmented_dist: Counter with distribution after augmentation
        title: Title for the plot
        filename: Output filename
        rarity_threshold: Value below which themes are considered rare
    """
    # Combine all labels
    all_labels = set(list(regular_dist.keys()) + list(augmented_dist.keys()))
    
    # Sort labels by regular dataset frequency (descending)
    sorted_labels = sorted(all_labels, key=lambda label: -regular_dist.get(label, 0))
    
    # Data preparation
    x = np.arange(len(sorted_labels))
    width = 0.35
    
    # Create the plot
    plt.figure(figsize=(20, 10))
    
    # Plot bars for both distributions
    regular_values = [regular_dist.get(label, 0) for label in sorted_labels]
    augmented_values = [augmented_dist.get(label, 0) for label in sorted_labels]
    
    # Plot the bars
    bars1 = plt.bar(x - width/2, regular_values, width, label='Regular Dataset', color='blue', alpha=0.7)
    bars2 = plt.bar(x + width/2, augmented_values, width, label='Augmented Dataset', color='red', alpha=0.7)
    
    # If rarity threshold is provided, draw a horizontal line
    if rarity_threshold is not None:
        plt.axhline(y=rarity_threshold, color='green', linestyle='--', linewidth=2)
        plt.text(len(sorted_labels)*0.95, rarity_threshold*1.1, 
                 f'Rarity Threshold = {rarity_threshold}', 
                 color='green', fontsize=12, ha='right')
    
    # Customize the plot
    plt.xlabel('Theme Labels (Sorted by Frequency)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    
    # Choose how many labels to show based on dataset size
    if len(sorted_labels) > 50:
        # For large datasets, show ~30 labels evenly distributed
        step = max(1, len(sorted_labels) // 30)
        plt.xticks(x[::step], [sorted_labels[i] for i in range(0, len(sorted_labels), step)], rotation=90)
    else:
        plt.xticks(x, sorted_labels, rotation=90)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename)
    print(f"Saved comparison histogram to {filename}")
    
    # Save data to text file
    txt_filename = filename.replace('.png', '_data.txt')
    with open(txt_filename, 'w') as f:
        f.write(f"{title}\n\n")
        f.write("Label\tRegular\tAugmented\tChange\tPercent Increase\tRare?\n")
        
        for label in sorted_labels:
            reg_count = regular_dist.get(label, 0)
            aug_count = augmented_dist.get(label, 0)
            change = aug_count - reg_count
            pct_increase = (change / reg_count * 100) if reg_count > 0 else float('inf')
            is_rare = "Yes" if rarity_threshold is not None and reg_count <= rarity_threshold else "No"
            
            f.write(f"{label}\t{reg_count}\t{aug_count}\t{change}\t{pct_increase:.2f}%\t{is_rare}\n")
    
    print(f"Saved distribution data to {txt_filename}")
    
    # Additional analysis
    print("\nDistribution summary:")
    reg_values = np.array(list(regular_dist.values()))
    aug_values = np.array(list(augmented_dist.values()))
    
    print(f"Regular distribution: min={reg_values.min()}, mean={reg_values.mean():.2f}, max={reg_values.max()}")
    print(f"Augmented distribution: min={aug_values.min()}, mean={aug_values.mean():.2f}, max={aug_values.max()}")
    
    # Coefficient of variation (lower is more balanced)
    cv_reg = reg_values.std() / reg_values.mean() if reg_values.mean() > 0 else 0
    cv_aug = aug_values.std() / aug_values.mean() if aug_values.mean() > 0 else 0
    
    print(f"Coefficient of variation (lower is more balanced):")
    print(f"  Regular: {cv_reg:.4f}")
    print(f"  Augmented: {cv_aug:.4f}")
    print(f"  Improvement: {(cv_reg - cv_aug) / cv_reg * 100:.2f}%")
    
    # Rare/common ratio (higher ratio indicates better representation of rare themes)
    rare_threshold = np.percentile(reg_values, 25)  # Bottom 25% are considered rare
    rare_labels = set(label for label, count in regular_dist.items() if count <= rare_threshold)
    
    rare_reg_sum = sum(regular_dist.get(label, 0) for label in rare_labels)
    common_reg_sum = sum(regular_dist.get(label, 0) for label in regular_dist if label not in rare_labels)
    
    rare_aug_sum = sum(augmented_dist.get(label, 0) for label in rare_labels)
    common_aug_sum = sum(augmented_dist.get(label, 0) for label in augmented_dist if label not in rare_labels)
    
    rare_ratio_reg = rare_reg_sum / common_reg_sum if common_reg_sum > 0 else 0
    rare_ratio_aug = rare_aug_sum / common_aug_sum if common_aug_sum > 0 else 0
    
    print(f"Rare-to-common label ratio:")
    print(f"  Regular: {rare_ratio_reg:.4f}")
    print(f"  Augmented: {rare_ratio_aug:.4f}")
    print(f"  Improvement: {(rare_ratio_aug - rare_ratio_reg) / rare_ratio_reg * 100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Compare theme distributions with and without augmentation")
    parser.add_argument("--csv", default="lichess_db_puzzle_test.csv", help="CSV file to use")
    parser.add_argument("--rarity-threshold", type=int, default=None, 
                       help="Rarity threshold for class-conditional augmentation")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs('analysis/histograms', exist_ok=True)
    
    # Get base name for output files
    base_name = os.path.basename(args.csv).split('.')[0]
    
    print(f"Loading regular dataset from {args.csv}...")
    regular_dataset = ChessPuzzleDataset(args.csv)
    
    print(f"Loading class-conditionally augmented dataset from {args.csv}...")
    augmented_dataset = ChessPuzzleDataset(args.csv, class_conditional_augmentation=True, 
                                           rarity_threshold=args.rarity_threshold)
    
    # Generate theme distribution histograms
    print("Analyzing theme distributions...")
    
    # Get distributions
    regular_themes = get_theme_distribution(regular_dataset, only_themes=True)
    augmented_themes = get_theme_distribution(augmented_dataset, only_themes=True)
    
    # Get rarity threshold - this is what determines which themes get augmented
    if hasattr(augmented_dataset, 'rare_combinations'):
        # Determine the frequency threshold that was used for rarity
        rarity_threshold = min(
            regular_themes[theme] for combo in augmented_dataset.rare_combinations 
            for theme in combo if theme in regular_themes
        )
        print(f"Detected rarity threshold for augmentation: {rarity_threshold}")
    else:
        # Use the 25th percentile as an approximation
        rarity_threshold = np.percentile(list(regular_themes.values()), 25)
        print(f"Using approximate rarity threshold (25th percentile): {rarity_threshold}")
    
    # Plot a single histogram
    histogram_file = f"analysis/histograms/{base_name}_theme_distribution_frequency_sorted.png"
    plot_combined_histogram(
        regular_themes,
        augmented_themes,
        f"Theme Distribution Before/After Class-Conditional Augmentation ({base_name})",
        histogram_file,
        rarity_threshold=rarity_threshold
    )
    
    # Print info about theme-combination vs individual theme augmentation
    print("\nExamining augmentation of common themes:")
    common_themes_augmented = []
    
    # Find common themes that got augmented
    for label, count in regular_themes.items():
        if count > rarity_threshold and augmented_themes.get(label, 0) > count:
            common_themes_augmented.append((label, count, augmented_themes.get(label, 0)))
    
    if common_themes_augmented:
        print("These common themes were augmented despite being above the rarity threshold:")
        for theme, original, augmented in sorted(common_themes_augmented, key=lambda x: -x[1])[:10]:
            print(f"  {theme}: {original} → {augmented} (+{augmented-original})")
        print("This occurs because these common themes co-occur with rare themes in specific puzzles.")
    else:
        print("No common themes were augmented - all augmented themes were below the rarity threshold.")
        
    print("\nFinished generating histograms.")

if __name__ == "__main__":
    main()