#!/usr/bin/env python3
"""
Generate histograms of chess puzzle themes and openings frequencies.
This script reads the frequency counts from the ChessPuzzleDataset
and generates three histograms: openings, themes, and both combined.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import ChessPuzzleDataset
import argparse

def plot_frequency_histogram(data, title, filename, limit=None, figsize=(12, 8), save_txt=True):
    """
    Plot a histogram of frequencies and save details to a text file.
    
    Args:
        data: List of (name, count) tuples
        title: Title for the histogram
        filename: Output filename
        limit: Optional limit on how many items to include
        figsize: Figure size
        save_txt: Whether to save a text file with frequency data
    """
    # Store original data for text file
    original_data = data.copy()
    
    # Limit the data if requested for plot
    if limit and len(data) > limit:
        plot_data = data[:limit]
        plot_title = f"{title} (Top {limit})"
    else:
        plot_data = data
        plot_title = title
    
    # Extract names and counts
    names = [item[0] for item in plot_data]
    counts = [item[1] for item in plot_data]
    
    # Create the plot
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(names)), counts, color='skyblue')
    
    # Add values on top of bars if there aren't too many
    if len(bars) <= 50:  # Only add text for bars if there aren't too many
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', rotation=0)
    
    # Customize the plot
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title(plot_title)
    
    # Adjust x-ticks based on number of items
    if len(names) <= 30:
        plt.xticks(range(len(names)), names, rotation=90)
    else:
        # For many items, show fewer labels to avoid overcrowding
        step = max(1, len(names) // 30)
        plt.xticks(range(0, len(names), step), [names[i] for i in range(0, len(names), step)], rotation=90)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(filename)
    plt.close()
    print(f"Saved histogram to {filename}")
    
    # Save frequency data to text file
    if save_txt:
        txt_filename = filename.replace('.png', '_data.txt')
        with open(txt_filename, 'w') as f:
            f.write(f"{title}\n")
            f.write(f"Total items: {len(original_data)}\n")
            f.write(f"Items shown in plot: {len(plot_data)}\n\n")
            f.write("Rank\tName\tCount\tPercentage\n")
            
            total_count = sum(item[1] for item in original_data)
            for i, (name, count) in enumerate(original_data):
                percentage = (count / total_count) * 100
                f.write(f"{i+1}\t{name}\t{count}\t{percentage:.2f}%\n")
        
        print(f"Saved frequency data to {txt_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate histograms of chess puzzle themes and openings frequencies")
    parser.add_argument("--csv", default="lichess_db_puzzle.csv", help="CSV file containing puzzle data")
    parser.add_argument("--limit", type=int, default=30, help="Limit the number of themes/openings shown in truncated histograms")
    parser.add_argument("--full-figsize", default="20,15", help="Figure size for full histograms (width,height in inches)")
    args = parser.parse_args()
    
    # Parse figure size
    full_figsize = tuple(map(float, args.full_figsize.split(',')))
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file {args.csv} not found.")
        return
    
    # Load the dataset
    print(f"Loading dataset from {args.csv}...")
    dataset = ChessPuzzleDataset(args.csv)
    
    # Get frequencies
    theme_freqs = dataset.get_theme_frequencies()
    opening_freqs = dataset.get_opening_frequencies()
    
    # Generate file names
    base_name = os.path.basename(args.csv).split('.')[0]
    themes_file = f"{base_name}_theme_histogram.png"
    openings_file = f"{base_name}_opening_histogram.png"
    combined_file = f"{base_name}_combined_histogram.png"
    
    # Full histogram file names
    full_themes_file = f"{base_name}_theme_histogram_full.png"
    full_openings_file = f"{base_name}_opening_histogram_full.png"
    full_combined_file = f"{base_name}_combined_histogram_full.png"
    
    # Generate histograms
    print("Generating histograms...")
    
    # Themes histograms
    if theme_freqs:
        # Truncated themes histogram
        plot_frequency_histogram(
            theme_freqs, 
            f"Chess Puzzle Theme Frequencies in {base_name}",
            themes_file,
            limit=args.limit
        )
        
        # Full themes histogram
        plot_frequency_histogram(
            theme_freqs, 
            f"Chess Puzzle Theme Frequencies in {base_name} (All Themes)",
            full_themes_file,
            figsize=full_figsize
        )
    else:
        print("No theme data available.")
    
    # Openings histograms
    if opening_freqs:
        # Truncated openings histogram
        plot_frequency_histogram(
            opening_freqs, 
            f"Chess Opening Tag Frequencies in {base_name}",
            openings_file,
            limit=args.limit
        )
        
        # Full openings histogram
        plot_frequency_histogram(
            opening_freqs, 
            f"Chess Opening Tag Frequencies in {base_name} (All Openings)",
            full_openings_file,
            figsize=full_figsize
        )
    else:
        print("No opening data available.")
    
    # Combined histograms
    if theme_freqs or opening_freqs:
        # Create combined data for both versions
        truncated_combined_data = []
        full_combined_data = []
        
        # For truncated histogram
        if theme_freqs:
            top_themes = theme_freqs[:min(args.limit // 2, len(theme_freqs))]
            truncated_combined_data.extend([(f"T:{name}", count) for name, count in top_themes])
        
        if opening_freqs:
            top_openings = opening_freqs[:min(args.limit // 2, len(opening_freqs))]
            truncated_combined_data.extend([(f"O:{name}", count) for name, count in top_openings])
        
        # Sort truncated combined data
        truncated_combined_data.sort(key=lambda x: x[1], reverse=True)
        
        # For full histogram
        if theme_freqs:
            full_combined_data.extend([(f"T:{name}", count) for name, count in theme_freqs])
        
        if opening_freqs:
            full_combined_data.extend([(f"O:{name}", count) for name, count in opening_freqs])
        
        # Sort full combined data
        full_combined_data.sort(key=lambda x: x[1], reverse=True)
        
        # Generate truncated combined histogram
        plot_frequency_histogram(
            truncated_combined_data, 
            f"Chess Puzzle Themes and Openings in {base_name}",
            combined_file,
            limit=args.limit
        )
        
        # Generate full combined histogram
        plot_frequency_histogram(
            full_combined_data, 
            f"Chess Puzzle Themes and Openings in {base_name} (All Items)",
            full_combined_file,
            figsize=full_figsize
        )
    
    print("Finished generating histograms.")

if __name__ == "__main__":
    main()