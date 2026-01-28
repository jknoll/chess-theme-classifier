#!/usr/bin/env python3
"""Create a matrix view of top-performing PR curves."""

import os
from PIL import Image
import math

def create_pr_matrix(pr_curves_dir, output_path, n_images=9):
    """Create a grid of top-performing PR curves.

    Args:
        pr_curves_dir: Directory containing PR curve images
        output_path: Output path for the matrix image
        n_images: Number of images (should be a perfect square)
    """
    # Get all PR curve files with their F1 scores
    files = []
    for f in os.listdir(pr_curves_dir):
        if f.endswith('_pr_curve.png'):
            try:
                f1_score = float(f.split('_')[0])
                files.append((f1_score, f))
            except ValueError:
                continue

    # Sort by F1 score descending and take top n
    files.sort(reverse=True)
    top_files = files[:n_images]

    print(f"Creating matrix with {n_images} images:")
    for f1, fname in top_files:
        print(f"  F1={f1:.2f}: {fname}")

    # Load images
    images = []
    for _, fname in top_files:
        img_path = os.path.join(pr_curves_dir, fname)
        img = Image.open(img_path)
        images.append(img)

    # Calculate grid dimensions
    grid_size = int(math.sqrt(n_images))

    # Get dimensions of first image (assume all same size)
    img_width, img_height = images[0].size

    # Create output image
    matrix_width = grid_size * img_width
    matrix_height = grid_size * img_height
    matrix = Image.new('RGB', (matrix_width, matrix_height), 'white')

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        x = col * img_width
        y = row * img_height
        matrix.paste(img, (x, y))

    # Save
    matrix.save(output_path, quality=95)
    print(f"Saved matrix to {output_path}")
    print(f"Dimensions: {matrix_width}x{matrix_height}")

if __name__ == '__main__':
    pr_curves_dir = 'analysis/pr-curves'
    output_path = 'analysis/pr-curves/top_25_pr_curves_matrix.png'
    create_pr_matrix(pr_curves_dir, output_path, n_images=25)
