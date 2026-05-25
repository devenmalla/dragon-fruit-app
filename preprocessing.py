# -*- coding: utf-8 -*-
"""
Image Preprocessing Pipeline
Dragon Fruit Disease Detection

Pipeline Steps:
1. Convert to grayscale
2. Resize to fixed size (224x224)
3. Apply Gaussian filtering (noise reduction)
4. Apply Adaptive Mean Filter (edge-preserving smoothing)

Input  : dataset/raw/<category>/
Output : dataset/processed_image/<category>/
"""

import os
import cv2 as cv
import numpy as np
from gaussian_kernel import gaussFilter

# =============================================================================
# ADAPTIVE MEAN FILTER
# =============================================================================
# Purpose:
# Reduces noise while preserving important image details.
# It adapts smoothing based on local variance.
# =============================================================================

def adaptive_mean_filter(image, kernel_size=3):
    from scipy.ndimage import uniform_filter

    # Convert to float for precise calculations
    img = image.astype(np.float64)

    # Local statistics
    local_mean    = uniform_filter(img, size=kernel_size)
    local_sq_mean = uniform_filter(img ** 2, size=kernel_size)
    local_var     = np.maximum(local_sq_mean - local_mean ** 2, 0)

    # Global variance
    global_var = np.var(img)

    # Compute adaptive ratio
    ratio = np.where(
        local_var == 0,
        1.0,
        np.minimum(global_var / (local_var + 1e-10), 1.0)
    )

    # Apply filtering
    output = img - ratio * (img - local_mean)

    # Clip values back to valid image range
    return np.clip(output, 0, 255).astype(np.uint8)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Target image size for all models
IMG_SIZE = (224, 224)

# Dataset categories (must match folder names exactly)
categories = ['Bad fruit', 'Bad leaf', 'Good fruit', 'Good leaf']

# Prefix for renaming processed images
prefix_map = {
    'Bad fruit' : 'bad_fruit_',
    'Bad leaf'  : 'bad_leaf_',
    'Good fruit': 'good_fruit_',
    'Good leaf' : 'good_leaf_',
}

# Input (raw images)
raw_base = 'dataset/raw'

# Output (processed images)
output_base = 'dataset/processed_image'

# Gaussian kernel (3x3, sigma=1)
gaussianFilter = gaussFilter((3, 3), 1)

# Counters for tracking progress
total_processed = 0
total_failed    = 0

print("=" * 60)
print("Preprocessing Started")
print("Pipeline: Grayscale → Resize → Gaussian → Adaptive Mean")
print("=" * 60)


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

for category in categories:

    # Define input and output directories
    input_dir  = os.path.join(raw_base, category)
    output_dir = os.path.join(output_base, category)

    # Create output folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all image filenames
    files = os.listdir(input_dir)

    print(f"\nProcessing: {category} ({len(files)} images)")

    category_count  = 0
    category_failed = 0

    for file in files:

        img_path = os.path.join(input_dir, file)

        # -------------------------------------------------
        # Step 1: Read image in grayscale
        # -------------------------------------------------
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        if image is None:
            category_failed += 1
            continue

        # -------------------------------------------------
        # Step 2: Resize to standard size
        # -------------------------------------------------
        image = cv.resize(image, IMG_SIZE, interpolation=cv.INTER_AREA)

        # -------------------------------------------------
        # Step 3: Apply Gaussian filter
        # -------------------------------------------------
        gaussian_filtered = cv.filter2D(image, -1, gaussianFilter)

        # -------------------------------------------------
        # Step 4: Apply Adaptive Mean Filter
        # -------------------------------------------------
        final_image = adaptive_mean_filter(gaussian_filtered, kernel_size=3)

        # -------------------------------------------------
        # Save processed image
        # -------------------------------------------------
        out_name = prefix_map[category] + file
        out_path = os.path.join(output_dir, out_name)

        if cv.imwrite(out_path, final_image):
            category_count += 1
        else:
            category_failed += 1

    # Category summary
    print(f"Saved: {category_count} | Failed: {category_failed}")

    total_processed += category_count
    total_failed    += category_failed


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("Preprocessing Complete")
print(f"Total saved  : {total_processed}")
print(f"Total failed : {total_failed}")
print("=" * 60)