# -*- coding: utf-8 -*-

"""
Pipeline-Aligned Analysis Script
Dragon Fruit Disease Detection

Purpose:
- Compare different preprocessing techniques
- Evaluate using PSNR, SSIM, MSE, Entropy
- Save visual and histogram results in results/analysis/
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
from gaussian_kernel import gaussFilter
from scipy.ndimage import uniform_filter


# =============================================================================
# PATH SETUP (FIXED)
# =============================================================================

# Get base directory (project root safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: RAW images (correct for analysis comparison)
dataset_dir = os.path.join(BASE_DIR, "dataset", "raw")

# Output: Save inside results/analysis/
results_dir = os.path.join(BASE_DIR, "results", "analysis")
os.makedirs(results_dir, exist_ok=True)


# =============================================================================
# ADAPTIVE MEAN FILTER (Custom Noise Reduction)
# =============================================================================

def adaptive_mean_filter(image, kernel_size=3):
    """
    Adaptive Mean Filter:
    Reduces noise while preserving edges
    """
    img = image.astype(np.float64)

    local_mean = uniform_filter(img, size=kernel_size)
    local_sq_mean = uniform_filter(img ** 2, size=kernel_size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)

    global_var = np.var(img)

    ratio = np.where(
        local_var == 0,
        1.0,
        np.minimum(global_var / (local_var + 1e-10), 1.0)
    )

    output = img - ratio * (img - local_mean)

    return np.clip(output, 0, 255).astype(np.uint8)


# =============================================================================
# CONFIGURATION
# =============================================================================

IMG_SIZE = (224, 224)

# Category paths (FIXED using BASE_DIR)
categories = {
    'Bad fruit': os.path.join(dataset_dir, 'Bad fruit'),
    'Bad leaf': os.path.join(dataset_dir, 'Bad leaf'),
    'Good fruit': os.path.join(dataset_dir, 'Good fruit'),
    'Good leaf': os.path.join(dataset_dir, 'Good leaf'),
}

NUM_SAMPLES = 5  # Number of images per class


print("=" * 70)
print("PIPELINE-ALIGNED ANALYSIS")
print("=" * 70)


# =============================================================================
# BASIC PREPROCESSING FUNCTIONS
# =============================================================================

def resize_gray(image):
    """Convert to grayscale and resize"""
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, IMG_SIZE, interpolation=cv.INTER_AREA)
    return image


def pipeline_ours(image):
    """Your proposed pipeline"""
    image = resize_gray(image)
    kernel = gaussFilter((3, 3), 1)
    image = cv.filter2D(image, -1, kernel)
    image = adaptive_mean_filter(image, 3)
    return image


def pipeline_hist_eq(image):
    """Histogram Equalization"""
    image = resize_gray(image)
    return cv.equalizeHist(image)


def pipeline_clahe(image):
    """CLAHE enhancement"""
    image = resize_gray(image)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def pipeline_median(image):
    """Median filtering"""
    image = resize_gray(image)
    return cv.medianBlur(image, 3)


# All pipelines grouped
pipelines = {
    'Our Pipeline': pipeline_ours,
    'Histogram Eq': pipeline_hist_eq,
    'CLAHE': pipeline_clahe,
    'Median': pipeline_median,
}


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def calculate_psnr(original, processed):
    """Peak Signal-to-Noise Ratio"""
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(original, processed):
    """Structural Similarity Index"""
    return ssim(original, processed, data_range=255)


def calculate_mse(original, processed):
    """Mean Squared Error"""
    return np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)


def calculate_entropy(image):
    """Entropy (information content)"""
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-10)
    return entropy(hist + 1e-10)


# =============================================================================
# HISTOGRAM PLOTTING FUNCTION
# =============================================================================

def plot_histograms(original, processed_dict, category):
    """
    Plot histograms for original and processed images
    """

    methods = ["Original"] + list(processed_dict.keys())
    images = [original] + list(processed_dict.values())

    plt.figure(figsize=(18, 4))

    for i, (name, img) in enumerate(zip(methods, images)):
        plt.subplot(1, len(images), i + 1)

        plt.hist(img.ravel(), bins=256, color='black', alpha=0.7)

        plt.title(name, fontsize=10, fontweight='bold')
        plt.xlabel("Intensity (0–255)", fontsize=8)
        plt.ylabel("Frequency", fontsize=8)

        plt.xticks([0, 64, 128, 192, 255])

    plt.suptitle(f"Histogram Comparison - {category}", fontsize=14, fontweight='bold')

    plt.subplots_adjust(top=0.80, wspace=0.4)

    # FIXED SAVE PATH
    plt.savefig(
        os.path.join(results_dir, f"hist_{category.replace(' ','_')}.png"),
        dpi=200,
        bbox_inches='tight'
    )

    plt.show()


# =============================================================================
# MAIN ANALYSIS LOOP
# =============================================================================

for category, path in categories.items():

    files = os.listdir(path)
    np.random.shuffle(files)
    sample_files = files[:NUM_SAMPLES]

    print(f"\n{'='*70}")
    print(f"Category: {category}")
    print(f"{'='*70}")

    # Store metrics for each pipeline
    metrics = {name: {'psnr': [], 'ssim': [], 'mse': [], 'entropy': []}
               for name in pipelines}

    for idx, file in enumerate(sample_files):

        img_path = os.path.join(path, file)
        original_color = cv.imread(img_path)

        if original_color is None:
            continue

        original = resize_gray(original_color)
        processed_dict = {}

        # ======================================================
        # VISUAL DISPLAY (ONLY FIRST IMAGE)
        # ======================================================

        if idx == 0:
            fig, axes = plt.subplots(1, len(pipelines) + 1, figsize=(22, 5))

            axes[0].imshow(original, cmap='gray')
            axes[0].set_title("Original", fontweight='bold')
            axes[0].axis('off')

        # ======================================================
        # APPLY PIPELINES
        # ======================================================

        for i, (name, func) in enumerate(pipelines.items()):

            processed = func(original_color)
            processed_dict[name] = processed

            # Store metrics
            metrics[name]['psnr'].append(calculate_psnr(original, processed))
            metrics[name]['ssim'].append(calculate_ssim(original, processed))
            metrics[name]['mse'].append(calculate_mse(original, processed))
            metrics[name]['entropy'].append(calculate_entropy(processed))

            # Plot images
            if idx == 0:
                axes[i + 1].imshow(processed, cmap='gray')
                axes[i + 1].set_title(name, fontweight='bold')
                axes[i + 1].axis('off')

        # ======================================================
        # SAVE VISUAL + HISTOGRAM
        # ======================================================

        if idx == 0:
            plt.suptitle(f"Category: {category}", fontweight='bold')

            plt.savefig(
                os.path.join(results_dir, f"visual_{category.replace(' ','_')}.png"),
                dpi=200,
                bbox_inches='tight'
            )

            plt.show()

            plot_histograms(original, processed_dict, category)

    # =============================================================================
    # METRIC SUMMARY
    # =============================================================================

    print("\nAveraged Metrics:")
    print(f"{'Method':<20} {'PSNR':>8} {'SSIM':>8} {'MSE':>10} {'Entropy':>10}")
    print("-" * 60)

    for name in pipelines:
        print(f"{name:<20} "
              f"{np.mean(metrics[name]['psnr']):>8.2f} "
              f"{np.mean(metrics[name]['ssim']):>8.4f} "
              f"{np.mean(metrics[name]['mse']):>10.2f} "
              f"{np.mean(metrics[name]['entropy']):>10.4f}")


print("\n" + "=" * 70)
print("Analysis Complete. Results saved in results/analysis/")
print("=" * 70)