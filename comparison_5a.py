# -*- coding: utf-8 -*-

"""
Phase 5A - Model Comparison
Custom CNN vs MobileNetV2

Purpose:
- Compare Custom CNN and MobileNetV2
- Compare Accuracy, Precision, Recall, F1-score
- Save plots inside results/comparison_5a/
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PATH SETUP
# =========================================================

# Project root directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Results folder
results_base = os.path.join(BASE_DIR, "results")

# Metric file paths
cnn_path = os.path.join(
    results_base,
    "custom_cnn",
    "cnn_metrics.txt"
)

mobilenet_path = os.path.join(
    results_base,
    "mobilenetv2",
    "mobilenetv2_metrics.txt"
)

# Output folder
output_dir = os.path.join(results_base, "comparison_5a")
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# METRIC PARSER
# =========================================================

def parse_metrics(filepath):

    metrics = {}

    if not os.path.exists(filepath):
        print(f"[WARNING] Missing file: {filepath}")
        return metrics

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # -------------------------------------------------
        # Per-class metrics
        # -------------------------------------------------

        try:
            support = int(parts[-1])
            f1 = float(parts[-2])
            recall = float(parts[-3])
            precision = float(parts[-4])

            name = " ".join(parts[:-4])

            if name in [
                'Bad fruit',
                'Bad leaf',
                'Good fruit',
                'Good leaf'
            ]:

                metrics[name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': support
                }

        except:
            pass

        # -------------------------------------------------
        # Accuracy
        # -------------------------------------------------

        if line.startswith("accuracy"):
            try:
                metrics['accuracy'] = float(parts[-2])
            except:
                pass

    return metrics


# =========================================================
# LOAD METRICS
# =========================================================

cnn_metrics = parse_metrics(cnn_path)
mobilenet_metrics = parse_metrics(mobilenet_path)

models = {
    'Custom CNN': cnn_metrics,
    'MobileNetV2': mobilenet_metrics,
}

classes = [
    'Bad fruit',
    'Bad leaf',
    'Good fruit',
    'Good leaf'
]

model_names = list(models.keys())

print("=" * 60)
print("PHASE 5A COMPARISON")
print("Custom CNN vs MobileNetV2")
print("=" * 60)

# =========================================================
# PLOT 1: ACCURACY COMPARISON
# =========================================================

accuracies = [
    m.get('accuracy', 0) * 100
    for m in models.values()
]

plt.figure(figsize=(7, 5))

bars = plt.bar(model_names, accuracies)

for bar, val in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.2,
        f'{val:.2f}%',
        ha='center'
    )

plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")
plt.ylim(80, 100)

plt.savefig(
    os.path.join(output_dir, "5a_accuracy_comparison.png"),
    dpi=200,
    bbox_inches='tight'
)

plt.show()


# =========================================================
# PLOT 2: PRECISION COMPARISON
# =========================================================

precision_scores = []

for model in models.values():

    precision = np.mean([
        model.get(cls, {}).get('precision', 0)
        for cls in classes
    ]) * 100

    precision_scores.append(precision)

plt.figure(figsize=(7, 5))

bars = plt.bar(model_names, precision_scores)

for bar, val in zip(bars, precision_scores):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.2,
        f'{val:.2f}%',
        ha='center'
    )

plt.ylabel("Precision (%)")
plt.title("Precision Comparison")
plt.ylim(80, 100)

plt.savefig(
    os.path.join(output_dir, "5a_precision_comparison.png"),
    dpi=200,
    bbox_inches='tight'
)

plt.show()


# =========================================================
# PLOT 3: RECALL COMPARISON
# =========================================================

recall_scores = []

for model in models.values():

    recall = np.mean([
        model.get(cls, {}).get('recall', 0)
        for cls in classes
    ]) * 100

    recall_scores.append(recall)

plt.figure(figsize=(7, 5))

bars = plt.bar(model_names, recall_scores)

for bar, val in zip(bars, recall_scores):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.2,
        f'{val:.2f}%',
        ha='center'
    )

plt.ylabel("Recall (%)")
plt.title("Recall Comparison")
plt.ylim(80, 100)

plt.savefig(
    os.path.join(output_dir, "5a_recall_comparison.png"),
    dpi=200,
    bbox_inches='tight'
)

plt.show()


# =========================================================
# PLOT 4: F1 SCORE COMPARISON
# =========================================================

f1_scores = []

for model in models.values():

    f1 = np.mean([
        model.get(cls, {}).get('f1', 0)
        for cls in classes
    ]) * 100

    f1_scores.append(f1)

plt.figure(figsize=(7, 5))

bars = plt.bar(model_names, f1_scores)

for bar, val in zip(bars, f1_scores):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.2,
        f'{val:.2f}%',
        ha='center'
    )

plt.ylabel("F1 Score (%)")
plt.title("F1 Score Comparison")
plt.ylim(80, 100)

plt.savefig(
    os.path.join(output_dir, "5a_f1_comparison.png"),
    dpi=200,
    bbox_inches='tight'
)

plt.show()

print("\nSaved comparison plots in:")
print(output_dir)

print("\nGenerated files:")
print("1. 5a_accuracy_comparison.png")
print("2. 5a_precision_comparison.png")
print("3. 5a_recall_comparison.png")
print("4. 5a_f1_comparison.png")