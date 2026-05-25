# -*- coding: utf-8 -*-

import os
import shutil
import random

# =========================================================
# PATHS
# =========================================================

# ORIGINAL RAW DATASET
SOURCE_DIR = 'dataset/raw'

# AUGMENTED DATASET
AUGMENTED_DIR = 'dataset/augmented'

# OUTPUT SPLITS
TRAIN_DIR = 'dataset/train'
VAL_DIR   = 'dataset/val'
TEST_DIR  = 'dataset/test'

# =========================================================
# SPLIT RATIOS
# =========================================================

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# =========================================================
# CLASSES
# =========================================================

categories = [
    'Bad fruit',
    'Bad leaf',
    'Good fruit',
    'Good leaf'
]

# =========================================================
# CREATE FOLDERS
# =========================================================

for category in categories:
    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)

print("=" * 60)
print("SAFE DATASET SPLITTING")
print("Raw → Train / Val / Test")
print("Augmented → Train Only")
print("=" * 60)

# =========================================================
# SPLIT LOGIC
# =========================================================

for category in categories:

    # -----------------------------------------------------
    # RAW IMAGES
    # -----------------------------------------------------

    src_folder = os.path.join(SOURCE_DIR, category)

    images = [
        img for img in os.listdir(src_folder)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end   = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_images = images[:train_end]
    val_images   = images[train_end:val_end]
    test_images  = images[val_end:]

    # =====================================================
    # COPY ORIGINAL IMAGES
    # =====================================================

    # TRAIN
    for img in train_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(TRAIN_DIR, category, img)
        )

    # VALIDATION
    for img in val_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(VAL_DIR, category, img)
        )

    # TEST
    for img in test_images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(TEST_DIR, category, img)
        )

    # =====================================================
    # ADD AUGMENTED IMAGES TO TRAIN ONLY
    # =====================================================

    aug_folder = os.path.join(AUGMENTED_DIR, category)

    augmented_added = 0

    if os.path.exists(aug_folder):

        aug_images = os.listdir(aug_folder)

        for img in train_images:

            # remove extension
            base_name = os.path.splitext(img)[0]

            # find matching augmented versions
            matching_aug = [
                aug for aug in aug_images
                if aug.startswith(base_name + "_aug")
            ]

            for aug_img in matching_aug:

                shutil.copy(
                    os.path.join(aug_folder, aug_img),
                    os.path.join(TRAIN_DIR, category, aug_img)
                )

                augmented_added += 1

    # =====================================================
    # FINAL COUNTS
    # =====================================================

    final_train = len(os.listdir(os.path.join(TRAIN_DIR, category)))
    final_val   = len(os.listdir(os.path.join(VAL_DIR, category)))
    final_test  = len(os.listdir(os.path.join(TEST_DIR, category)))

    print(f"\n{category}")
    print(f"Original Train Images : {len(train_images)}")
    print(f"Augmented Added       : {augmented_added}")
    print(f"Final Train Count     : {final_train}")
    print(f"Validation Count      : {final_val}")
    print(f"Test Count            : {final_test}")

print("\n" + "=" * 60)
print("Done splitting dataset.")
print("Validation/Test untouched.")
print("No augmentation leakage.")
print("=" * 60)