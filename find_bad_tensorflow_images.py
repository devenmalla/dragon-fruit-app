# -*- coding: utf-8 -*-

import os
import tensorflow as tf

# ==========================================
# PATHS
# ==========================================

DATASET_PATH = "dataset"
folders_to_check = ["train", "val", "test"]

bad_files = []

print("=" * 60)
print("CHECKING IMAGES USING TENSORFLOW DECODER")
print("=" * 60)

for folder in folders_to_check:

    folder_path = os.path.join(DATASET_PATH, folder)

    print(f"\nChecking {folder}...")

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            file_path = os.path.join(root, file)

            try:
                image = tf.io.read_file(file_path)
                image = tf.image.decode_jpeg(image)

            except Exception as e:
                bad_files.append(file_path)
                print(f"\nBAD FILE:")
                print(file_path)
                print(f"ERROR: {e}")

print("\n" + "=" * 60)

if len(bad_files) == 0:
    print("No bad TensorFlow images found.")
else:
    print(f"Found {len(bad_files)} problematic image(s).")

print("=" * 60)