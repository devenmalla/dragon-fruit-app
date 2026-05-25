# -*- coding: utf-8 -*-

import os
import cv2

DATASET_DIR = "dataset"

folders = ["train", "val", "test"]

bad_files = []

print("=" * 60)
print("CHECKING FOR CORRUPTED IMAGES (OpenCV)")
print("=" * 60)

for folder in folders:

    folder_path = os.path.join(DATASET_DIR, folder)

    print(f"\nChecking {folder}...")

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                img_path = os.path.join(root, file)

                try:
                    img = cv2.imread(img_path)

                    if img is None:
                        print(f"CORRUPTED: {img_path}")
                        bad_files.append(img_path)

                    else:
                        h, w = img.shape[:2]

                        if h == 0 or w == 0:
                            print(f"INVALID IMAGE: {img_path}")
                            bad_files.append(img_path)

                except Exception:
                    print(f"FAILED TO READ: {img_path}")
                    bad_files.append(img_path)

print("\n" + "=" * 60)

if bad_files:
    print(f"Found {len(bad_files)} corrupted images.\n")

    for f in bad_files:
        print(f)

else:
    print("No corrupted images found.")

print("=" * 60)