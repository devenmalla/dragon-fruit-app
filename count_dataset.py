# count_dataset.py

import os

folders = ["train", "val", "test"]

for folder in folders:
    print(f"\n{folder.upper()}")
    print("-" * 30)

    path = os.path.join("dataset", folder)

    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)

        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            print(f"{cls:<15} : {count}")