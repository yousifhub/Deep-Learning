import os
import shutil
import random

# =========================
# CONFIG
# =========================

RAW_DATA_DIR = "output"
OUTPUT_DIR = "dataset"
SPLIT_RATIO = 0.8

# =========================
# CREATE OUTPUT STRUCTURE
# =========================

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dir(OUTPUT_DIR)
create_dir(os.path.join(OUTPUT_DIR, "train"))
create_dir(os.path.join(OUTPUT_DIR, "val"))

# =========================
# GROUP FILES BY CLASS
# =========================

files_by_class = {}

for filename in os.listdir(RAW_DATA_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        class_name = filename.split("_")[0].lower()

        if class_name not in files_by_class:
            files_by_class[class_name] = []

        files_by_class[class_name].append(filename)

# =========================
# SPLIT & MOVE FILES
# =========================

for class_name, file_list in files_by_class.items():
    random.shuffle(file_list)

    split_index = int(len(file_list) * SPLIT_RATIO)
    train_files = file_list[:split_index]
    val_files = file_list[split_index:]

    train_class_dir = os.path.join(OUTPUT_DIR, "train", class_name)
    val_class_dir = os.path.join(OUTPUT_DIR, "val", class_name)

    create_dir(train_class_dir)
    create_dir(val_class_dir)

    for file in train_files:
        shutil.copy(
            os.path.join(RAW_DATA_DIR, file),
            os.path.join(train_class_dir, file)
        )

    for file in val_files:
        shutil.copy(
            os.path.join(RAW_DATA_DIR, file),
            os.path.join(val_class_dir, file)
        )

print("Dataset successfully organized!")