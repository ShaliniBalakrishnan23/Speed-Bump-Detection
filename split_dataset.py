import os
import random
import shutil
from pathlib import Path

# === Base dataset path ===
base_path = Path("C:/Users/shali/SpeedBumpDetection/datasets")

# Input folders
images_path = base_path / "all_images"
labels_path = base_path / "all_labels"

# Collect all image files
image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
random.shuffle(image_files)

# Total images
total_images = len(image_files)
train_count = int(total_images * 0.7)
val_count = int(total_images * 0.2)
test_count = total_images - train_count - val_count

# Split counts
split_dirs = {
    'train': train_count,
    'val': val_count,
    'test': test_count
}

# Prepare output folders
for split in split_dirs:
    img_dir = base_path / split / "images"
    lbl_dir = base_path / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Clear old files if any
    for f in img_dir.glob("*"):
        f.unlink()
    for f in lbl_dir.glob("*"):
        f.unlink()

# Perform splitting
start = 0
for split, count in split_dirs.items():
    end = start + count
    for img_path in image_files[start:end]:
        label_path = labels_path / f"{img_path.stem}.txt"

        shutil.copy(img_path, base_path / split / "images" / img_path.name)
        shutil.copy(label_path, base_path / split / "labels" / label_path.name)

    start = end

# ✅ Summary
print("\n Dataset split completed:")
for split in split_dirs:
    img_count = len(list((base_path / split / "images").glob("*")))
    print(f"  {split}: {img_count} images")
