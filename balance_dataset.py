from pathlib import Path

# Path to your training label directory
label_dir = Path("C:/Users/shali/SpeedBumpDetection/datasets/train/labels")
image_dir = Path("C:/Users/shali/SpeedBumpDetection/datasets/train/images")

with_bumps = []
without_bumps = []

for label_file in label_dir.glob("*.txt"):
    lines = label_file.read_text().strip()
    img_name = label_file.stem + ".jpg"  # or .png depending on your format
    img_path = image_dir / img_name
    if lines:
        with_bumps.append(img_path)
    else:
        without_bumps.append(img_path)

print(f" Images with speed bumps: {len(with_bumps)}")
print(f" Images without speed bumps: {len(without_bumps)}")
