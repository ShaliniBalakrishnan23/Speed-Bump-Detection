import os
import shutil

def polygon_to_yolo_bbox(points):
    x_coords = points[0::2]
    y_coords = points[1::2]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def process_label_line(parts):
    class_id = parts[0]
    if len(parts) == 5:
        # Already bbox
        return f"{class_id} {float(parts[1]):.6f} {float(parts[2]):.6f} {float(parts[3]):.6f} {float(parts[4]):.6f}\n"
    elif len(parts) == 9:
        # Polygon
        points = list(map(float, parts[1:]))
        x_center, y_center, width, height = polygon_to_yolo_bbox(points)
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    else:
        return None

def convert_folder_labels(folder_path, backup_folder=None):
    if backup_folder and not os.path.exists(backup_folder):
        os.makedirs(backup_folder)

    label_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for filename in label_files:
        input_path = os.path.join(folder_path, filename)

        if backup_folder:
            shutil.copy2(input_path, os.path.join(backup_folder, filename))

        with open(input_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            processed_line = process_label_line(parts)
            if processed_line is None:
                print(f"Skipping line in {filename} (unexpected format): {line.strip()}")
                continue
            new_lines.append(processed_line)

        with open(input_path, 'w') as f:
            f.writelines(new_lines)

        print(f"Converted and overwritten: {input_path}")

if __name__ == "__main__":
    base_folders = [
        r"C:\Users\shali\SpeedBumpDetection\datasets\train\labels",
        r"C:\Users\shali\SpeedBumpDetection\datasets\test\labels",
        r"C:\Users\shali\SpeedBumpDetection\datasets\val\labels"
    ]
    backup_root = r"C:\Users\shali\SpeedBumpDetection\datasets\backup_labels"

    for folder in base_folders:
        folder_name = folder.split("\\")[-2] + "_labels"
        backup_folder = os.path.join(backup_root, folder_name)
        print(f"Processing folder: {folder}")
        convert_folder_labels(folder, backup_folder=backup_folder)

    print("Conversion completed. Originals backed up in backup_labels folder.")
