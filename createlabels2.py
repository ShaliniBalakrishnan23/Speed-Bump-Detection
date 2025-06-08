import os

# Directory where you want to create the .txt files
output_dir = r"C:\Users\shali\SpeedBumpDetection\datasets\all_labels"

# List of filenames (without extension)
filenames = [
    "000001", "000011", "000021", "000031", "000041", "000051",
    "000061", "000071", "000081", "000091", "000101", "000111",
    "000121", "000131", "000141", "000151", "000161", "000171",
    "000181", "000191", "000201", "000211", "000221", "000231",
    "000241", "000261", "000271", "000281", "000291", "000301",
    "000311", "000321", "000331", "000341", "000351", "000361",
    "000371", "000381", "000391", "000401", "000411", "000421",
    "000431", "000441", "000451", "000461", "000471", "000481",
    "000491", "000501", "000511", "000521", "000531", "000541",
    "000551", "000561", "000571", "000581", "000591", "000601"
]

# Create each empty .txt file
for name in filenames:
    file_path = os.path.join(output_dir, f"{name}.txt")
    with open(file_path, 'w') as f:
        pass  # Creates an empty file
    print(f"Created: {file_path}")
