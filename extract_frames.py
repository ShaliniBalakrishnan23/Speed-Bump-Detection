import cv2
import os

video_path = r"C:\Users\shali\Downloads\Recording 2025-05-28 101252.mp4"
output_dir = r"C:\Users\shali\SpeedBumpDetection\extracted_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 0.5)  # extract every 0.5 seconds

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames.")
