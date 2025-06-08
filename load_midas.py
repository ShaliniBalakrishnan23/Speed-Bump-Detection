import cv2
import torch
import numpy as np
from ultralytics import YOLO

# --- Paths ---
VIDEO_PATH = r"C:\Users\shali\Downloads\Recording 2025-05-28 101252.mp4"
MODEL_PATH = r"C:\Users\shali\SpeedBumpDetection\runs\detect\train25\weights\best.pt"
SAVE_VIDEO = True  # Set to True to save output video

# --- Hyperparameters ---
CONFIDENCE_THRESHOLD = 0.7
ALERT_DISTANCE_METERS = 8.0

# --- Load YOLOv8 Model ---
yolo_model = YOLO(MODEL_PATH)

# --- Load MiDaS Model for Depth ---
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# --- Load MiDaS Transform ---
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if "DPT" in str(midas) else midas_transforms.small_transform

# --- Load Video ---
cap = cv2.VideoCapture(VIDEO_PATH)
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- Output Writer ---
if SAVE_VIDEO:
    out = cv2.VideoWriter("output_speedbump_depth.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Run YOLOv8 Detection ---
    results = yolo_model(frame, conf=CONFIDENCE_THRESHOLD, iou=0.3)  # Adjust iou if needed

    # --- Run MiDaS Depth Estimation ---
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bilinear",
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    # --- Visualize Depth Map ---
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

# Show depth map (optional)
    cv2.imshow("Raw Depth Map", depth_colored)


    has_valid_detection = False  # Track if this frame has a good detection

    # --- Process Detections ---
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0].item())

            # --- Get bounding box coordinates ---
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # --- Depth Estimation in the bbox region ---
            depth_roi = depth_map[y1:y2, x1:x2]
            if depth_roi.size > 0:
                depth_roi = np.clip(depth_roi, 0, np.percentile(depth_roi, 95))  # Remove outliers
                avg_depth = np.median(depth_roi)
            else:
                avg_depth = 0

            has_valid_detection = True

            # --- Draw BBox and Depth Label ---
            label = f"{yolo_model.names[class_id]} ~{avg_depth:.2f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Distance Alert if Bump is Close ---
            if avg_depth < ALERT_DISTANCE_METERS:
                cv2.putText(frame, "!! BUMP AHEAD !!", (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

            print(f"Detected: {yolo_model.names[class_id]} | Confidence: {conf:.2f} | Depth: {avg_depth:.2f}m")

    # --- Show and Save Frame ---
    cv2.imshow("Speed Bump Detection + Depth", frame)
    if SAVE_VIDEO and has_valid_detection:
        out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
if SAVE_VIDEO:
    out.release()   
cv2.destroyAllWindows()


