import cv2
import numpy as np
from utils.detection_utils import load_yolo_model, detect_objects

# Load YOLO model
yolo_model = load_yolo_model('models/yolo/yolov3.weights')

# Read video
video_path = 'data/videos/input_video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    detections = detect_objects(frame, yolo_model)

    # Visualization
    # Replace with your visualization logic
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
