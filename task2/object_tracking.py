import cv2
from utils.tracking_utils import initialize_tracker

# Read video
video_path = 'data/videos/input_video.mp4'
cap = cv2.VideoCapture(video_path)

# Read the first frame to initialize the tracker
ret, frame = cap.read()
bbox = cv2.selectROI('Object Tracking', frame, False)
tracker = initialize_tracker(frame, bbox)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    success, bbox = tracker.update(frame)
    if success:
        # Visualization
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)

    # Visualization
    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
