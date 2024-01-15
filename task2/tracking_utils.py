import cv2

def initialize_tracker(frame, bbox):
    # Initialize object tracker
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bbox)
    return tracker
