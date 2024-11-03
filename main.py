import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8s.pt')

# Load the video or image for detection
cap = cv2.VideoCapture('easy1.mp4')

# Load class labels
with open("coco.txt", "r") as f:
    class_list = f.read().splitlines()

# Define parking areas if needed (optional)
areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],  # Area 1
    # ... (additional areas)
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run model predictions
    results = model.predict(frame)

    # Process predictions
    boxes = results[0].boxes.data.cpu().numpy()  # Move to CPU and convert to numpy
    free_spaces = []

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box

        # Assuming the label for free space is 0
        if cls == 0:  # Change this based on your label for 'free'
            free_spaces.append((x1, y1, x2, y2))

            # Draw bounding box for free spaces
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green for free spaces
            cv2.putText(frame, 'Free', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Optionally draw polygons for parking areas (if defined)
    for area in areas:
        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)  # Blue for areas

    # Count and display the number of free spaces
    free_space_count = len(free_spaces)
    cv2.putText(frame, f'Free Spaces: {free_space_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Parking Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cap.release()
cv2.destroyAllWindows()
