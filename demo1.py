import cv2
import numpy as np
import glob
import os
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

# Load YOLOv8 detection model
model = YOLO("yolov8n.pt")

# Path to the image sequence folder
sequence_path = "./../datasets/VisDrone2019-VID-dataset/sequences/uav0000020_00406_v"
image_files = sorted(glob.glob(os.path.join(sequence_path, "*.jpg")))

# Read the first frame and convert to grayscale
old_frame = cv2.imread(image_files[0])
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=500, qualityLevel=0.05, minDistance=4, blockSize=9)

# Detect initial feature points
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(9, 9), maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))

frame_count = 0

for image_file in image_files[1:]:
    frame_count += 1

    # Process only every 2th frame
    if frame_count % 2 != 0:

        continue

    frame = cv2.imread(image_file)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Re-detect feature points
    if p0 is None or len(p0) < 10:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Calculate optical flow
    p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is not None:
        good_new = p1[status == 1]
        good_old = p0[status == 1]
        
        # Estimate global motion using homography
        H, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
        
        # Transform old points to compensate for camera motion
        transformed_old = cv2.perspectiveTransform(good_old.reshape(-1, 1, 2), H).reshape(-1, 2)
        motion_vectors = good_new - transformed_old
        
        # Filter small vectors (noise)
        motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)
        valid_indices = motion_magnitudes > 0.5
        filtered_points = good_new[valid_indices]
        
        if len(filtered_points) > 0:
            # Clustering feature points into objects using DBSCAN
            clustering = DBSCAN(eps=20, min_samples=3).fit(filtered_points)
            labels = clustering.labels_
            unique_labels = set(labels)

            detected_objects = []  # List to store detected bounding boxes

            for label in unique_labels:
                if label == -1:  # Ignore noise points
                    continue
                
                cluster_points = filtered_points[labels == label]
                
                # Get bounding box around cluster
                x_min, y_min = np.min(cluster_points, axis=0)
                x_max, y_max = np.max(cluster_points, axis=0)
                
                # Store detected object
                detected_objects.append((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Run YOLOv8 detection on the full frame
            yolo_results = model(frame)

            for result in yolo_results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # YOLO detection box
                    cls_id = int(box.cls[0])  # Class ID
                    conf = float(box.conf[0])  # Confidence score
                    label = f"{result.names[cls_id]} {conf:.2f}"  # Class label

                    # Check if YOLO detection overlaps with any moving object cluster
                    for (cx1, cy1, cx2, cy2) in detected_objects:
                        if (x1 < cx2 and x2 > cx1 and y1 < cy2 and y2 > cy1):  # Overlapping boxes
                            # Draw bounding box and class label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            break  # Stop checking other clusters once matched

        # Show the frame
        cv2.imshow('Moving Object Detection & Classification', frame)
        
        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
