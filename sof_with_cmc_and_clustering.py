import cv2
import numpy as np
import glob
import os

from scipy.stats import kurtosis
from sklearn.cluster import DBSCAN


def plot_points(image, points, radius=3, color=(0,255,0)):
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), radius, color, thickness=-1)
    return image

def get_color(index):
    """ Converts an integer index to a color """
    blue = int(index * 30 % 256)
    green = int(index * 103 % 256)
    red = int(index * 50 % 256)
    return red, blue, green

def cluster_to_bbox(cluster_points):
    x_coords = cluster_points[:, 0]
    y_coords = cluster_points[:, 1]
    x_min = int(np.min(x_coords))
    y_min = int(np.min(y_coords))
    x_max = int(np.max(x_coords))
    y_max = int(np.max(y_coords))
    return (x_min, y_min, x_max, y_max)


sequence_dir = "./../datasets/VisDrone2019-VID-dataset-marked/uav0000355_00001_v"
image_paths = sorted(glob.glob(os.path.join(sequence_dir, "*.jpg")))

previous_frame = cv2.imread(image_paths[0])
frame_index = 0

for current_image_path in image_paths[1:]:
    frame_index += 1

    current_frame = cv2.imread(current_image_path)

    # Convert frames to grayscale
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    # Detect features
    features_prev = cv2.goodFeaturesToTrack(previous_gray, 5000, qualityLevel=0.05, minDistance=10)

    features_next, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, features_prev, None)

    valid_prev = features_prev[status == 1]
    valid_next = features_next[status == 1]

    max_points = 5000
    if max_points > valid_prev.shape[0]:
        max_points = valid_prev.shape[0]

    selected_indices = np.random.choice(valid_prev.shape[0], size=max_points, replace=False)
    affine_matrix, _ = cv2.estimateAffine2D(valid_prev[selected_indices], valid_next[selected_indices], method=cv2.RANSAC)

    affine_matrix = np.vstack((affine_matrix, np.zeros((1, 3))))  # Convert to 3x3

    # Compensate camera motion
    homogeneous_prev = np.hstack((valid_prev, np.ones((len(valid_prev), 1))))
    motion_compensated = homogeneous_prev @ affine_matrix.T
    motion_compensated = motion_compensated[:, :2]

    # Compute optical flow vectors
    original_flow = valid_next - valid_prev
    residual_flow = valid_next - motion_compensated

    flow_magnitude = np.linalg.norm(residual_flow, ord=2, axis=1)

    outlier_factor = 2
    if kurtosis(flow_magnitude, bias=False) < 1:
        outlier_factor /= 2

    motion_threshold = np.mean(flow_magnitude) + outlier_factor * np.std(flow_magnitude, ddof=1)

    motion_mask = (flow_magnitude >= motion_threshold)
    motion_keypoints = valid_next[motion_mask]

    motion_vectors = motion_keypoints - motion_compensated[motion_mask]
    vector_magnitudes = np.linalg.norm(motion_vectors, ord=2, axis=1)
    vector_angles = np.arctan2(motion_vectors[:, 0], motion_vectors[:, 1])

    motion_features = np.hstack((motion_keypoints, np.c_[vector_magnitudes], np.c_[vector_angles]))

    # DBSCAN clustering
    dbscan_model = DBSCAN(eps=30.0, min_samples=3)
    dbscan_model.fit(motion_features)

    # Filter clusters
    valid_clusters = []
    border_threshold = 20
    frame_height, frame_width, _ = previous_frame.shape
    angle_variance_threshold = 0.2
    max_cluster_members = 50

    frame_border = np.array([frame_width - border_threshold, frame_height - border_threshold])

    for cluster_label in np.unique(dbscan_model.labels_):
        cluster_mask = dbscan_model.labels_ == cluster_label
        angle_std = vector_angles[cluster_mask].std(ddof=1)

        if angle_std <= angle_variance_threshold:
            cluster_points = motion_keypoints[cluster_mask]
            centroid = cluster_points.mean(axis=0)

            if (len(cluster_points) < max_cluster_members) and \
               not (np.any(centroid < border_threshold) or np.any(centroid > frame_border)):
                valid_clusters.append(cluster_points)

    previous_frame = current_frame.copy()

    # Draw clusters and bounding boxes
    for cluster_id, cluster_points in enumerate(valid_clusters):
        cluster_color = get_color((cluster_id + 1) * 5)

        current_frame = plot_points(current_frame, cluster_points, radius=4, color=cluster_color)

        x1, y1, x2, y2 = cluster_to_bbox(cluster_points)
        cv2.rectangle(current_frame, (x1, y1), (x2, y2), cluster_color, 2)

    cv2.imshow('Moving Object Detection & Classification', current_frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('p'):
        print("Paused. Press any key to continue...")
        while True:
            if cv2.waitKey(0) & 0xFF != 255:
                print("Resuming...")
                break
    elif key == ord('q') or key == 27:
        print("Stopping the program...")
        break

cv2.destroyAllWindows()
