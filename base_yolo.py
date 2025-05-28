import cv2
import os
import glob
from ultralytics import YOLO

yolo = YOLO('yolo11n.pt')

sequence_path = "./../datasets/VisDrone2019-VID-dataset-marked/uav0000355_00001_v"


# Get all image files in the sequence folder (assuming .jpg format)
image_files = sorted(glob.glob(os.path.join(sequence_path, "*.jpg")))

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Loop through each image in the sequence
for img_file in image_files:
    frame = cv2.imread(img_file)  # Read image

    if frame is None:
        print(f"Error reading {img_file}")
        continue

    # Run YOLO tracking
    results = yolo.track(frame, stream=True)

    for result in results:
        classes_names = result.names  # Get class names

        # Iterate over detected objects
        for box in result.boxes:
            if box.conf[0] > 0.4:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                cls = int(box.cls[0])  # Get class index
                class_name = classes_names[cls]  # Get class name
                colour = getColours(cls)  # Get color for class

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    cv2.imshow('VisDrone Tracking', frame)

    # if cv2.waitKey(30) & 0xFF == ord('q'):
    #     break

    key = cv2.waitKey(30) & 0xFF

    if key == ord('p'):
        print("Paused. Press any key to continue...")
        while True:
            key2 = cv2.waitKey(0) & 0xFF
            if key2 != 255:
                print("Resuming...")
                break

    elif key == ord('q') or key == 27:
        print("Stopping the program...")
        break

cv2.destroyAllWindows()