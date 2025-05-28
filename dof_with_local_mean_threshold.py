import cv2
import os
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time

sequence_path = "./../datasets/VisDrone2019-VID-dataset-marked/uav0000079_00480_v" #uav0000013_00000_v uav0000355_00001_v uav0000020_00406_v uav0000088_00290_v
annotation_path = "./../datasets/VisDrone2019-VID-dataset-marked/uav0000079_00480_v.txt"

image_files = sorted(glob.glob(os.path.join(sequence_path, "*.jpg")))

step = 3

if len(image_files) < step:
    print("Not enough frames in the sequence.")
    exit()

prev_hist = None

for i in range(0, len(image_files) - step, step):
    frame1 = cv2.imread(image_files[i])
    frame2 = cv2.imread(image_files[i + step])

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev=gray1, next=gray2, flow=None, pyr_scale=0.5, levels=3,
        winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # # mag_flat = mag.flatten()
    # # mag_scaled = (mag_flat * 10).astype(int)
    # # mode_mag = Counter(mag_scaled).most_common(1)[0][0] / 10.0
    # # mean_mag = np.mean(mag_flat)
    # # std_mag = np.std(mag_flat)
    
    # # plt.figure(figsize=(10, 5))
    # # plt.hist(mag_flat, bins=50, color='blue', edgecolor='black')
    # # plt.title('Histogram of Flow Magnitudes')
    # # plt.xlabel('Magnitude')
    # # plt.ylabel('Frequency')
    # # plt.axvline(mode_mag, color='red', linestyle='dashed', linewidth=1.5, label=f'Mode: {mode_mag}')
    # # plt.axvline(mean_mag, color='green', linestyle='dotted', linewidth=1.5, label=f'Mean: {mean_mag:.2f}')
    # # plt.axvline(std_mag, color='blue', linestyle='solid', linewidth=1.5, label=f'Std: {std_mag:.2f}')
    # # plt.legend()
    # # plt.grid(True)
    # # plt.tight_layout()
    # # plt.show()

    # # cv2.imshow("Initial motion mask", mag)
    mag_mask = mag < (np.mean(mag))
    # # mag_mask_display = mag >= (np.mean(mag))
    # # mask_display = (mag_mask_display * 255).astype(np.uint8)

    # # cv2.imshow("Motion mask after global mean threshold", mask_display)
    flow[mag_mask] = (0, 0)

    # flow_x, flow_y = cv2.polarToCart(mag, ang)
    # print(np.min(mag), np.max(mag), np.mean(mag))
    # Split flow into horizontal and vertical components
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # Define kernel size (e.g., 15x15)
    kernel_size = 15

    # Compute local mean using a box filter
    mean_x = cv2.blur(flow_x, (kernel_size, kernel_size))
    mean_y = cv2.blur(flow_y, (kernel_size, kernel_size))

    mean_x = cv2.blur(mean_x, (kernel_size, kernel_size))
    mean_y = cv2.blur(mean_y, (kernel_size, kernel_size))
    # Subtract local mean
    flow_x_norm = flow_x - mean_x
    flow_y_norm = flow_y - mean_y

    # Reconstruct normalized flow
    flow_norm = np.stack((flow_x_norm, flow_y_norm), axis=-1)
    mag2, ang2 = cv2.cartToPolar(flow_norm[..., 0], flow_norm[..., 1])
    
    # # cv2.imshow("Motion mask after local mean threshold", mag2)
    flow_norm = cv2.normalize(flow_norm, None, -1, 1, cv2.NORM_MINMAX)

    mag_flat = mag.ravel()
    mag_flat_2 = mag2.ravel()

    # Create interpolated colors (blue → red)
    blue = np.array([255, 0, 0], dtype=np.float32)
    red = np.array([0, 0, 255], dtype=np.float32)

    # Create an empty image to store RGB values
    h, w = flow_norm.shape[:2]
    interp_colors = np.zeros((h, w, 3), dtype=np.float32)

    # Add blue component based on normalized x
    interp_colors[..., 0] = flow_norm[..., 0] * blue[0] + flow_norm[..., 1] * red[0]
    interp_colors[..., 1] = flow_norm[..., 0] * blue[1] + flow_norm[..., 1] * red[1]
    interp_colors[..., 2] = flow_norm[..., 0] * blue[2] + flow_norm[..., 1] * red[2]


    # print(interp_colors[:10, :10])
    # Clip and convert to uint8
    interp_colors = np.clip(interp_colors, 0, 255).astype(np.uint8)

    # Blend with original frame
    overlay = cv2.addWeighted(frame2, 0.5, interp_colors, 0.9, 0)

    cv2.imshow("Motion Magnitude (Blue→Red, Mode Subtracted)", overlay)
    # if cv2.waitKey(30) & 0xFF == 27:
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
