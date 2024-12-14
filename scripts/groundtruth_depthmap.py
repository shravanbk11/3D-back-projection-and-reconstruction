import numpy as np
import cv2
import matplotlib.pyplot as plt
import struct

# Calibration parameters from calib.txt
baseline = 193.001  # Baseline in mm
focal_length = 3979.911  # Focal length in pixels (same for both cameras)
doffs = 124.343  # Disparity offset


# Load the image with IMREAD_UNCHANGED
disparity_map = cv2.imread("/Users/shravan/XR_Final_project/final data/datasets/Motorcycle-perfect/disp0.pfm", cv2.IMREAD_UNCHANGED)

# Check properties
print(f"Shape: {disparity_map.shape}, Data Type: {disparity_map.dtype}")

# Handle invalid disparities (e.g., 0 or negative values)
#disparity_map[disparity_map <= 0] = np.nan  # Mark invalid disparities as NaN

# Compute the depth map using Z = baseline * f / (d + doffs)
depth_map = baseline * focal_length / (disparity_map + doffs)

# Save raw depth map in millimeters (16-bit PNG)
cv2.imwrite("ground_truth_depth_map.png", depth_map.astype(np.uint16))
print("Raw depth map saved as 16-bit PNG.")


# Normalize the depth map for visualization
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# Visualize the depth map
plt.figure(figsize=(10, 6))
plt.title("Ground Truth Depth Map")
plt.imshow(depth_map_normalized, cmap="jet")
plt.colorbar(label="Depth (m)")
plt.show()

