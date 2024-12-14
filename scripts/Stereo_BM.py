import cv2
import numpy as np
import time

# Calibration parameters from calib.txt
cam0 = np.array([[3979.911, 0, 1244.772], [0, 3979.911, 1019.507], [0, 0, 1]])  # Left camera matrix
cam1 = np.array([[3979.911, 0, 1369.115], [0, 3979.911, 1019.507], [0, 0, 1]])  # Right camera matrix
doffs = 124.343  # Disparity offset
baseline = 193.001  # Baseline in mm
focal_length = 3979.911  # Focal length in pixels (same for both cameras)

# Image dimensions
width, height = 2964, 2000  # Image width and height

# Paths to stereo images
left_image_path = "/Users/shravan/XR_Final_project/final data/datasets/Motorcycle-perfect/im0.png"  # Left stereo image
right_image_path = "/Users/shravan/XR_Final_project/final data/datasets/Motorcycle-perfect/im1.png"  # Right stereo image

# Load stereo images in grayscale
left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

if left_image is None or right_image is None:
    raise FileNotFoundError("Check the paths for the left and right images!")


# Create a StereoBM matcher
num_disparities = 16 * 14  # Must be divisible by 16
block_size = 15  # Matching block size (must be an odd number >= 5)



stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

stereo_start = time.time()

# Compute the disparity map
start_time = time.time()  # Start timing
disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
end_time = time.time()  # End timing
print(f"Disparity computation took {end_time - start_time:.4f} seconds.")


# Apply Fast Global Smoothing (FGS)
print("Applying Fast Global Smoothing...")
lambda_value = 8000  # Regularization parameter
sigma_color = 1.5    # Edge-awareness parameter

start_time = time.time()
fgs_filter = cv2.ximgproc.createFastGlobalSmootherFilter(left_image, lambda_value, sigma_color)
refined_disparity = fgs_filter.filter(disparity)
end_time = time.time()
print(f"FGS applied in {end_time - start_time:.4f} seconds.")


# Compute the depth map using the refined disparity
depth_map = (baseline * focal_length) / (refined_disparity + doffs)

stereo_end = time.time()

print(f" Inference time for the algorithm {stereo_end - stereo_start:.4f} seconds.")

# Save raw depth map in millimeters (16-bit PNG)
cv2.imwrite("BM_depth_map.png", depth_map.astype(np.uint16))
print("BM depth map saved as 16-bit PNG.")

print(f"Shape: {depth_map.shape}, Data Type: {depth_map.dtype}")

# Normalize the depth map for visualization
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
refined_disparity_normalized = cv2.normalize(refined_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

# Display the results
cv2.imshow("Left Image", left_image)
cv2.imshow("Right Image", right_image)
cv2.imshow("Unrefined Disparity Map (Stereo BM)", disparity_normalized)
cv2.imshow("Disparity Map (Refined with FGS) (Stereo BM)", refined_disparity_normalized)
cv2.imshow("Depth Map (Colored) (Stereo BM)", depth_map_colored)

# Save results
cv2.imwrite("BM_results/Disparity_map.png", disparity_normalized)
cv2.imwrite("BM_results/Refined_disparity_map_fgs.png", refined_disparity_normalized)
cv2.imwrite("BM_results/Depth_map_colored_fgs.png", depth_map_colored)

cv2.waitKey(0)
cv2.destroyAllWindows()