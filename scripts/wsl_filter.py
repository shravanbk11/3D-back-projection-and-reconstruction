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


# Create a StereoSGBM matcher
min_disparity = 16
nDispFactor = 14
num_disparities = 16 * nDispFactor - min_disparity  # Must be divisible by 16
block_size = 7  # Matching block size

left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size**2,  # Controls the disparity smoothness
    P2=32 * 3 * block_size**2,  # Controls the disparity smoothness
    disp12MaxDiff=1,  # Maximum allowed difference in the left-right disparity check
    uniquenessRatio=15,  # Margin by which the best (minimum) computed cost function should “win”
    speckleWindowSize=0,  # Maximum size of smooth disparity regions
    speckleRange=2,  # Maximum disparity variation within each connected component
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

'''
# Version 2 Create StereoSGBM matcher for left and right disparity maps
min_disparity = 16
nDispFactor = 14
num_disparities = 16*nDispFactor-min_disparity  # Must be divisible by 16
block_size = 7  # Matching block size

left_matcher = cv2.StereoSGBM_create(
    minDisparity=min_disparity,
    numDisparities=num_disparities,
    blockSize=block_size,
    P1=8 * 3 * block_size**2,  # Controls the disparity smoothness
    P2=32 * 3 * block_size**2,  # Controls the disparity smoothness
    disp12MaxDiff=10,  # Maximum allowed difference in the left-right disparity check
    uniquenessRatio=5,  # Margin by which the best (minimum) computed cost function should “win”
    speckleWindowSize=100,  # Maximum size of smooth disparity regions
    speckleRange=16,  # Maximum disparity variation within each connected component
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
'''
# Create the right matcher for WLS filtering
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# Create the WLS filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)  # Regularization parameter
wls_filter.setSigmaColor(1.5)  # Edge-awareness parameter

# Compute the disparity maps
start_time = time.time()
left_disparity = left_matcher.compute(left_image, right_image).astype(np.float32) / 16.0
right_disparity = right_matcher.compute(right_image, left_image).astype(np.float32) / 16.0
print(f"Disparity computation took {time.time() - start_time:.4f} seconds.")

# Apply WLS filter
start_time = time.time()
filtered_disparity = wls_filter.filter(left_disparity, left_image, disparity_map_right=right_disparity)
print(f"WLS filtering took {time.time() - start_time:.4f} seconds.")


# Compute the depth map using the calibration parameters
depth_map = (baseline * focal_length) / (filtered_disparity + doffs)

# Normalize the depth map for visualization (convert to 8-bit for display)
filtered_disparity_normalized = cv2.normalize(filtered_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

# Display the results
cv2.imshow("Left Image", left_image)
cv2.imshow("Right Image", right_image)
cv2.imshow("Disparity Map (Filtered)", filtered_disparity_normalized)
cv2.imshow("Depth Map (Colored)", depth_map_colored)

# Save results
#cv2.imwrite("refined_disparity_map.png", filtered_disparity_normalized)
cv2.imwrite("depth_map_colored_ver_1.png", depth_map_colored)

cv2.waitKey(0)
cv2.destroyAllWindows()