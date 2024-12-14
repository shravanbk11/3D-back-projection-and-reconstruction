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

stereo = cv2.StereoSGBM_create(
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
# this is a second version Create a StereoSGBM matcher still need tro be tuned 
min_disparity = 16
nDispFactor = 14
num_disparities = 16*nDispFactor-min_disparity  # Must be divisible by 16
block_size = 7  # Matching block size

stereo = cv2.StereoSGBM_create(
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

# Compute the disparity map
start_time = time.time()  # Start timing
disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
end_time = time.time()  # End timing
print(f"Disparity computation took {end_time - start_time:.4f} seconds.")

# Apply Guided Filter
print("Applying Guided Filter...")
guided_radius = 9  # Radius of the guided filter
guided_eps = 1e-3  # Regularization term (epsilon)
start_time = time.time()

# Use the left image as the guide
refined_disparity = cv2.ximgproc.guidedFilter(left_image, disparity, guided_radius, guided_eps)
end_time = time.time()
print(f"Guided filter applied in {end_time - start_time:.4f} seconds.")

# Handle invalid disparities (negative values)
#refined_disparity[refined_disparity <= 0] = np.nan

# Compute the depth map using the calibration parameters
depth_map = (baseline * focal_length) / (refined_disparity + doffs)

# Normalize the depth map for visualization (convert to 8-bit for display)
refined_disparity_normalized = cv2.normalize(refined_disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

# Display the results
cv2.imshow("Left Image", left_image)
cv2.imshow("Right Image", right_image)
cv2.imshow("Disparity Map (Refined)", refined_disparity_normalized)
cv2.imshow("Depth Map (Colored)", depth_map_colored)

# Save results
#cv2.imwrite("refined_disparity_map.png", refined_disparity_normalized)
#cv2.imwrite("depth_map_colored.png", depth_map_colored)

cv2.waitKey(0)
cv2.destroyAllWindows()