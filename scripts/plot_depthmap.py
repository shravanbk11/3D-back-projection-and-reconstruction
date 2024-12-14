import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Paths to the depth maps
base_path = "/Users/shravan/XR_Final_project/final data/scripts/"
files = {
    "Stereo BM (absolute)": "BM_depth_map.png",
    "Stereo SGBM (absolute)": "SGBM_depth_map.png",
    "Depth Anything v2 (relative)": "depth_anything_v2_depth_map.png",
    "ML Depth Pro (absolute)": "ml_depth_pro_depth_map.png",
}

# Function to load and preprocess depth maps
def load_depth_map(file_path):
    depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise FileNotFoundError(f"Depth map not found: {file_path}")
    
    depth_map = depth_map.astype(np.float32)
    depth_map[~np.isfinite(depth_map)] = np.nan  # Handle invalid values
    depth_map[depth_map <= 0] = np.nan  # Replace non-positive values with NaN
    return depth_map

# Load the depth maps
depth_maps = {name: load_depth_map(os.path.join(base_path, path)) for name, path in files.items()}

# Separate depth maps into two rows
top_row = ["Stereo BM (absolute)", "Stereo SGBM (absolute)"]
bottom_row = ["Depth Anything v2 (relative)", "ML Depth Pro (absolute)"]

# Create a 2x2 plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot the top row
for idx, name in enumerate(top_row):
    depth_map = depth_maps[name]
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    axes[0, idx].imshow(depth_map_normalized, cmap="jet")
    axes[0, idx].set_title(name, fontsize=14, fontweight="bold")
    axes[0, idx].axis("off")

# Plot the bottom row
for idx, name in enumerate(bottom_row):
    depth_map = depth_maps[name]
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    axes[1, idx].imshow(depth_map_normalized, cmap="jet")
    axes[1, idx].set_title(name, fontsize=14, fontweight="bold")
    axes[1, idx].axis("off")

# Adjust layout
plt.tight_layout()

# Save the figure
output_dir = "/Users/shravan/XR_Final_project/final data/scripts/"
output_image_path = os.path.join(output_dir, "depth_maps_comparison.png")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Save the image
plt.savefig(output_image_path, dpi=300)
print(f"Comparison image saved at {output_image_path}")

# Show the plot
plt.show()