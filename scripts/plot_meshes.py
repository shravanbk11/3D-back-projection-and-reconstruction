import matplotlib.pyplot as plt
import cv2

# Paths to the provided images
base_path = "/Users/shravan/XR_Final_project/final data/scripts/mesh_images/"
images = {
    "Stereo BM (absolute)": f"{base_path}Stereo_BM.png",
    "Stereo SGBM (absolute)": f"{base_path}Stereo_SGBM.png",
    "Depth Anything V2 (relative)": f"{base_path}Depth_anything_V2.png",
    "ML Depth Pro (absolute)": f"{base_path}ml_depth_pro.png"
}

# Load the images
loaded_images = {name: cv2.imread(path, cv2.IMREAD_GRAYSCALE) for name, path in images.items()}

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Plot each image
for idx, (name, img) in enumerate(loaded_images.items()):
    ax = axes[idx // 2, idx % 2]
    if img is not None:
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.axis("off")
    else:
        ax.set_title(f"{name} (not found)", fontsize=14, fontweight="bold")
        ax.axis("off")

# Adjust layout and show
plt.tight_layout()

# Save the plot as an image file
output_path = "/Users/shravan/XR_Final_project/final data/scripts/mesh_images/depth_map_comparison.png"
plt.savefig(output_path)
print(f"Depth map comparison saved to: {output_path}")

# Show the plot
plt.show()