from PIL import Image
import depth_pro
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import time

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb("/Users/shravan/XR_Final_project/final data/datasets/Motorcycle-perfect/im0.png")
image = transform(image)

# Run inference.
start_time = time.time()  # Start timing
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.
end_time = time.time()  # End timing
print(f"Depth map computation took {end_time - start_time:.4f} seconds.")


# Save raw depth map in millimeters (16-bit PNG)
depth = depth.cpu().numpy()
depth = depth * 1000 #conversion from meters to millimeters
cv2.imwrite("/Users/shravan/XR_Final_project/final data/scripts/ml_depth_pro_depth_map.png", depth.astype(np.uint16))
print("ml_depth_pro depth map saved as 16-bit PNG.")

'''
depth_np = depth
depth_normalized = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))

# Display the depth map using Matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(depth, cmap="jet")  # Use a colormap like 'jet' for better visualization
plt.colorbar(label="Depth (normalized)")
plt.title("Predicted Depth Map")
plt.axis("off")
plt.show()
'''

depth_map_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
cv2.imshow("Depth Map (Colored) (ml_depth_pro)", depth_map_colored)
cv2.imwrite("Depth_map_colored_ml_depth_pro.png", depth_map_colored)

# Print the focal length
print(f"Focal length in pixels: {focallength_px}")

cv2.waitKey(0)
cv2.destroyAllWindows()