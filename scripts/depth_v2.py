import cv2
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import matplotlib.pyplot as plt
import numpy as np 
import torch
import time


from depth_anything_v2.dpt import DepthAnythingV2

# Visualize and save the depth map
def visualize_depth_map(depth_map, output_path="depth_map.png"):
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())  # Normalize to [0, 1]
    plt.figure(figsize=(10, 8))
    plt.imshow(depth_normalized, cmap="plasma")  # Use plasma colormap for better visualization
    plt.colorbar(label="Depth (normalized)")
    plt.title("Depth Map")
    plt.axis("off")
    #plt.savefig(output_path)  # Save the depth map as an image
    plt.show()

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'/Users/shravan/XR_Final_project/depth_anything_v2/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


raw_img = cv2.imread('/Users/shravan/XR_Final_project/final data/datasets/Motorcycle-perfect/im0.png')

# Run inference.
start_time = time.time()  # Start timing
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
end_time = time.time()  # End timing
print(f"Depth map computation took {end_time - start_time:.4f} seconds.")

print(f"Depth map min value: {depth.min()}")  # Minimum depth value
print(f"Depth map max value: {depth.max()}")  # Maximum depth value

depth = depth * 1000 #conversion from meters to millimeters
cv2.imwrite("/Users/shravan/XR_Final_project/final data/scripts/depth_anything_v2_depth_map.png", depth.astype(np.uint16))
print("depth_anything_v2 depth map saved as 16-bit PNG.")

depth_map_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
cv2.imshow("Depth Map (Colored) (depth_anything_v2)", depth_map_colored)
cv2.imwrite("Depth_map_colored_depth_anything_v2.png", depth_map_colored)


#visualize_depth_map(depth, output_path="output_depth_map.png")


cv2.waitKey(0)
cv2.destroyAllWindows()
