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


'''
raw_img = cv2.imread('/Users/shravan/XR_Final_project/final data/datasets/Motorcycle-perfect/im0.png')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy

visualize_depth_map(depth, output_path="output_depth_map.png")
'''

capObj = cv2.VideoCapture(0)



while True:
    start_time = time.time()  
    ret,frame = capObj.read()
    depthMap = model.infer_image(frame) 
    depthMap_normalized = cv2.normalize(depthMap, None, 0, 255, cv2.NORM_MINMAX)
    depthMap_8bit = np.uint8(depthMap_normalized)
    depthMap_inferno = cv2.applyColorMap(depthMap_8bit, cv2.COLORMAP_INFERNO)
    combined = np.hstack((frame, depthMap_inferno))


    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Overlay FPS on the combined output
    cv2.putText(combined, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Combined', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capObj.release()
cv2.destroyAllWindows()

