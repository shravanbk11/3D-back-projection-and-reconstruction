import numpy as np
from skimage import measure
import open3d as o3d
import cv2
from skimage.measure import block_reduce


def compute_tsdf(depth_map, voxel_size, truncation_distance, max_depth=1.0):
    """
    Compute a TSDF volume from a depth map.
    :param depth_map: 2D numpy array representing the depth map.
    :param voxel_size: The size of each voxel in meters.
    :param truncation_distance: Truncation distance for the TSDF.
    :param max_depth: Maximum depth value to consider (for limiting volume size).
    :return: TSDF volume as a 3D numpy array.
    """
    h, w = depth_map.shape
    depth_map = np.clip(depth_map, 0, max_depth)  # Clip depth to max range
    volume_dim = (h, w, int(max_depth / voxel_size))  # Limit volume depth

    tsdf_volume = np.ones(volume_dim, dtype=np.float32)  # Initialize with +1 (free space)

    for z in range(volume_dim[2]):
        z_world = z * voxel_size
        tsdf_layer = (depth_map - z_world) / truncation_distance
        tsdf_layer = np.clip(tsdf_layer, -1, 1)
        tsdf_volume[:, :, z] = np.minimum(tsdf_volume[:, :, z], tsdf_layer)

    return tsdf_volume


# Load depth map
depth_map_path = "/Users/shravan/XR_Final_project/final data/scripts/BM_results/Depth_map_colored_fgs.png"  # Replace with your file path
depth_image = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

# Normalize depth map to 0-1 range
depth_image = depth_image.astype(np.float32) / 255.0

# Resize depth map to reduce resolution
depth_image = cv2.resize(depth_image, (depth_image.shape[1] // 2, depth_image.shape[0] // 2))

# TSDF parameters
voxel_size = 0.05  # Larger voxel size reduces resolution
truncation_distance = 0.1
max_depth = 1.0

# Compute TSDF volume
tsdf_volume = compute_tsdf(depth_image, voxel_size, truncation_distance, max_depth)

# Validate TSDF volume
assert np.isfinite(tsdf_volume).all(), "TSDF volume contains NaN or infinite values."
assert np.unique(tsdf_volume).size > 1, "TSDF volume has no variations."

# Downsample TSDF volume
tsdf_volume = block_reduce(tsdf_volume, block_size=(2, 2, 2), func=np.mean)

# Extract mesh using Marching Cubes
verts, faces, normals, values = measure.marching_cubes(tsdf_volume, level=0)

# Debugging: Validate vertices and faces
print("Vertices shape:", verts.shape)
print("Faces shape:", faces.shape)
assert verts.shape[0] > 0, "Vertices array is empty."
assert faces.shape[0] > 0, "Faces array is empty."

# Ensure correct data types
verts = verts.astype(np.float64)  # Vertices must be float64
faces = faces.astype(np.int32)   # Faces must be int32

# Ensure face indices are within bounds
assert faces.max() < verts.shape[0], "Face indices exceed vertex array bounds."

# Create Open3D mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# Simplify the mesh
mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=20000)  # Target 20k triangles

mesh.remove_unreferenced_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()

# Optional: Compute normals for better visualization
mesh.compute_vertex_normals()

# Save and visualize
output_mesh_path = "Stereo_BM_mesh.stl"
o3d.io.write_triangle_mesh(output_mesh_path, mesh)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)