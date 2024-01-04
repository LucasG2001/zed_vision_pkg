import numpy as np
import open3d as o3d
import time

# Record the start time
start_time = time.time()

# Step 0: Create a random 3x3 camera intrinsics matrix
intrinsics_matrix = np.random.rand(3, 3)

# Step 1: Create a numpy array of size 1280x720 with random values between 0 and 1
image1 = np.random.rand(720, 1280)

# Step 2: Create a second image with an additional dimension for random color vectors
image2 = np.random.rand(720, 1280, 3)

# Record the start time
start_time = time.time()

# Step 3: Extract non-zero points and their coordinates
non_zero_points = np.argwhere(image1 != 0)
uvd_matrix = np.column_stack((non_zero_points[:, 1], non_zero_points[:, 0], image1[non_zero_points[:, 0], non_zero_points[:, 1]]))

# Multiply by intrinsics matrix and transpose if necessary
world_coords = np.dot(uvd_matrix, intrinsics_matrix.T)

# Step 4: Extract colors at those locations
colors = image2[non_zero_points[:, 0], non_zero_points[:, 1]]

# Step 5: Create an empty Open3D point cloud
point_cloud = o3d.geometry.PointCloud()

# Step 6: Assign points and colors to the point cloud
point_cloud.points = world_coords
point_cloud.colors = colors

# Step 7: Visualize the point cloud
#o3d.visualization.draw_geometries([point_cloud])

# Record the end time
end_time = time.time()

# Print the execution time
print(f"Execution time: {end_time - start_time} seconds")
