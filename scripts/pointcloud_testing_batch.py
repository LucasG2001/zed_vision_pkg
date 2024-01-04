import numpy as np
import open3d as o3d
import time

# Function to generate a random intrinsics matrix
def generate_intrinsics_matrix():
    return np.random.rand(3, 3)

# Function to generate a random image with color information
def generate_random_image():
    return np.random.rand(720, 1280)

# Function to generate a random color image
def generate_random_color_image():
    return np.random.rand(720, 1280, 3)

# Function to create a point cloud from an image and intrinsics matrix
def create_point_cloud(image, color_image, intrinsics_matrix):
    non_zero_points = np.argwhere(image != 0)
    uvd_matrix = np.column_stack((non_zero_points[:, 1], non_zero_points[:, 0], image[non_zero_points[:, 0], non_zero_points[:, 1]]))
    world_coords = np.dot(uvd_matrix, intrinsics_matrix.T)
    colors = color_image[non_zero_points[:, 0], non_zero_points[:, 1]]
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(world_coords)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud

# Record the start time
start_time = time.time()

# Generate arrays of intrinsics matrices, images, and point clouds
num_images = 60
intrinsics_matrices = [generate_intrinsics_matrix() for _ in range(num_images)]
images = [generate_random_image() for _ in range(num_images)]
color_images = [generate_random_color_image() for _ in range(num_images)]
# Record the start time
start_time = time.time()

point_clouds = [create_point_cloud(images[i], color_images[i], intrinsics_matrices[i]) for i in range(num_images)]


# Record the end time
end_time = time.time()

# Print the total execution time
print(f"Total execution time: {end_time - start_time} seconds")
