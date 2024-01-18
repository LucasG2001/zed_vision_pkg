import torch
import cv2
import numpy as np
import time
import open3d as o3d
import os
import cProfile
import pstats
from scipy.sparse import coo_matrix

# Path to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
# Path to the image file
image_path1 = os.path.join(script_directory, 'color_img1.png')
depth_path1 = os.path.join(script_directory, 'depth_img1.png')
image_path2 = os.path.join(script_directory, 'color_img2.png')
depth_path2 = os.path.join(script_directory, 'depth_img2.png')

min_bound = np.array([-0.2, -0.6, -0.1])
max_bound = np.array([0.8, 0.6, 0.9])
# cre eate an axis-aligned bounding box
workspace = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def create_blobs(visualize=False):
    num_masks = 30
    height = 720
    width = 1280

    # Initialize an empty tensor
    binary_masks = np.zeros((num_masks, height, width), dtype=np.uint8)

    # Generate random blobs of ones
    for i in range(num_masks):
        num_blobs = np.random.randint(1, 6)  # You can adjust the range as needed
        for _ in range(num_blobs):
            blob_size = np.random.randint(50, 300)  # You can adjust the range as needed
            blob_center = np.random.randint(0, height, size=2)
            blob_mask = np.zeros((height, width), dtype=np.uint8)
            y, x = np.ogrid[-blob_center[0]:height - blob_center[0], -blob_center[1]:width - blob_center[1]]
            mask = x ** 2 + y ** 2 <= (blob_size // 2) ** 2
            blob_mask[mask] = 1
            binary_masks[i] += blob_mask

    # Clip values to ensure binary (0 or 1)
    binary_masks = np.clip(binary_masks, 0, 1)
    # Move the NumPy array to PyTorch Tensor on CPU
    binary_masks_tensor_cpu = torch.from_numpy(binary_masks)
    # Move the PyTorch Tensor to GPU
    binary_masks_tensor_gpu = binary_masks_tensor_cpu.cuda()
    print("shape of masks is ", binary_masks_tensor_gpu.shape, " on ", binary_masks_tensor_gpu.device)
    # Sum over the last two dimensions
    sum_tensor = torch.sum(binary_masks_tensor_gpu, dim=(-2, -1), keepdim=True)
    print(sum_tensor)
    if (visualize):
        # Move the array to CPU (not necessary if you're already running on CPU)

        binary_masks_cpu = binary_masks.astype(np.uint8)
        for i in range(binary_masks_cpu.shape[0]):
            # Plot one of the masks using OpenCV
            mask_to_plot = binary_masks_cpu[i]  # Change index to plot a different mask

            # Create a grayscale image
            image_to_plot = np.zeros((height, width), dtype=np.uint8)
            image_to_plot[mask_to_plot == 1] = 255  # Set ones to white

            # Display the image using OpenCV
            cv2.imshow('Mask to Plot', image_to_plot)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return binary_masks_tensor_gpu


def create_xu_yv_meshgrid(image_height=720, image_width=1280):
    # TODO: add intrinsic as parameter
    # Parameters for camera projection
    cx, cy = image_width / 2.0, image_height / 2.0
    fx, fy, depth_scale = 525.45, 546.3, 1.0
    # Create tensors for u and v coordinates
    u = torch.arange(0, image_width).float().cuda().unsqueeze(0)
    v = torch.arange(0, image_height).float().cuda().unsqueeze(1)
    # print("Size of u:", u.size())
    # print("Size of v:", v.size())
    # Calculate the unscaled x_u and y_v coordinates
    x_u = (u - cx) / fx
    y_v = (v - cy) / fy
    # print("Size of x_u (unscaled x):", x_u.size())
    # print("Size of y_v (unscaled y):", y_v.size())

    return x_u, y_v


def create_stacked_xyz_tensor(np_depth_image):
    depth_tensor = torch.tensor(np_depth_image.copy(), device='cuda', dtype=torch.float32).cuda()
    # Assuming you have an image of size 720x1280
    image_height, image_width = 720, 1280
    # Depth factor (adjust as needed)
    depth_factor = 1.0
    # Scale the depth tensor by the depth factor
    z_tensor = depth_tensor * depth_factor
    # print("Size of z_tensor:", z_tensor.size())
    x_u, y_v = create_xu_yv_meshgrid(image_height, image_width)
    # Broadcast and calculate the final x, y, and z coordinates
    x_coordinates_final = x_u.unsqueeze(0).expand_as(z_tensor.unsqueeze(0)) * z_tensor
    y_coordinates_final = y_v.unsqueeze(0).expand_as(z_tensor.unsqueeze(0)) * z_tensor
    # print("Size of x_coordinates_final:", x_coordinates_final.size())
    # print("Size of y_coordinates_final:", y_coordinates_final.size())
    # Stack x, y, and z coordinates along the batch dimension
    stacked_tensor = torch.cat([x_coordinates_final, y_coordinates_final, z_tensor.unsqueeze(0)], dim=0)
    # print("Size of stacked_tensor:", stacked_tensor.size())
    return stacked_tensor


def visualize_masked_tensor(color_image, binary_masks_tensor_gpu, height=720, width=1280):
    binary_masks_cpu = binary_masks_tensor_gpu.detach().cpu().numpy().astype(np.uint8)
    for i in range(binary_masks_cpu.shape[0]):
        color_to_plot = np.copy(color_image)
        # Plot one of the masks using OpenCV
        mask_to_plot = binary_masks_cpu[i]  # Change index to plot a different mask
        # Create a grayscale image
        image_to_plot = np.zeros((height, width), dtype=np.uint8)
        image_to_plot[mask_to_plot == 1] = 255  # Set ones to white
        color_to_plot[mask_to_plot == 0] = 0
        # Display the image using OpenCV
        cv2.imshow('Mask to Plot', image_to_plot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('Masked image ', color_to_plot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_masked_image(masked_batch_images, binary_masks_tensor_gpu):
    selected_image = masked_batch_images[5]
    selected_image = selected_image.to_dense()
    selected_image_np = masked_batch_images.to_dense().cpu().numpy().astype(np.float32)
    # Select one channel of the selected image
    selected_channel = selected_image
    selected_channel = selected_image.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    # Visualize the selected channel
    cv2.imshow("Selected Channel", selected_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Visualize the corresponding mask
    corresponding_mask = binary_masks_tensor_gpu[5].to_dense()  # Assuming the mask tensor has the same batch size
    corresponding_mask_np = corresponding_mask.cpu().numpy().astype(
        np.uint8)  # Assuming the mask tensor has the same batch size
    a = 1
    cv2.imshow("Corresponding Mask", corresponding_mask_np * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_pointcloud_tensor_from_color_and_depth(color_image, depth_image, masks_tensor, transform, workspace,
                                                  visualize=False):
    batch_size = masks_tensor.shape[0]
    color_image_list = np.reshape(color_image, (-1, 3))
    mask_list = masks_tensor.reshape(batch_size, 1, -1)
    xyz_tensor = create_stacked_xyz_tensor(depth_image).reshape(1, 3, -1)
    depth_tensor = xyz_tensor[:, 2, :]
    xy_grid_np = xyz_tensor[:, 0:2, :].detach().cpu().numpy()
    # mask only the depth tensor, all else would be redundant
    masked_depth_image_list = depth_tensor.expand(batch_size, 1, -1) * mask_list
    # TODO: Downsampling has some unforseen effect on created pointclouds
    # TODO: the index number of the list entry gets divided by the subsampling factor.
    # so when the old list has size n, the new has size n/4 and all corresponding indices will get /4
    masked_depth_image_list = masked_depth_image_list[:, :, ::4]
    color_image_list = color_image_list[::4, :]
    xy_grid_np = xy_grid_np[:, :, ::4]
    # Initialize an empty list to store Open3D point clouds
    pointclouds = []
    masked_depth_image_list_np = masked_depth_image_list.detach().cpu().numpy()  # THIS IS THE BOTTLENECK!!!!
    # Next problem here: This length is static!
    for i in range(batch_size):
        # TODO: still not correct but at least pointclouds are not empty and it's way faster
        nonzero = np.nonzero(masked_depth_image_list_np[i, 0])  # number of nonzero pixels
        # Loop over the batch dimension
        coords = np.zeros((len(nonzero[0]), 3))
        colors = np.zeros((len(nonzero[0]), 3))
        coords[:, 0] = xy_grid_np[0, 0][nonzero]  # x coordinate (maybe we can batch extract this earlier?
        coords[:, 1] = xy_grid_np[0, 1][nonzero]  # y coordinate
        coords[:, 2] = masked_depth_image_list_np[i, 0][nonzero]  # z coordinate
        # TODO: There is a mismatch between the color image size and the mask size, since it was massively downsampled
        colors[:, 0] = color_image_list[:, 0][nonzero] / 255.0  # z coordinate
        colors[:, 1] = color_image_list[:, 1][nonzero] / 255.0  # x coordinate
        colors[:, 2] = color_image_list[:, 2][nonzero] / 255.0  # y coordinate
        # print("x coords shape is ", x_coords.shape)
        # visualize
        # empty = np.zeros((720 * 1280, 3))
        # empty[nonzero, 0:3] = colors[:, 0:3] * 255
        # empty = np.reshape(empty, (720, 1280, 3), order='C').astype(np.uint8)
        # empty = empty[::2, ::2, :]
        # cv2.imshow("constructed color", empty)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # visualize over
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(coords)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        pointcloud.uniform_down_sample(every_k_points=2)
        pointcloud.transform(transform)
        # Append the point cloud to the list
        pointcloud = pointcloud.crop(workspace)
        if len(pointcloud.points) > 100:
        # print("total of ", len(pointcloud.points), " points")
            pointclouds.append(pointcloud)

    # Visualize the point clouds (optional)

    print("length of pointclouds is ", len(pointclouds))

    if visualize:
        # visualize_masked_tensor(color_image, masks_tensor)
        for pc in pointclouds:
            o3d.visualization.draw_geometries([pc])

    return pointclouds


if __name__ == "__main__":  # This is not a function but an if clause !!'
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("using ", DEVICE)

    # Set the dimensions
    color_image = cv2.imread(image_path1)
    depth_image = cv2.imread(depth_path1, cv2.IMREAD_GRAYSCALE) / 100
    binary_mask_tensor = create_blobs(visualize=False)  # returns sparse tensor
    xyz_list = create_stacked_xyz_tensor(depth_image).reshape(1, 3, -1).detach().cpu().numpy()
    profiler = cProfile.Profile()
    profiler.enable()
    start = time.time()
    for i in range(2):
        pointclouds = create_pointcloud_tensor_from_color_and_depth(color_image, depth_image, binary_mask_tensor,
                                                                    transform=np.eye(4, 4), workspace=workspace,
                                                                    visualize=False)
    profiler.disable()
    elapsed_time = time.time() - start
    print("loop took ", elapsed_time, "seconds or ", elapsed_time / 2, " seconds per iteration")

    # Print detailed profiling information

    stats = pstats.Stats(profiler)
    stats.print_stats()
