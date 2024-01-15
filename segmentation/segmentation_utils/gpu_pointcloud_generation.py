import torch
import cv2
import numpy as np
import time
import open3d as o3d

import os
# Path to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
# Path to the image file
image_path1 = os.path.join(script_directory, 'color_img1.png')
depth_path1 = os.path.join(script_directory, 'depth_img1.png')
image_path2 = os.path.join(script_directory, 'color_img2.png')
depth_path2 = os.path.join(script_directory, 'depth_img2.png')



def create_blobs(visualize=False):
    num_masks = 70
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
    if (visualize):
        # Move the array to CPU (not necessary if you're already running on CPU)
        binary_masks_cpu = binary_masks.astype(np.uint8)

        # Plot one of the masks using OpenCV
        mask_to_plot = binary_masks_cpu[5]  # Change index to plot a different mask

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
    print("Size of stacked_tensor:", stacked_tensor.size())
    return stacked_tensor


def visualize_masked_tensor(masked_batch_grids, binary_masks_tensor_gpu):
    selected_tensor = masked_batch_grids[5]
    selected_tensor = selected_tensor.to_dense()
    selected_image_np = masked_batch_grids.to_dense().cpu().numpy().astype(np.float32)
    # Select one channel of the selected image
    selected_channel = selected_tensor
    selected_channel = selected_tensor.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    # Visualize the selected channel
    cv2.imshow("Selected Channel", (selected_channel * 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Visualize the corresponding mask
    corresponding_mask = binary_masks_tensor_gpu[5].to_dense()  # Assuming the mask tensor has the same batch size
    corresponding_mask_np = corresponding_mask.cpu().numpy().astype(np.uint8)
    a = 1
    cv2.imshow("Corresponding Mask", corresponding_mask_np * 255)
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


def create_pointclouds_from_depth_and_maks(depth_image, color_image, masks):
    color_image_tensor = torch.from_numpy(color_image.transpose(2, 0, 1)).int().cuda()
    masks_tensor = masks.unsqueeze(1).expand(10, 3, -1, -1).to_sparse()
    xyz_tensor = create_stacked_xyz_tensor(depth_image)
    # xyz_tensor = color_image_tensor
    # Expand the stacked tensor to create a batch of size 30 x 3 x 720 x 1280
    stacked_triplet = xyz_tensor.unsqueeze(0).expand(10, 3, -1, -1)
    # Mask the stacked batch with the random mask tensor
    masked_stacked_batch = stacked_triplet * masks_tensor
    # Display the resulting masked stacked batch shape
    print("Shape of the stacked pointcloud tensors:", masked_stacked_batch.shape)
    print("Final Pointclouds are Sparse Tensor ", masked_stacked_batch.is_sparse)

    return masked_stacked_batch


def create_pointcloud_tensor_from_color_and_depth(color_image, depth_image, masks_tensor, visualize=False):
    batch_size = masks_tensor.shape[0]
    color_image_tensor = torch.tensor(color_image.transpose(2, 0, 1).copy(), device='cuda', dtype=torch.int8).cuda().to_sparse()
    mask_list = masks_tensor.reshape(batch_size, 1, -1)
    xyz_tensor = create_stacked_xyz_tensor(depth_image).reshape(1, 3, -1)
    # Expand the stacked tensor to create a batch of size 30 x 3 x 720 x 1280
    stacked_pointcloud_list = xyz_tensor.expand(batch_size, 3, -1) * mask_list.expand(batch_size, 3, -1).to_sparse()
    # print("stacked pointcloud list has shape ", stacked_pointcloud_list.shape, "on ", stacked_pointcloud_list.device)
    # Mask the stacked batch with the random mask tensor
    # Initialize an empty list to store Open3D point clouds
    pointclouds = []
    stacked_pointclouds_np = stacked_pointcloud_list.to_dense().cpu().numpy()
    nonzero = np.nonzero(stacked_pointclouds_np[0, 0])
    # Loop over the batch dimension
    coords = np.zeros((len(nonzero[0]), 3))
    for i in range(batch_size):
        coords[:, 0] = stacked_pointclouds_np[i, 0][nonzero]
        coords[:, 1] = stacked_pointclouds_np[i, 1][nonzero]
        coords[:, 2] = stacked_pointclouds_np[i, 2][nonzero]
        # print("x coords shape is ", x_coords.shape)
        # print("concatenated coordinates have shape ", coords.shape)
        # print("coords nonzero have shape ", torch.nonzero(coords).shape)
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(coords)
        # pointcloud.colors = o3d.utility.Vector3dVector(colors_values.cpu().numpy())    #
        # Append the point cloud to the list
        # pointclouds.append(pointcloud)

    # Visualize the point clouds (optional)
    if visualize:
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
    start = time.time()
    for i in range(10):
        create_pointcloud_tensor_from_color_and_depth(color_image, depth_image, binary_mask_tensor, visualize=False)
        # masked_stack_batch = create_pointclouds_from_depth_and_maks(depth_image, color_image, binary_mask_tensor)
    elapsed_time = time.time() - start
    print("loop took ", elapsed_time, "seconds or ", elapsed_time / 10, " seconds per iteration")
