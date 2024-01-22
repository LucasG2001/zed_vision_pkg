import torch
import cv2
import numpy as np
import time
import open3d as o3d
import os
import cProfile
import pstats
from segmentation.fastsam import FastSAM, FastSAMPrompt
from segmentation.segmentation_matcher import SegmentationMatcher
from segmentation.segmentation_matcher import SegmentationParameters
from bilateral_filter import apply_bilateral_filter, apply_aggressive_gaussian_filter, apply_aggressive_median_filter

model_path = 'FastSAM-s.pt'
nn_model = FastSAM(model_path)

# Path to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
# Path to the image file
image_path1 = os.path.join(script_directory, 'color_image1.png')
depth_path1 = os.path.join(script_directory, 'depth_image1.png')
image_path2 = os.path.join(script_directory, 'color_image2.png')
depth_path2 = os.path.join(script_directory, 'depth_image2.png')

min_bound = np.array([-0.2, -0.6, -0.1])
max_bound = np.array([0.8, 0.6, 0.9])
# cre eate an axis-aligned bounding box
workspace = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def segment_images(color_images, device, image_size=1024, confidence=0.6, iou=0.95, prompt=False):
    segstart = time.time()
    print("segmenting images and masking on gpu")
    with torch.no_grad():
        results = nn_model(color_images, device=device, retina_masks=True,
                           imgsz=image_size, conf=confidence,
                           iou=iou)
        if prompt:
            prompt_process = FastSAMPrompt(image_path1, results, device=DEVICE)
            # text prompt
            ann = prompt_process.text_prompt(text='laptop')
            print("plotting rsult of text prompt")
            prompt_process.plot(annotations=ann, output_path='output.jpg', )

    print("Segmentation took, ", time.time() - segstart, " seconds")
    mask_tensor1 = results[0].masks.data
    mask_tensor2 = results[1].masks.data
    print("shape of mask tensors is ", mask_tensor1.shape, " and ", mask_tensor2.shape)

    return mask_tensor1, mask_tensor2


def create_random_blobs(visualize=False):
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


def create_xu_yv_meshgrid(intrinsic, image_height=720, image_width=1280):
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


def create_stacked_xyz_tensor(intrinsic, np_depth_image):
    depth_tensor = torch.tensor(np_depth_image.copy(), device='cuda', dtype=torch.float32).cuda()
    # Assuming you have an image of size 720x1280
    image_height, image_width = 720, 1280
    # Depth factor (adjust as needed)
    depth_factor = 1.0
    # Scale the depth tensor by the depth factor
    z_tensor = depth_tensor * depth_factor
    # print("Size of z_tensor:", z_tensor.size())
    x_u, y_v = create_xu_yv_meshgrid(intrinsic, image_height, image_width)
    # Broadcast and calculate the final x, y, and z coordinates
    x_coordinates_final = x_u.unsqueeze(0).expand_as(z_tensor.unsqueeze(0)) * z_tensor
    y_coordinates_final = y_v.unsqueeze(0).expand_as(z_tensor.unsqueeze(0)) * z_tensor
    # print("Size of x_coordinates_final:", x_coordinates_final.size())
    # print("Size of y_coordinates_final:", y_coordinates_final.size())
    # Stack x, y, and z coordinates along the batch dimension
    stacked_tensor = torch.cat([x_coordinates_final, y_coordinates_final, z_tensor.unsqueeze(0)], dim=0)
    # print("Size of stacked_tensor:", stacked_tensor.size())
    return stacked_tensor


# TODO: visualize full segmented image
def visualize_masked_tensor(color_image, binary_masks_tensor_gpu, height=720, width=1280):
    binary_masks_cpu = binary_masks_tensor_gpu.detach().cpu().numpy().astype(np.uint8)
    # for i in range(binary_masks_cpu.shape[0]):
    #    color_to_plot = np.copy(color_image)
    #    # Plot one of the masks using OpenCV
    #    mask_to_plot = binary_masks_cpu[i]  # Change index to plot a different mask
    #    # Create a grayscale image
    #    image_to_plot = np.zeros((height, width), dtype=np.uint8)
    #    image_to_plot[mask_to_plot == 1] = 255  # Set ones to white
    #    color_to_plot[mask_to_plot == 0] = 0
    #    # Display the image using OpenCV
    #    cv2.imshow('Mask to Plot', image_to_plot)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
    #    cv2.imshow('Masked image ', color_to_plot)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

    for i in range(binary_masks_cpu.shape[0]):
        # Plot one of the masks using OpenCV
        mask_to_plot = binary_masks_cpu[i]  # Change index to plot a different mask
        # Create a grayscale image
        image_to_plot = np.zeros((height, width), dtype=np.uint8)
        color_image[mask_to_plot == 1, 0:3] = np.random.randint(0, 256, size=3)  # Set ones to white
        # Display the image using OpenCV
    cv2.imshow('Masked image ', color_image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


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


def preprocessImage(color_image, depth_image, transform, intrinsic, min_bounds, max_bounds):
    xyz_tensor = create_stacked_xyz_tensor(intrinsic, depth_image)
    print("shape of xyz_tensor is ", xyz_tensor.shape)
    xyz_tensor_np = xyz_tensor.detach().cpu().numpy()
    xyz_homogenous = np.vstack((xyz_tensor_np, np.ones((1, xyz_tensor_np.shape[1], xyz_tensor.shape[2]))))
    # Apply homogeneous transformation
    xyz_transformed = np.matmul(transform, xyz_homogenous.reshape(4, -1)).reshape(xyz_homogenous.shape)[:3, :, :]
    # Apply bounds check and mask
    outside_bounds_mask = np.any((xyz_transformed < min_bounds[:, np.newaxis, np.newaxis]) |
                                 (xyz_transformed > max_bounds[:, np.newaxis, np.newaxis]), axis=0)

    # Set pixels to 0 in both the depth and color image where outside of bounds
    xyz_transformed[:, outside_bounds_mask] = 0
    depth_image[outside_bounds_mask] = 0
    color_image[outside_bounds_mask] = 0
    cv2.imshow("cropped color image", color_image)
    cv2.imshow("cropped depth image", depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return color_image, depth_image, xyz_transformed


def create_pointcloud_tensor_from_color_and_depth(color_image, depth_image, masks_tensor, transform, workspace,
                                                  intrinsic,
                                                  visualize=False):
    batch_size = masks_tensor.shape[0]
    color_image_list = np.reshape(color_image, (-1, 3))
    mask_list = masks_tensor.reshape(batch_size, 1, -1)
    xyz_tensor = create_stacked_xyz_tensor(intrinsic, depth_image).reshape(1, 3, -1)
    depth_tensor = xyz_tensor[:, 2, :]
    xy_grid_np = xyz_tensor[:, 0:2, :].detach().cpu().numpy()
    # mask only the depth tensor, all else would be redundant
    masked_depth_image_list = depth_tensor.expand(batch_size, 1, -1) * mask_list
    # TODO: Downsampling has some unforseen effect on created pointclouds
    # TODO: the index number of the list entry gets divided by the subsampling factor.
    # so when the old list has size n, the new has size n/4 and all corresponding indices will get /4
    masked_depth_image_list = masked_depth_image_list[:, :, ::6]
    color_image_list = color_image_list[::6, :]
    xy_grid_np = xy_grid_np[:, :, ::6]
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
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(coords)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        pointcloud.transform(transform)
        # Crop points not contined in the relevant workspace
        pointcloud = pointcloud.crop(workspace)
        if len(pointcloud.points) > 10:
            # print("total of ", len(pointcloud.points), " points")
            pointcloud, _ = pointcloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)
            pointclouds.append(pointcloud)

    # Visualize the point clouds (optional)

    print("length of pointclouds is ", len(pointclouds))

    if visualize:
        visualize_masked_tensor(color_image, masks_tensor)
        o3d.visualization.draw_geometries(pointclouds, width=1280, height=1280)
        # for pc in pointclouds:
        #     o3d.visualization.draw_geometries([pc])

    return pointclouds


if __name__ == "__main__":  # This is not a function but an if clause !!'
    o3d_intrinsic1 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                       fx=533.77, fy=533.53,
                                                       cx=661.87, cy=351.29)

    o3d_intrinsic2 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                       fx=534.28, fy=534.28,
                                                       cx=666.59, cy=354.94)

    T_SC1 = np.array([[-0.9521, 0.2895, -0.0985, 0.0455],
                      [-0.3032, -0.8516, 0.4276, -0.8019],
                      [0.0399, 0.4369, 0.8986, -1.3511],
                      [0.0000, 0.0000, 0.0000, 1.0000]])

    T_SC2 = np.array([[0.4548, 0.8205, -0.3464, 0.83],
                      [-0.8904, 0.4108, -0.1961, 1.77],
                      [-0.0186, 0.3976, 0.9174, -1.71],
                      [0.0000, 0.0000, 0.0000, 1.0000]])

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("using ", DEVICE)

    # Set the dimensions
    color_image1 = cv2.imread(image_path1)
    color_image2 = cv2.imread(image_path2)
    depth_image1 = cv2.imread(depth_path1, -1) / 10000
    depth_image2 = cv2.imread(depth_path2, -1) / 10000

    segmentation_parameters = SegmentationParameters(1024, conf=0.5, iou=0.9)
    segmenter = SegmentationMatcher(segmentation_parameters=segmentation_parameters,
                                    color_images=[color_image1, color_image2],
                                    depth_images=[depth_image1, depth_image2], DEVICE=DEVICE,
                                    min_dist=0.7, cutoff=3.5, ws_min=[-0.9, -1.5, -0.8], ws_max=[1.0, 1.5, 1.5])

    color_image1, depth_image1, xyz_cropped = preprocessImage(color_image1, depth_image1, T_SC1,
                                                              intrinsic=o3d_intrinsic1,
                                                              min_bounds=segmenter.min_bound,
                                                              max_bounds=segmenter.max_bound)

    color_image2, depth_image2, xyz_cropped2 = preprocessImage(color_image2, depth_image2, T_SC2,
                                                               intrinsic=o3d_intrinsic2,
                                                               min_bounds=segmenter.min_bound,
                                                               max_bounds=segmenter.max_bound)

    segmenter.intrinsics = [o3d_intrinsic1, o3d_intrinsic2]
    segmenter.transforms = [T_SC1, T_SC2]
    # activate profiler
    profiler = cProfile.Profile()
    profiler.enable()
    # start by image preprocessing
    segmenter.get_image_mask()
    # ICP align pointclouds
    segmenter.generate_global_pointclouds(visualize=False)
    print("aligning pointclouds ")
    segmenter.get_icp_transform(visualize=False)
    # segment images
    # binary_mask_tensor = create_random_blobs(visualize=False)  # returns sparse tensor
    binary_mask_tensor, binary_mask_tensor2 = segment_images([color_image1, color_image2], DEVICE, image_size=1024,
                                                             confidence=0.5,
                                                             iou=0.9)

    start = time.time()
    pointclouds1 = create_pointcloud_tensor_from_color_and_depth(color_image1, depth_image1, binary_mask_tensor,
                                                                 transform=T_SC1, workspace=segmenter.workspace,
                                                                 intrinsic=o3d_intrinsic1,
                                                                 visualize=False)

    pointclouds2 = create_pointcloud_tensor_from_color_and_depth(color_image2, depth_image2, binary_mask_tensor2,
                                                                 transform=T_SC2, workspace=segmenter.workspace,
                                                                 intrinsic=o3d_intrinsic2,
                                                                 visualize=False)

    elapsed_time = time.time() - start
    print("loop took ", elapsed_time, "seconds or ", elapsed_time / 2, " seconds per iteration")

    segmenter.pc_array_1 = pointclouds1
    segmenter.pc_array_2 = pointclouds2

    # TODO: We do not seem to have a nice overlap between pointclouds
    # segmenter.transform_pointclouds_icp(visualize=False)
    # visualize both pointclouds
    if True:
        pointclouds_total = []
        for element in pointclouds1:
            pointclouds_total.append(element)
        for element in pointclouds2:
            pointclouds_total.append(element)
        o3d.visualization.draw_geometries(pointclouds_total)

    correspondences, scores, _, _ = segmenter.match_segmentations(voxel_size=0.08, threshold=0.001)
    # Here we get the "stitched" objects matched by both cameras
    corresponding_pointclouds, matched_objects = segmenter.stitch_scene(correspondences, scores, visualize=False)
    # get all unique pointclouds
    pointclouds = segmenter.get_final_pointclouds()
    bounding_boxes = []
    for element in matched_objects:
        element, _ = element.remove_statistical_outlier(25, 0.5)
        bbox = element.get_minimal_oriented_bounding_box(robust=True)
        bbox.color = (1, 0, 0)  # open3d RED
        bounding_boxes.append(bbox)  # here bbox center is not 0 0 0

    matched_objects.extend(bounding_boxes)
    print(f"recognized and matched {len(bounding_boxes)} objects")
    o3d.visualization.draw_geometries(matched_objects)

    # Print detailed profiling information
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
