from fastsam import FastSAMPrompt
import numpy as np
import open3d as o3d


def compute_3d_iou(point_cloud_1: o3d.geometry.PointCloud, point_cloud_2: o3d.geometry.PointCloud, voxel_size=0.005):
    """
    Computes the 3D Intersection over Union (IOU) between two point cloud segmentation instances.

    Parameters:
        point_cloud_1 (np.ndarray): First point cloud segment as an Nx3 numpy array.
        point_cloud_2 (np.ndarray): Second point cloud segment as an Mx3 numpy array.

    Returns:
        float: The computed 3D IOU value.
    """
    # ToDo: (change volume intersection method to monte carlo)
    # ToDo: Find a way to save unnecessary voxelizations and iou computations

    center1 = point_cloud_1.get_center()
    center2 = point_cloud_2.get_center()
    difference = center2 - center1
    # if object pointcloud centers are further away than 0.4m then return 0 IOU
    if (np.linalg.norm(difference) > 0.4):
        return 0
    voxel_diff = np.round((difference/voxel_size), decimals=0).astype(int)
    voxel_size = voxel_size  # Adjust voxel size as needed
    # Voxelization
    voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud_1, voxel_size)
    voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud_2, voxel_size)
    voxels1 = voxel_grid1.get_voxels()
    voxels2 = voxel_grid2.get_voxels()
    a = np.asarray([voxel.grid_index for voxel in voxels1])
    b = np.asarray([voxel.grid_index for voxel in voxels2])
    b = b + voxel_diff
    # Convert voxel coordinates to strings for comparison This method works for intersection comparison, but somehow
    # both voxel grids are created in a local coordiante system
    # Compute intersection volume
    a_coords = set(map(tuple, a))
    b_coords = set(map(tuple, b))
    # Calculate the intersection of voxel coordinates
    intersection_coords = a_coords & b_coords
    # Convert the intersection coordinates back to numpy array
    intersection_voxels = np.array(list(intersection_coords))
    # Calculate the overlap volume based on the number of intersecting voxels
    intersection_volume = len(intersection_voxels)
    # Compute union volume
    volume1 = len(a)
    volume2 = len(b)
    union = volume1 + volume2 - intersection_volume
    # Calculate IOU
    iou = intersection_volume / union
    # Done: find out why IOU is greater than 0 for non overlapping objects
    # Because voxels are in local coordinate system, i.e every point-cloud has a voxel at 0 0 0
    # current solution: translate voxel_grid center  to point-cloud center does NOT WORK
    return iou


def segment_image(image_path, nn, imgsz=736, conf=0.5, iou=0.8, output_path="./output/l_ws.jpg"):
    model = nn
    print("loaded NN model")
    IMAGE_PATH = image_path  # './images/l_ws2.jpg'
    DEVICE = 'cpu'
    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=imgsz, conf=conf, iou=iou)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
    # everything prompt
    # mask_array = prompt_process.everything_prompt()  # results.mask.data
    annotations = prompt_process._format_results(result=everything_results[0], filter=0)
    annotations, _ = prompt_process.filter_masks(annotations)
    prompt_process.plot(annotations=annotations, output_path=output_path)
    mask_array = [ann["segmentation"] for ann in annotations]  # is of type np_array
    result_dict = prompt_process._format_results(everything_results[0])
    ids = [d['id'] for d in result_dict]

    return mask_array, ids


def match_segmentations2D(mask_array1, mask_array2, id1, id2):
    # Iterate over each segmentation in image1 and find corresponding in image2
    # Calculate intersection over union (IoU) for all pairs of labels
    iou_matrix = np.zeros((len(mask_array1), len(mask_array2)))
    for i, mask1 in enumerate(mask_array1):
        for j, mask2 in enumerate(mask_array2):
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            iou_matrix[i, j] = intersection / union

        # Find the best matching label pairs based on maximum IoU values
    corresponding_labels = []
    corresponding_ids = []
    while np.any(iou_matrix > 0):
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        iou_matrix[i, :] = 0  # Remove row i
        iou_matrix[:, j] = 0  # Remove column j
        corresponding_labels.append((mask_array1[i], mask_array2[j]))
        corresponding_ids.append((id1[i], id2[j]))

    return corresponding_ids


def match_segmentations_3d(pc_arr_1, pc_arr_2, voxel_size, threshold=0.05, visualize=False):
    # Iterate over each segmentation in image1 and find corresponding in image2
    # Calculate intersection over union (IoU) for all pairs of labels
    voxels = []
    if False:
        for element in pc_arr_1:
            colors = np.asarray(element.colors)
            colors[:, 0] = 0 # no blue
            element.colors = o3d.utility.Vector3dVector(colors)
            voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud(element, voxel_size)
            voxels.append(voxel_grid1)
        for element in pc_arr_2:
            colors = np.asarray(element.colors)
            colors[:, 1] = 0 #no green
            element.colors = o3d.utility.Vector3dVector(colors) 
            voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud(element, voxel_size)
            voxels.append(voxel_grid2)
        o3d.visualization.draw_geometries(voxels, window_name="voxelized and colored scene")

    corresponding_indices_1 = []
    corresponding_indices_2 = []
    iou_matrix = np.zeros((len(pc_arr_1), len(pc_arr_2)))
    for i, pc1 in enumerate(pc_arr_1):
        for j, pc2 in enumerate(pc_arr_2):
            iou_matrix[i, j] = compute_3d_iou(pc1, pc2, voxel_size)
    # threshold IOU matrix to filter noisy observations
    if (np.any(iou_matrix) < 0):
        print("IOU values under 0 detected!")
    iou_matrix[iou_matrix < threshold] = -1

    # Find the best matching label pairs based on maximum IoU values
    corresponding_pointclouds = []
    corresponding_iou = []
    while np.any(iou_matrix >= 0):
        i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        corresponding_iou.append(iou_matrix[i, j])
        iou_matrix[i, :] = -1  # Remove row i
        iou_matrix[:, j] = -1  # Remove column j
        corresponding_pointclouds.append((pc_arr_1[i], pc_arr_2[j]))
        corresponding_indices_1.append(i)
        corresponding_indices_2.append(j)
        # ToDo: Add additional inspection on 2D image, using image id's
    return corresponding_pointclouds, corresponding_iou, corresponding_indices_1, corresponding_indices_2


def project_point_clouds_to_global(pcd_arr, R, t, paint=None):
    homogeneous_matrix = np.eye(4, dtype=np.float64)
    homogeneous_matrix[0:3, 0:3] = R
    homogeneous_matrix[0:3, 3:4] = t

    # Transform point clouds to global coordinate system
    pcds = []
    for pcd in pcd_arr:
        # o3d.visualization.draw_geometries([pcd], width=1280, height=720)
        pcd.transform(homogeneous_matrix)
        if paint is not None:
            pcd.paint_uniform_color(paint)
        # o3d.visualization.draw_geometries([pcd_global], width=1280, height=720)
        pcds.append(pcd)

    return pcds, homogeneous_matrix


