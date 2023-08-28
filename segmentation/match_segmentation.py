import numpy as np
import open3d.visualization
import cv2
from tryout import visualize_segmentation
from segmentation_matching_helpers import *
import pickle
from segmentation_matcher import SegmentationMatcher, SegmentationParameters
import time
import torch
# Done: overlay both pointclouds
# Done: filter "complete" pointcloud
# Done: make colors consistent for debugging
# Done: Make depth (max dist.) consistent with image
# Done: (Change Git Repo)
# ToDo: (optimize for open3d gpu support)
# ToDo: (try mobile sam)
# ToDo: Add additional inspection on 2D image, using image id's

def homogenous_transform(R, t):
    homogeneous_matrix = np.eye(4, dtype=np.float64)
    homogeneous_matrix[0:3, 0:3] = R
    homogeneous_matrix[0:3, 3:4] = t

    return homogeneous_matrix


def crop_image_to_workspace(image, intrinsics, transform=np.eye(4, 4), min_bound=(-0.3, -0.5, -0.1),
                            max_bound=(0.9, 0.5, 1.1)):
    """
    Crops Image to only encompass the workspace of the robot,
    for faster performance

    Args:
        image: image to be cropped
        min_bound: coordinate limit minimums in xyz order
        max_bound: coordinate limit maximmums in xyz order
        transform: Transform from robot base to camera frame (T_0_C)
        intrinsics: camera intrinsics

    Returns:
        cropped image

    """
    img_size = image.shape
    R = transform[0:3, 0:3]
    t = transform[0:3, 3:4]
    T_inv = homogenous_transform(np.transpose(R), -1 * np.transpose(R) @ t)
    # Generate coordinate grids using meshgrid
    x_vals = np.linspace(min_bound[0], max_bound[0], num=2)
    y_vals = np.linspace(min_bound[1], max_bound[1], num=2)
    z_vals = np.linspace(min_bound[2], max_bound[2], num=2)
    x, y, z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    # Combine the coordinate grids to get the global coordinate vertices
    global_coords = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    homogeneous_coords = np.hstack((global_coords, np.ones((global_coords.shape[0], 1))))
    camera_coords = T_inv @ homogeneous_coords.T  # will have (x, y, z, 1) as column vectors
    c_z = camera_coords[2, :]  # z coordinates in camera frame
    unnormalized_pixels = intrinsics.intrinsic_matrix @ np.vstack((camera_coords[0:2, :],
                                                                   np.ones([1, camera_coords.shape[1]])))
    pixels = unnormalized_pixels[0:2, :] / c_z
    pixels[0, :] = np.clip(pixels[0, :], 0, img_size[1])  # u = clip 1280 (second image dimension -columns)
    pixels[1, :] = np.clip(pixels[1, :], 0, img_size[0])  # v = clip 720 (first image dimension -rows)
    # Find the minimum and maximum coordinates for cropping
    min_u = int(np.min(pixels[0, :]))
    max_u = int(np.max(pixels[0, :]))
    min_v = int(np.min(pixels[1, :]))
    max_v = int(np.max(pixels[1, :]))

    # Crop the image
    cropped_image = image[min_v:max_v, min_u:max_u]

    return cropped_image


if __name__ == "__main__":
    # ToDo: When adjusting workspace directly crop image to save segmentation runtime
    # ToDO: implement pipeline to read and save camera intrinsics, transformations etc. from file
    # "global" parameters
    model = FastSAM('FastSAM-x.pt')
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("using device ", DEVICE)

    T_0S = np.array([[-1, 0, 0, 0.41],  # Transformations from Robot base (0) to Checkerboard Frame (S)
                     [0, 1, 0, 0.0],
                     [0, 0, -1, 0.006],
                     [0, 0, 0, 1]])
    rotations = {"camera0": np.array([[0.15065033, -0.75666915, 0.63620458],  # (weiter oben)
                                      [0.98780181, 0.14086295, -0.06637176],
                                      [-0.0393962, 0.63844297, 0.76866021]]),

                 "camera1": np.array([[0.38072735, 0.73977138, -0.55478373],
                                      [-0.92468093, 0.30682222, -0.22544466],
                                      [0.00344246, 0.59883088, 0.8008681]])}

    translations = {"camera0": np.array([[-0.45760198], [0.38130433], [-0.84696597]]),
                    "camera1": np.array([[0.59649782], [0.49823864], [-0.6634929]])}

    H1 = T_0S @ homogenous_transform(rotations["camera0"], translations["camera0"])  # T_0S @ T_S_c1
    H2 = T_0S @ homogenous_transform(rotations["camera1"], translations["camera1"])  # T_0S @ T_S_c2

    # read in images
    depth_image1 = cv2.imread("./images/depth_img1.png", -1)  # read in as 1 channel
    depth_image2 = cv2.imread("./images/depth_img2.png", -1)  # read in as 1 channel
    depth_image1 = cv2.medianBlur(depth_image1, 5)
    depth_image2 = cv2.medianBlur(depth_image2, 5)
    rgb_image_path1 = "./images/color_img1.png"
    rgb_image_path2 = "./images/color_img2.png"
    color_image1 = cv2.imread(rgb_image_path1, -1)[:, :, 0:3]  # read in as 3-channel
    color_image2 = cv2.imread(rgb_image_path2, -1)[:, :, 0:3]  # -1 means cv2.UNCHANGED
    # convert color scale
    color_image1 = color_image1[:, :, ::-1]  # change color from rgb to bgr for o3d
    color_image2 = color_image2[:, :, ::-1]  # change color from rgb to bgr for o3d
    # create o3d images
    # float or int doesn't make a difference, scale later, so it's not truncated
    # image 1
    o3d_depth_1 = o3d.geometry.Image(depth_image1.astype(np.uint16))
    o3d_color_1 = o3d.geometry.Image(color_image1.astype(np.uint8))
    # image 2
    o3d_depth_2 = o3d.geometry.Image(depth_image2.astype(np.uint16))
    o3d_color_2 = o3d.geometry.Image(color_image2.astype(np.uint8))
    # Done: Check what happens when depth = 0 everywhere. Are there no points in the pc anymore?
    # Done: we SHOULD NOT scale anymore, save the depth image as uint16 scaled by 10k
    # -> The Pointcloud is empty
    # ZED camera intrinsics
    # ToDo: find out why intrinsics from ZED API are not equal to intrinsics from zed explorer
    o3d_intrinsic1 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                       fx=533.77, fy=535.53,
                                                       cx=661.87, cy=351.29)

    o3d_intrinsic2 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                       fx=523.68, fy=523.68,
                                                       cx=659.51, cy=365.34)
    print("starting")
    start_time = time.time()
    segmentation_parameters = SegmentationParameters(736, conf=0.6, iou=0.9)
    segmenter = SegmentationMatcher(segmentation_parameters, cutoff=1.5, model_path='FastSAM-x.pt', DEVICE=DEVICE)
    initialization_time = time.time() - start_time
    print("took ", initialization_time, " to initialize")
    segmenter.set_camera_params([o3d_intrinsic1, o3d_intrinsic2], [H1, H2])
    segmenter.set_images([color_image1, color_image2], [depth_image1, depth_image2])
    segmenter.preprocess_images(visualize=False)
    image_set_time = time.time() - initialization_time - start_time
    print("Loading images took ", image_set_time)
    # mask_arrays = segmenter.segment_color_images(filter_masks=False)
    mask_arrays = segmenter.segment_color_images_batch(filter_masks=False)  # batch processing of two images saves meagre 0.3 seconds
    segmentation_time = time.time() - image_set_time - initialization_time - start_time
    print("Segmentation took ", segmentation_time)
    segmenter.generate_pointclouds_from_masks()
    pointcloud_time = time.time() - segmentation_time - image_set_time - initialization_time - start_time
    print("Creating pointclouds took ", pointcloud_time)
    global_pointclouds = segmenter.project_pointclouds_to_global()
    transform_time = time.time() - pointcloud_time - segmentation_time - image_set_time - initialization_time - start_time
    print("Transforming pointclouds took ", transform_time)
    correspondences, scores = segmenter.match_segmentations(voxel_size=0.05, threshold=0.0)
    correspondence_time = time.time() - transform_time - pointcloud_time - segmentation_time - image_set_time - initialization_time - start_time
    print("Finding correspondences took ", correspondence_time)
    corresponding_pointclouds = segmenter.align_corresponding_objects(correspondences, scores, visualize=False)
    icp_time = time.time() - correspondence_time - transform_time - pointcloud_time - segmentation_time - image_set_time - initialization_time - start_time
    total_time = time.time() - start_time
    print("Aligning corresponding point-clouds took ", icp_time)
    print("Total process took ", total_time)

    # ToDo: Maybe we can use fastsam background to help our segmentation (with point prompt)

    cut_off_depth = 2.0  # [m] define how far the generated pointclouds should reach in depth
    pc_array_1 = []
    pc_array_2 = []
    # segmentation parameters
    fresh_segment = True  # if False, load masks from file, if True segment the image again
    segmentation_confidence = 0.6
    segmentation_iou = 0.9
    image_size = 736

    # create scene overview
    rgbd_image1 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color_1, o3d_depth_1,
                                                                     depth_scale=10000, depth_trunc=1.8,
                                                                     convert_rgb_to_intensity=False)
    point_cloud1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image1, o3d_intrinsic1)

    rgbd_image2 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color_2, o3d_depth_2,
                                                                     depth_scale=10000, depth_trunc=1.3,
                                                                     convert_rgb_to_intensity=False)
    point_cloud2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image2, o3d_intrinsic2)

    # Visualizations (mainly for debugging)
    # o3d.visualization.draw_geometries([point_cloud1], width=1280, height=720)
    # o3d.visualization.draw_geometries([point_cloud2], width=1280, height=720)
    # o3d.visualization.draw_geometries([point_cloud1, point_cloud2], width=1280, height=720)
    # o3d.visualization.draw_geometries([point_cloud1.transform(H1), point_cloud2.transform(H2)], width=1280, height=720)

    # Done: Check why the pointclouds are such dogshit -> Mix of scaling (because uint8) and image smoothing
    # ToDO: Check overlap in segmentations and think about how to combat it
    print("initialized parameters, starting segmentations")

    if fresh_segment:  # set this parameter at the top of the file. Load masks from file for faster debug
        mask_array_1, ids1 = segment_image(rgb_image_path1, nn=model, imgsz=image_size, conf=segmentation_confidence,
                                           iou=segmentation_iou, output_path="./output/l_ws1.jpg")

        mask_array_2, ids2 = segment_image(rgb_image_path2, nn=model, imgsz=image_size, conf=segmentation_confidence,
                                           iou=segmentation_iou, output_path="./output/l_ws2.jpg")
        # Save the masks using pickle
        with open('masks1.pkl', 'wb') as f:
            pickle.dump(mask_array_1, f)
        # Save the masks using pickle
        with open('masks2.pkl', 'wb') as f:
            pickle.dump(mask_array_2, f)
    else:
        # Load the masks from the pickle file
        with open('masks1.pkl', 'rb') as f:
            mask_array_1 = pickle.load(f)
        with open('masks2.pkl', 'rb') as f:
            mask_array_2 = pickle.load(f)

    colors_ocv1 = visualize_segmentation(mask_array_1, depth_image1, wait=10)
    for color in colors_ocv1:
        color = color[::-1]  # rgb to bgr (opencv to o3d)
        # color[1] = 0

    print("getting pointclouds 1")
    geometry_list = []
    for i, mask in enumerate(mask_array_1):
        # mask = np.asarray(mask.cpu().numpy())
        local_depth = o3d.geometry.Image(depth_image1 * mask)
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color_1, local_depth,
                                                                      depth_scale=10000, depth_trunc=cut_off_depth,
                                                                      convert_rgb_to_intensity=False)
        # fill in the extrinsic parameter for bounding box visualization
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic1)
        # pc.paint_uniform_color(np.divide(colors_ocv1[i], 255))
        # ToDo: Maybe already delete here all the pointclouds outside of the workspace (GLOBAL)
        pc = pc.uniform_down_sample(every_k_points=3)
        # pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.99)
        pc, _ = pc.remove_radius_outlier(nb_points=25, radius=0.05)
        # o3d.visualization.draw_geometries([pc], width=1280, height=720)
        if pc != 1 and len(pc.points) > 100:  # delete all pointclouds with less than 100 points
            pc_array_1.append(pc)
            bbox = pc.get_minimal_oriented_bounding_box()
            bbox.color = (1, 0, 0)
            geometry_list.append(bbox)
            geometry_list.append(pc)

    colors_ocv2 = visualize_segmentation(mask_array_2, depth_image2, wait=10)
    # Done: Handle case where pointcloud has 0 or near 0 points to avoid problems with voxelization
    # We don't use all pointclouds with below 100 points

    print("getting pointclouds 2")
    for color in colors_ocv2:
        color = color[::-1]  # rgb to bgr
        # color[2] = 0

    for i, mask in enumerate(mask_array_2):
        # mask = np.asarray(mask.cpu().numpy())
        local_depth = o3d.geometry.Image(depth_image2 * mask)
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color_2, local_depth,
                                                                      depth_scale=10000, depth_trunc=cut_off_depth,
                                                                      convert_rgb_to_intensity=False)
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic2)
        # pc.paint_uniform_color(np.divide(colors_ocv2[i], 255))

        pc = pc.uniform_down_sample(every_k_points=3)
        # pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.99)
        pc, _ = pc.remove_radius_outlier(nb_points=25, radius=0.05)
        if pc != 1 and len(pc.points) > 100:
            pc_array_2.append(pc)
            bbox = pc.get_minimal_oriented_bounding_box()
            bbox.color = (1, 0, 0)
            geometry_list.append(bbox)
            geometry_list.append(pc)

    o3d.visualization.draw_geometries(geometry_list, width=1280, height=720)
    print("Converting pointclouds to global coordinate system...")

    # convert pointclouds to global coordinate system
    global_pc1, T_S_c1 = project_point_clouds_to_global(pc_array_1, rotations["camera0"], translations["camera0"],
                                                        paint=None)
    global_pc2, T_S_c2 = project_point_clouds_to_global(pc_array_2, rotations["camera1"], translations["camera1"],
                                                        paint=None)

    print("matching segmentations now!")
    # Done: Improve Matching. For some reasons objects far apart are matched together
    # See detailed implementation. Problem was that voxel grid is always set with local reference frame
    # correspondences, scores = match_segmentations_3d(pc_array_1, pc_array_2, voxel_size=0.05, threshold=0.0)
    correspondences, scores = match_segmentations_3d(global_pc1, global_pc2, voxel_size=0.05, threshold=0.0)

    print("visualizing geometry")
    for pc_tuple, iou in zip(correspondences, scores):
        # align both pointclouds
        max_dist = 2 * np.linalg.norm(pc_tuple[0].get_center() - pc_tuple[1].get_center())
        # Estimate normals for both point clouds
        pc_tuple[0].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pc_tuple[1].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        try:
            # use default colored icp parameters
            icp = o3d.pipelines.registration.registration_colored_icp(pc_tuple[0], pc_tuple[1],
                                                                      max_correspondence_distance=max_dist)
            # transform point cloud 1 onto point cloud 2
            pc_tuple[0].transform(icp.transformation)
        except RuntimeError as e:
            # sometimes no correspondence is found. Then we simply overlay the untransformed point-clouds to avoid a
            # complete stop of the program
            print(f"Open3D Error: {e}")
            print("proceeding by overlaying point-clouds without transformation")
        print(f"Visualizing point-cloud pair with IOU of {iou}")
        o3d.visualization.draw_geometries(pc_tuple, width=1280, height=720)
