import open3d as o3d
import numpy as np


def get_pointcloud_from_depth_image(depth_image, min_val, max_val):
    scaled_array = min_val + (depth_image.astype(np.float32) / 255 * (max_val - min_val))
    scaled_array[depth_image == 0] = np.nan
    float_array = np.divide(scaled_array, 1).astype(np.float32)
    o3d_image = o3d.geometry.Image(float_array)
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720, fx=946.026, fy=946.026, cx=652.250,
                                                      cy=351.917)
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_image, o3d_intrinsic, depth_scale=1.0,
                                                                  project_valid_depth_only=True)
    # Set the voxel size based on your needs
    voxel_size = 0.03  # Adjust this value
    point_cloud = point_cloud.uniform_down_sample(every_k_points=15)

    # Apply voxel grid downsampling
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    return downsampled_pcd


def get_pointcloud_from_rgbd_image(color_image, depth_image, intrinsic, min_val, max_val, mask=1, cutoff=1.0):
    depth_image = depth_image[:, :, 0] * mask
    depth_array = min_val + (depth_image.astype(np.float32) / 255 * (max_val - min_val))
    depth_array[depth_image == 0] = np.nan
    float_array = depth_array.astype(np.float32)
    o3d_depth_image = o3d.geometry.Image(float_array)
    o3d_color_image = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color_image, o3d_depth_image, depth_scale=1.0,
                                                                    depth_trunc=cutoff, convert_rgb_to_intensity=False)
    # Done: Check what happens when depth = 0 everywhere. Are there no points in the pc anymore?
    # -> The Pointcloud is empty
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, project_valid_depth_only=True)
    # visualize for debugging
    # o3d.visualization.draw_geometries([point_cloud], width=1280, height=720)
    return point_cloud


def align_pointclouds(pcd_source, pcd_target, rotations, translations):
    """
    T_S_c1 and 2 are transformation matrices between a common
    global coordinate frame S and the camera frame 1 and 2 respectively
    """
    T_S_c1 = np.eye(4, dtype=np.float64)
    T_S_c1[0:3, 0:3] = rotations["camera0"]
    T_S_c1[0:3, 3:4] = translations["camera0"]

    T_S_c2 = np.eye(4, dtype=np.float64)
    T_S_c2[0:3, 0:3] = rotations["camera1"]
    T_S_c2[0:3, 3:4] = translations["camera1"]

    T_c2_S = np.eye(4, dtype=np.float64)
    T_c2_S[0:3, 0:3] = np.transpose(rotations["camera1"])
    T_c2_S[0:3, 3:4] = -1 * np.transpose(rotations["camera1"]) @ translations["camera1"]

    # alignment parameters
    initial_transform = T_c2_S @ T_S_c1
    threshold = 0.02

    # Estimate normals for both point clouds
    pcd_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Perform initial alignment using ICP
    icp_coarse = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, max_correspondence_distance=threshold, init=initial_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Transform the source point cloud
    pcd_source.transform(icp_coarse.transformation)
    return pcd_source, pcd_target
