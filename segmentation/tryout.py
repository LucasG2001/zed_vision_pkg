from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import random
import cv2
import open3d as o3d


def get_pointcloud_from_depth_and_mask(depth_map, segmentation_mask: np.ndarray, intrinsics, color_ocv, max_val=1.0,
                                       min_val=0.2, paint=False):
    """
    takes o3d pointcloud as input and generates bounding box
    """
    # ToDo: Filter Pointcloud outliers
    # ToDo: Write function to create PC from depth and color
    # extract single segmented object out of depth map
    color_o3d = color_ocv[::-1]  # bgr to rgb
    single_object_in_depth = depth_map[:, :, 0] * segmentation_mask

    # compute pointcloud
    # Scale the values to the range 0.7 to 1.8
    scaled_array = min_val + (single_object_in_depth.astype(np.float32) / 255 * (max_val - min_val))
    scaled_array[single_object_in_depth == 0] = 0
    float_array = np.divide(scaled_array, 1).astype(np.float32)
    o3d_image = o3d.geometry.Image(float_array)
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_image, intrinsics, depth_scale=1.0,
                                                                  project_valid_depth_only=True)
    point_cloud = point_cloud.uniform_down_sample(every_k_points=15)
    # filter statistical outliers
    point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.5)
    # point_cloud, ind = point_cloud.remove_radius_outlier(nb_points=25, radius=0.05)

    if paint:
        point_cloud.paint_uniform_color(color_o3d)

    if point_cloud.has_points() > 0:

        # create_geometry(point_cloud) #  (uncomment for visual debugging)
        """
        # Visual debugging Purposes
        cv2.imshow("segmented depth Image", single_object_in_depth)
        cv2.waitKey(10)
        o3d.visualization.draw_geometries([point_cloud], width=1280, height=720)
        cv2.waitKey(2000)
        
        """
        return point_cloud

    else:
        return 1


def create_geometry(single_object_in_point_cloud: o3d.geometry.PointCloud):
    points = np.asarray(single_object_in_point_cloud.points)
    x_list, y_list, z_list = [points[:, i] for i in range(3)]
    [width, height, depth] = [np.abs(np.max(element) - np.min(element)) for element in
                              [x_list, y_list, z_list]]
    # correct if object is detected as 2-dimensional
    correction = [0 if x > 0 else 0.05 for x in [width, height, depth]]
    if points.size > 0:
        points[0, :] += correction
        single_object_in_point_cloud.points = o3d.utility.Vector3dVector(points)
        bbox = single_object_in_point_cloud.get_minimal_oriented_bounding_box()
        bbox.color = (random.random(), random.random(), random.random())
        # uncomment only for visual debugging
        # o3d.visualization.draw_geometries([bbox, single_object_in_point_cloud], width=1280, height=720)
        return bbox
    else:
        return 1


def visualize_segmentation(mask_array, depth_image, wait=10):
    depth_image = depth_image / 257  # scale for uint8 conversion
    if len(depth_image.shape) < 3:
        rgb_image = np.stack((depth_image.astype(np.uint8),) * 3, axis=-1)  # uint8 scaling for visualization
    else:
        rgb_image = depth_image.astype(np.uint8)  # scale down the depth image to uint8 to be compatible with rgb
    color_list = []
    for mask in mask_array:
        int_array = mask.astype(np.uint8)
        # int_array = mask.cpu().numpy().astype(np.uint8)
        # Find coordinates of white pixels in the binary image
        white_pixels_coords = np.where(int_array == 1)
        # Choose a random color for the white patch (excluding black and white)
        random_color = [random.randint(2, 254) for _ in range(3)]
        color_list.append(random_color)
        # Assign the random color to the white pixels in the RGB image
        rgb_image[white_pixels_coords] = random_color

    cv2.imshow("RGB Image", rgb_image)
    cv2.waitKey(wait)

    return color_list


if __name__ == "__main__":

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720, fx=946.026,
                                                          fy=946.026, cx=652.250, cy=351.917)
    model = FastSAM('FastSAM-x.pt')
    print("loaded NN model")
    IMAGE_PATH = './images/l_ws2.jpg'
    seg_image = cv2.imread(IMAGE_PATH)
    depth_image = cv2.imread("./images/depth_avg2.png", -1)[:, :, 0:3]  # read in as 3 channel
    DEVICE = 'cpu'
    everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=736, conf=0.25, iou=0.5)
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    # everything prompt
    ann = prompt_process.everything_prompt()  # results.mask.data
    print("marked ", ann.shape[0], " objects")
    print("length of annotation is ", len(ann))
    mask = ann[0]
    mask = mask.cpu().numpy()
    print("shape of mask is ", mask.shape)
    cv2.imwrite('mask.jpg', mask)

    # visualize
    colors_ocv = visualize_segmentation(ann, depth_image)
    prompt_process.plot(annotations=ann, output_path='./output/l_ws.jpg')
    # get all possible results
    result_dict = prompt_process._format_results(everything_results[0])
    print("area, = ", result_dict[0]["area"])
    for i, dictionary in enumerate(result_dict):
        print("id is ", dictionary["id"], "i = ", i)

    print("starting geometry generation")
    geometry_list = []
    for i, mask in enumerate(ann):
        mask = np.asarray(mask.cpu().numpy())
        pc = get_pointcloud_from_depth_and_mask(depth_image, segmentation_mask=mask,
                                                color_ocv=(np.divide(colors_ocv[i], 255)),
                                                intrinsics=camera_intrinsics, paint=True)

        if pc != 1:
            bounding_box = create_geometry(pc)
            geometry_list.append(pc)
            geometry_list.append(bounding_box)

    o3d.visualization.draw_geometries(geometry_list, width=1280, height=720)
