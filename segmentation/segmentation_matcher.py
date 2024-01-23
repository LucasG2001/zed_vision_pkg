import cv2
from fastsam import FastSAM, FastSAMPrompt
from segmentation_matching_helpers import *
import time
import torch


class SegmentationParameters:
    def __init__(self, image_size=736, conf=0.6, iou=0.9):
        self.image_size = image_size
        self.confidence = conf
        self.iou = iou

    def set_parameters(self, imgsz, conf, iou):
        self.image_size = imgsz
        self.confidence = conf
        self.iou = iou


class SegmentationMatcher:
    """
    Note that this Matcher works with tuples or lists of images, camera parameters and so on
    """

    def __init__(self, segmentation_parameters, color_images, depth_images,
                 model_path='FastSAM-x.pt', DEVICE="gpu", depth_scale=1.0):
        # TODO: Initialize Images with something
        self.workspace_mask = None
        self.color_images = color_images
        self.depth_images = depth_images
        self.intrinsics = []
        self.transforms = []
        self.mask_arrays = []
        self.seg_params = segmentation_parameters
        self.nn_model = FastSAM(model_path)
        # check what retina_masks = false does -> creates resolution mismatch between image and masks
        self.results = self.nn_model(self.color_images[0], device=DEVICE, retina_masks=True,
                                     imgsz=self.seg_params.image_size, conf=self.seg_params.confidence,
                                     iou=self.seg_params.iou)
        self.prompt_process = FastSAMPrompt(self.color_images[0], self.results, device=DEVICE)
        self.pc_array_1 = []
        self.pc_array_2 = []
        self.final_pc_array = []
        self.device = DEVICE
        self.nn_model.to(self.device)
        self.depth_scale = depth_scale
        # full point cloud for ICP
        self.global_pointclouds = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
        self.icp = o3d.pipelines.registration.RegistrationResult()
        self.icp.transformation = np.eye(4,4)
        # Define the workspace box
        self.min_bound = np.array([-0.2, -0.9, -0.1])
        self.max_bound = np.array([1.0, 0.9, 0.9])

        # Create an axis-aligned bounding box
        self.workspace = o3d.geometry.AxisAlignedBoundingBox(self.min_bound, self.max_bound)

    def get_final_pointclouds(self):
        return self.final_pc_array

    def set_camera_params(self, intrinsics, transforms):
        self.intrinsics = intrinsics
        self.transforms = transforms

    def set_segmentation_model(self, model_path):
        self.nn_model = FastSAM(model_path)
    
    def set_segmentation_params(self, segmentation_params):
        self.seg_params = segmentation_params

    def set_images(self, color_images, depth_images):
        """
        sets depth and color images form segmentations
        Args:
            color_images: color image tuple in opencv (bgr, numpy format)
            depth_images: depth image tuple in float32 format

        Returns:
            sets self.color_images and self.depth_images to new value
        """
        # Convert color scale from bgr to rgb (opencv -> open3d)
        color_images[0] = color_images[0][:, :, ::-1]  # change color from rgb to bgr for o3d  
        color_images[1] = color_images[1][:, :, ::-1]  # change color from rgb to bgr for o3d
        self.color_images = color_images
        self.depth_images = depth_images

    
    def create_xu_yv_meshgrid(self, intrinsic, image_height=720, image_width=1280):
        # Parameters for camera projection
        cx = intrinsic.intrinsic_matrix[0, 2]
        cy = intrinsic.intrinsic_matrix[1, 2]
        fx = intrinsic.intrinsic_matrix[0, 0]
        fy = intrinsic.intrinsic_matrix[1, 1]
        # Create tensors for u and v coordinates
        u = torch.arange(0, image_width).float().cuda().unsqueeze(0)
        v = torch.arange(0, image_height).float().cuda().unsqueeze(1)
        x_u = (u - cx) / fx
        y_v = (v - cy) / fy
        # print("Size of x_u (unscaled x):", x_u.size())
        # print("Size of y_v (unscaled y):", y_v.size())

        return x_u, y_v


    def create_stacked_xyz_tensor(self, intrinsic, np_depth_image):
        depth_tensor = torch.tensor(np_depth_image.copy(), device='cuda', dtype=torch.float32).cuda()
        # Assuming you have an image of size 720x1280
        image_height, image_width = 720, 1280
        # Depth factor (adjust as needed)
        depth_factor = 1.0
        # Scale the depth tensor by the depth factor
        z_tensor = depth_tensor * depth_factor
        # print("Size of z_tensor:", z_tensor.size())
        x_u, y_v = self.create_xu_yv_meshgrid(intrinsic, image_height, image_width)
        # Broadcast and calculate the final x, y, and z coordinates
        x_coordinates_final = x_u.unsqueeze(0).expand_as(z_tensor.unsqueeze(0)) * z_tensor
        y_coordinates_final = y_v.unsqueeze(0).expand_as(z_tensor.unsqueeze(0)) * z_tensor
        # print("Size of x_coordinates_final:", x_coordinates_final.size())
        # print("Size of y_coordinates_final:", y_coordinates_final.size())
        # Stack x, y, and z coordinates along the batch dimension
        stacked_tensor = torch.cat([x_coordinates_final, y_coordinates_final, z_tensor.unsqueeze(0)], dim=0)
        # print("Size of stacked_tensor:", stacked_tensor.size())
        return stacked_tensor


    def preprocessImages(self, visualize=False):
        for i, (color_image, depth_image, transform, intrinsic) in enumerate(zip(self.color_images, self.depth_images, self.transforms, self.intrinsics)):
            xyz_tensor = self.create_stacked_xyz_tensor(intrinsic, depth_image)
            print("shape of xyz_tensor is ", xyz_tensor.shape)
            xyz_tensor_np = xyz_tensor.detach().cpu().numpy()
            xyz_homogenous = np.vstack((xyz_tensor_np, np.ones((1, xyz_tensor_np.shape[1], xyz_tensor.shape[2]))))
            # Apply homogeneous transformation
            xyz_transformed = np.matmul(transform, xyz_homogenous.reshape(4, -1)).reshape(xyz_homogenous.shape)[:3, :, :]
            # Apply bounds check and mask
            outside_bounds_mask = np.any((xyz_transformed < self.min_bound[:, np.newaxis, np.newaxis]) |
                                         (xyz_transformed > self.max_bound[:, np.newaxis, np.newaxis]), axis=0)

            # Set pixels to 0 in both the depth and color image where outside of bounds
            xyz_transformed[:, outside_bounds_mask] = 0
            depth_image[outside_bounds_mask] = 0
            color_image[outside_bounds_mask] = 0
            if visualize:
                cv2.imshow("cropped color image", color_image)
                cv2.imshow("cropped depth image", depth_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return color_image, depth_image, xyz_transformed
    
    def segment_images(self, device, image_size=1024, confidence=0.6, iou=0.95, prompt=False):
        segstart = time.time()
        print("segmenting images and masking on gpu")
        with torch.no_grad():
            results = self.nn_model(self.color_images, device=device, retina_masks=True,
                               imgsz=image_size, conf=confidence,
                               iou=iou)
            if prompt:
                prompt_process = FastSAMPrompt("./output.jpg", results, device=self.DEVICE)
                # text prompt
                ann = prompt_process.text_prompt(text='laptop')
                print("plotting rsult of text prompt")
                prompt_process.plot(annotations=ann, output_path='output.jpg', )

        print("Segmentation took, ", time.time() - segstart, " seconds")
        mask_tensor1 = results[0].masks.data
        mask_tensor2 = results[1].masks.data
        print("shape of mask tensors is ", mask_tensor1.shape, " and ", mask_tensor2.shape)

        return mask_tensor1, mask_tensor2
    
    def visualize_masked_tensor(self, color_image, binary_masks_tensor_gpu, height=720, width=1280):
        binary_masks_cpu = binary_masks_tensor_gpu.detach().cpu().numpy().astype(np.uint8)

        for i in range(binary_masks_cpu.shape[0]):
            # Plot one of the masks using OpenCV
            mask_to_plot = binary_masks_cpu[i]  # Change index to plot a different mask
            # Create a grayscale image
            image_to_plot = np.zeros((height, width), dtype=np.uint8)
            color_image[mask_to_plot == 1, 0:3] = np.random.randint(0, 256, size=3)  # Set ones to white
            # Display the image using OpenCV
        cv2.imshow('Masked image ', color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

 
    def generate_global_pointclouds(self, visualize=False):
        for i, (depth_image, color_image, intrinsic, transform) in enumerate(zip(self.depth_images, self.color_images,
                                                                                 self.intrinsics, self.transforms)):
            cx = intrinsic.intrinsic_matrix[0, 2]
            cy = intrinsic.intrinsic_matrix[1, 2]
            fx = intrinsic.intrinsic_matrix[0, 0]
            fy = intrinsic.intrinsic_matrix[1, 1]
            # Get image shape
            height, width = depth_image.shape
            # Create grid of pixel coordinates
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            # Flatten pixel coordinates
            indices = np.vstack((v.flatten(), u.flatten()))
            z_g = depth_image.flatten() / self.depth_scale
            x_g = (indices[1] - cx) * z_g / fx
            y_g = (indices[0] - cy) * z_g / fy
            # Combine x, y, z to form 3D points
            global_points_3d = np.column_stack((x_g, y_g, z_g))
            # create global pointclouds
            self.global_pointclouds[i].points = o3d.utility.Vector3dVector(global_points_3d)
            self.global_pointclouds[i].colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)
            self.global_pointclouds[i].uniform_down_sample(every_k_points=7)
            self.global_pointclouds[i].transform(transform)
            self.global_pointclouds[i] = self.global_pointclouds[i].crop(self.workspace)
            self.global_pointclouds[i], _ = self.global_pointclouds[i].remove_statistical_outlier(nb_neighbors=30,
                                                                                                  std_ratio=0.6)

        if (visualize):
            o3d.visualization.draw_geometries([self.global_pointclouds[0]])
            o3d.visualization.draw_geometries([self.global_pointclouds[1]])
            o3d.visualization.draw_geometries(self.global_pointclouds)

    def transform_pointclouds_icp(self, visualize=False):
        # only transform first pc array, as it is target of first pointcloud w.r.t to the cameras and icp registration
        for pc in self.pc_array_1:
            pc.transform(self.icp.transformation)
            if visualize:
                o3d.visualization.draw_geometries([pc])

    def match_segmentations(self, voxel_size=0.05, threshold=0.0):
        correspondences, scores, indx1, indx2 = match_segmentations_3d(self.pc_array_1, self.pc_array_2,
                                                                       voxel_size=voxel_size,
                                                                       threshold=threshold)
        # delete matched objects from single-detection list
        for i, element in enumerate(self.pc_array_1):
            if i not in indx1:
                self.final_pc_array.append(element)    
        
        for i, element in enumerate(self.pc_array_2):
            if i not in indx1:
                self.final_pc_array.append(element)

        print("length of corresponding pointclouds is ", len(correspondences))
        return correspondences, scores, indx1, indx2

    def stitch_scene(self, correspondences, scores, visualize=False):
        # ToDo: (Here or in other function) -> take correspondences and create single pointcloud
        #  out of them for ROS publishing
        corresponding_pointclouds = []
        stitched_objects = []

        for pc_tuple, iou in zip(correspondences, scores):
            # transform point cloud 1 onto point cloud 2
            corresponding_pointclouds.append(pc_tuple)
            # We use the open3d convienience operator "+" to combine two pointclouds
            stitched_objects.append((pc_tuple[0] + pc_tuple[1]))

        self.final_pc_array.extend(stitched_objects)
        if visualize:
            o3d.visualization.draw_geometries([self.global_pointclouds[0], self.global_pointclouds[1]])
            o3d.visualization.draw_geometries(stitched_objects)

        return corresponding_pointclouds, stitched_objects


    # TODO: make nice icp transform once and then leave it be
    def get_icp_transform(self, visualize=False):
        # ToDo: (Here or in other function) -> take correspondences and create single pointcloud
        #  out of them for ROS publishing
        max_dist = 0.1
        print("estimating normals")
        self.global_pointclouds[0].estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.global_pointclouds[1].estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("aligning pointclouds")
        try:
            # use default colored icp parameters
            self.icp = o3d.pipelines.registration.registration_colored_icp(self.global_pointclouds[0],
                                                                           self.global_pointclouds[1],
                                                                           max_correspondence_distance=max_dist)
            # transform point cloud 1 onto point cloud 2
            self.global_pointclouds[0].transform(self.icp.transformation)
            print("found transformation between the two pointclouds")

        except RuntimeError as e:
            # sometimes no correspondence is found. Then we simply overlay the untransformed point-clouds to avoid a
            # complete stop of the program
            print(f"Open3D Error: {e}")
            print("proceeding by overlaying point-clouds without transformation")
            print("proceeding by overlaying point-clouds without transformation")

        if visualize:
            o3d.visualization.draw_geometries([self.global_pointclouds[0], self.global_pointclouds[1]])

    def create_pointcloud_array(self, color_image, depth_image, masks_tensor, transform, intrinsic, visualize=False):
        batch_size = masks_tensor.shape[0]
        color_image_list = np.reshape(color_image, (-1, 3))
        mask_list = masks_tensor.reshape(batch_size, 1, -1)
        xyz_tensor = self.create_stacked_xyz_tensor(intrinsic, depth_image).reshape(1, 3, -1)
        depth_tensor = xyz_tensor[:, 2, :]
        xy_grid_np = xyz_tensor[:, 0:2, :].detach().cpu().numpy()
        # mask only the depth tensor, all else would be redundant
        masked_depth_image_list = depth_tensor.expand(batch_size, 1, -1) * mask_list
        masked_depth_image_list = masked_depth_image_list[:, :, ::6]
        # Need to subsample rest as well otherwise list-form indices do not correspond anymore
        color_image_list = color_image_list[::6, :]
        xy_grid_np = xy_grid_np[:, :, ::6]
        pointclouds = []
        masked_depth_image_list_np = masked_depth_image_list.detach().cpu().numpy()  # THIS IS THE BOTTLENECK!!!!
        # Next problem here: This length is static!
        for i in range(batch_size):
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
            pointcloud = pointcloud.crop(self.workspace)
            if len(pointcloud.points) > 10 and len(pointcloud.points) < 5000:
                # print("total of ", len(pointcloud.points), " points")
                pointcloud, _ = pointcloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)
                pointclouds.append(pointcloud)

        # Visualize the point clouds (optional)

        print("length of pointclouds is ", len(pointclouds))

        if visualize:
            self.visualize_masked_tensor(color_image, masks_tensor)
            o3d.visualization.draw_geometries(pointclouds, width=1280, height=1280)
            for pc in pointclouds:
                 o3d.visualization.draw_geometries([pc])

        return pointclouds