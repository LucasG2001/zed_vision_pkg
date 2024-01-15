import cv2
from segmentation_matching_helpers import *
import time
import torch
from segmentation_utils.gpu_pointcloud_generation import create_pointcloud_tensor_from_color_and_depth


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

    def __init__(self, segmentation_parameters, color_images, depth_images, min_dist=1.0, cutoff=2.0, model_path='FastSAM-x.pt', DEVICE="gpu", depth_scale=1.0):
        #TODO: Initialize Images with something
        self.color_images = color_images
        self.depth_images = depth_images
        self.intrinsics = []
        self.transforms = []
        self.mask_arrays = []
        self.seg_params = segmentation_parameters
        self.nn_model = FastSAM(model_path)
        #check what retina_masks = false does -> creates resolution mismatch between image and masks
        self.results = self.nn_model(self.color_images[0], device=DEVICE, retina_masks=True,
                                               imgsz=self.seg_params.image_size, conf=self.seg_params.confidence,
                                               iou=self.seg_params.iou)
        self.prompt_process = FastSAMPrompt(self.color_images[0], self.results, device=DEVICE)
        self.max_depth = cutoff  # truncation depth
        self.min_dist = min_dist
        self.pc_array_1 = []
        self.pc_array_2 = []
        self.final_pc_array = []
        self.device = DEVICE
        self.nn_model.to(self.device)
        self.depth_scale = depth_scale
        # full point cloud for ICP
        self.global_pointclouds = [o3d.geometry.PointCloud(), o3d.geometry.PointCloud()]
        self.icp = o3d.pipelines.registration.RegistrationResult()
        # Define the workspace box
        self.min_bound = np.array([-0.2, -0.6, -0.1])
        self.max_bound = np.array([0.8, 0.6, 0.9])
        
        # Create an axis-aligned bounding box
        self.workspace = o3d.geometry.AxisAlignedBoundingBox(self.min_bound, self.max_bound)

    def get_final_pointclouds(self):
        return self.final_pc_array

    def set_camera_params(self, intrinsics, transforms):
        self.intrinsics = intrinsics
        self.transforms = transforms

    def set_segmentation_model(self, model_path):
        self.nn_model = FastSAM(model_path)

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

    def get_image_mask(self):
        for i, (depth_image, color_image) in enumerate(zip(self.depth_images, self.color_images)):
            # color_image, depth_image = crop_images(color_image, depth_image, intrinsics=self.intrinsics[i].intrinsic_matrix, 
            #                                        extrinsics=self.transforms[i], min_bounds=self.min_bound, max_bounds=self.max_bound)
            max_mask = depth_image > self.max_depth 
            min_mask = depth_image < self.min_dist
            depth_mask = np.logical_or(max_mask, min_mask)
            self.workspace_mask = depth_mask
            
    def preprocess_images(self, visualize= False):
        for i, (depth_image, color_image) in enumerate(zip(self.depth_images, self.color_images)):
            depth_image[self.workspace_mask] = 0  # should make depth image black at these points -> non-valid pointcloud
            # Set corresponding color image pixels to 0
            color_image[np.stack([self.workspace_mask] * 3, axis=-1)] = 0
            if visualize:
                cv2.imshow('masked color image', self.color_images[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def set_segmentation_params(self, segmentation_params):
        self.seg_params = segmentation_params

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
                self.global_pointclouds[i].colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3)/ 255.0)
                self.global_pointclouds[i].uniform_down_sample(every_k_points = 7)
                self.global_pointclouds[i].transform(transform)
                self.global_pointclouds[i] = self.global_pointclouds[i].crop(self.workspace)
                self.global_pointclouds[i], _ = self.global_pointclouds[i].remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)

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
        correspondences, scores, indx1, indx2 = match_segmentations_3d(self.pc_array_1, self.pc_array_2, voxel_size=voxel_size,
                                                         threshold=threshold)
        # delete matched objects from single-detection list
        for index in sorted(indx1, reverse=True):
            del self.pc_array_1[index]

        for index in sorted(indx2, reverse=True):
            del self.pc_array_2[index]

        self.final_pc_array = self.pc_array_1
        self.final_pc_array.extend(self.pc_array_2)
        
        print("length of corresponding pointclouds is ", len(correspondences))
        return correspondences, scores, indx1, indx2

    #TODO: align first correspond later
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
    
    def project_pointclouds_to_global(self, visualize=True):
        for pc_array, transform in zip([self.pc_array_1, self.pc_array_2], self.transforms):
            for pc in pc_array:
                pc.transform(transform)
                if visualize:
                        o3d.visualization.draw_geometries([pc])
        return self.pc_array_1, self.pc_array_2


    # TODO: make nice icp transform once and then leave it be
    def get_icp_transform(self, visualize=False):
        # ToDo: (Here or in other function) -> take correspondences and create single pointcloud
        #  out of them for ROS publishing
        max_dist = 0.1
        print("estimating normals")
        self.global_pointclouds[0].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.global_pointclouds[1].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("aligning pointclouds")
        try:
            # use default colored icp parameters
            self.icp = o3d.pipelines.registration.registration_colored_icp(self.global_pointclouds[0], self.global_pointclouds[1],
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


    def segment_and_mask_images_gpu(self, visualize=True):
        postprocessing = 0
        pc_creating = 0
        rgbd_creating = 0
        self.pc_array_1.clear()
        self.pc_array_2.clear()
        point_cloud_arrays = [self.pc_array_1, self.pc_array_2]
        start_time = time.time()
        print("segmenting images and masking on gpu")
        with torch.no_grad():
            self.results = self.nn_model(self.color_images, device=self.device, retina_masks=True,
                                               imgsz=self.seg_params.image_size, conf=self.seg_params.confidence,
                                               iou=self.seg_params.iou)
            
        print("Segmentation took, ", time.time()-start_time, " seconds") 
        mask_tensor1 = self.results[0].masks.data
        mask_tensor2 = self.results[1].masks.data
        mask_tensor1.to_sparse()
        mask_tensor2.to_sparse()
        # now we filter masks that are too big
        # Calculate the sum along the 1280x720 dimensions for each mask
        sum_mask1 = torch.sum(mask_tensor1, dim=(1, 2))
        sum_mask2 = torch.sum(mask_tensor2, dim=(1, 2))
        # Find indices of masks with sum >= 15000
        indices_to_keep_mask1 = torch.nonzero(sum_mask1 < 15000).squeeze()
        indices_to_keep_mask2 = torch.nonzero(sum_mask2 < 15000).squeeze()
        # Filter masks based on indices
        mask_tensor1 = mask_tensor1[indices_to_keep_mask1]
        mask_tensor2 = mask_tensor2[indices_to_keep_mask2]
        # Empty the GPU cache
        # We have a Problem when running body tracking on three cameras the segmentation and postprocessing with the model all on gpu"
        del sum_mask1, sum_mask2, indices_to_keep_mask1, indices_to_keep_mask2
        torch.cuda.empty_cache()
        print(f"shape of filtered mask tensors is {mask_tensor1.size()} and {mask_tensor2.size()} ")
        # Convert NumPy arrays to PyTorch tensors and move to GPU
        depth_tensor1 = torch.tensor(self.depth_images[0].copy(), device='cuda', dtype=torch.float32)
        depth_tensor2 = torch.tensor(self.depth_images[1].copy(), device='cuda', dtype=torch.float32)
        #print("creating masked depth tensors")
        # Assuming mask_tensor is of size (n, 720, 180) and color_tensor1 is of size (1, 720, 1280, 3), x and y are inversed in opencv!
        # Apply masks to depth image
        masked_depth1 = depth_tensor1 * mask_tensor1
        masked_depth2 = depth_tensor2 * mask_tensor2
        # start transform to 3D
        masks1 = masked_depth1.cpu().numpy()
        masks2 = masked_depth2.cpu().numpy()
        # Release depth tensors from memory
        del mask_tensor1, mask_tensor2, masked_depth1, masked_depth2
        # Empty the GPU cache
        torch.cuda.empty_cache()
       

        # Global Pointcloud once should be enough
        # u,v, depth -> points, then just select at indices
        for i, (masked_depth_tensor, color_image, intrinsic, transform) in enumerate(
                zip([masks1, masks2], self.color_images,
                    self.intrinsics, self.transforms)):
            cx = intrinsic.intrinsic_matrix[0, 2]
            cy = intrinsic.intrinsic_matrix[1, 2]
            fx = intrinsic.intrinsic_matrix[0, 0]
            fy = intrinsic.intrinsic_matrix[1, 1]
            for j in range (masked_depth_tensor.shape[0]):
                start_time = time.time()
                # filter relevant points in image
                depth_image = masked_depth_tensor[j]
                indices = np.where((depth_image) > 0) # gives [row indices (y, v), columns indices (x, u)]
                # local_color =  color_image[indices]
                #intrinsics transformation
                z = depth_image[indices] / self.depth_scale
                x = (indices[1] - cx) * z / fx
                y = (indices[0] - cy) * z / fy
                # Combine x, y, z to form 3D points
                points_3d = np.column_stack((x, y, z))
                # point creation time
                rgbd_time = time.time() - start_time
                rgbd_creating += rgbd_time
                #Pointcloud 
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(points_3d)
                # pc.colors = o3d.utility.Vector3dVector(local_color / 255.0)  # Normalize colors to the range [0, 1]
                pc_creation_time = time.time() - rgbd_time - start_time
                pc_creating += pc_creation_time
                #Downsample
                pc = pc.uniform_down_sample(every_k_points=3)
                # NEW
                pc.transform(self.transforms[i])
                pc = pc.crop(self.workspace)
                
                if len(pc.points) > 100:  # delete all pointclouds with less than 100 points
                    pc, _ = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)
                    point_cloud_arrays[i].append(pc)
                    if (visualize):
                        o3d.visualization.draw_geometries([pc], width=1280, height=720)
                pc_postprocess_time = time.time() - pc_creation_time - rgbd_time - start_time
                postprocessing += pc_postprocess_time

            print("size of pc array is ", len(self.pc_array_1) + len(self.pc_array_2))

        print("time for creating rgbd image is ", rgbd_creating)
        print("creating pointcloud took ", pc_creating)
        print("cleaning pointcloud took ", postprocessing)

        if visualize:    
                self.prompt_process = FastSAMPrompt(self.color_images[0], self.results[0], device=self.device)
                annotations1 = self.prompt_process._format_results(result=self.results[0], filter=0)
                self.prompt_process.plot(annotations=annotations1, output_path=f'segmentation1.jpg')
                self.prompt_process = FastSAMPrompt(self.color_images[1], self.results[1], device=self.device)
                annotations2 = self.prompt_process._format_results(result=self.results[1], filter=0)
                self.prompt_process.plot(annotations=annotations2, output_path=f'segmentation2.jpg')
                o3d.visualization.draw_geometries(point_cloud_arrays[0], width=1280, height=720)
                o3d.visualization.draw_geometries(point_cloud_arrays[1], width=1280, height=720)

        
    def full_gpu_segment_and_mask(self, visualize=True):
        self.pc_array_1.clear()
        self.pc_array_2.clear()
        point_cloud_arrays = [self.pc_array_1, self.pc_array_2]
        start_time = time.time()
        print("segmenting images and masking on gpu")
        with torch.no_grad():
            self.results = self.nn_model(self.color_images, device=self.device, retina_masks=True,
                                               imgsz=self.seg_params.image_size, conf=self.seg_params.confidence,
                                               iou=self.seg_params.iou)
            
        print("Segmentation took, ", time.time()-start_time, " seconds") 
        mask_tensor1 = self.results[0].masks.data
        mask_tensor2 = self.results[1].masks.data
        mask_tensor1.to_sparse()
        mask_tensor2.to_sparse()
        # now we filter masks that are too big
        # Calculate the sum along the 1280x720 dimensions for each mask
        sum_mask1 = torch.sum(mask_tensor1, dim=(1, 2))
        sum_mask2 = torch.sum(mask_tensor2, dim=(1, 2))
        # Find indices of masks with sum >= 15000
        indices_to_keep_mask1 = torch.nonzero(sum_mask1 < 15000).squeeze()
        indices_to_keep_mask2 = torch.nonzero(sum_mask2 < 15000).squeeze()
        # Filter masks based on indices
        mask_tensor1 = mask_tensor1[indices_to_keep_mask1]
        mask_tensor2 = mask_tensor2[indices_to_keep_mask2]
        # Empty the GPU cache
        # We have a Problem when running body tracking on three cameras the segmentation and postprocessing with the model all on gpu"
        del sum_mask1, sum_mask2, indices_to_keep_mask1, indices_to_keep_mask2
        torch.cuda.empty_cache()
        print(f"shape of filtered mask tensors is {mask_tensor1.size()} and {mask_tensor2.size()} ")
        # Convert NumPy arrays to PyTorch tensors and move to GPU
        pc_start_time = time.time()
        self.pc_array_1 = create_pointcloud_tensor_from_color_and_depth(color_image= self.color_images[0], depth_image=self.depth_images[0], masks_tensor=mask_tensor1, visualize=False)
        self.pc_array_2 = create_pointcloud_tensor_from_color_and_depth(color_image= self.color_images[1], depth_image=self.depth_images[1], masks_tensor=mask_tensor2, visualize=False)
        pc_end_time = time.time()
        print("creating pointcloud took ", pc_end_time - pc_start_time)

        if visualize:    
                self.prompt_process = FastSAMPrompt(self.color_images[0], self.results[0], device=self.device)
                annotations1 = self.prompt_process._format_results(result=self.results[0], filter=0)
                self.prompt_process.plot(annotations=annotations1, output_path=f'segmentation1.jpg')
                self.prompt_process = FastSAMPrompt(self.color_images[1], self.results[1], device=self.device)
                annotations2 = self.prompt_process._format_results(result=self.results[1], filter=0)
                self.prompt_process.plot(annotations=annotations2, output_path=f'segmentation2.jpg')
                o3d.visualization.draw_geometries(point_cloud_arrays[0], width=1280, height=720)
                o3d.visualization.draw_geometries(point_cloud_arrays[1], width=1280, height=720)
