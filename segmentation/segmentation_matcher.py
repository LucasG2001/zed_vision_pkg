import cv2
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

    def __init__(self, segmentation_parameters, cutoff, model_path='FastSAM-x.pt', DEVICE="cpu", depth_scale=1.0):
        self.color_images = []
        self.depth_images = []
        self.intrinsics = []
        self.transforms = []
        self.mask_arrays = []
        self.seg_params = segmentation_parameters
        self.nn_model = FastSAM(model_path)
        self.max_depth = cutoff  # truncation depth
        self.pc_array_1 = []
        self.pc_array_2 = []
        self.final_pc_array = []
        self.device = DEVICE
        self.nn_model.to(self.device)
        self.depth_scale = depth_scale

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
        stacked_images = np.vstack([self.color_images[0], self.color_images[1]])
        # for i in range(50):
            # self.color_images.append(self.color_images[0])

        batch_img = torch.from_numpy(stacked_images).float()
        self.batch_img = batch_img.to(self.device)

    def preprocess_images(self, visualize=False):
        for i, (depth_image, color_image) in enumerate(zip(self.depth_images, self.color_images)):
            depth_mask = depth_image > self.max_depth
            depth_image[depth_mask] = 0  # should make depth image black at these points -> non-valid pointcloud
            # Set corresponding color image pixels to 0
            color_image[np.stack([depth_mask] * 3, axis=-1)] = 0
            if visualize:
                cv2.imshow('color_img', self.color_images[i])
                cv2.waitKey(0)

    def set_segmentation_params(self, segmentation_params):
        self.seg_params = segmentation_params

    def segment_color_images(self, filter_masks=True, visualize=False):
        # ToDo: test if one can scam runtime of the model by combining the them at the same time
        # ToDo: Yes we can do exactly that
        DEVICE = self.device
        print("loaded NN model")
        color_images = self.color_images
        for i, image in enumerate(color_images):
            everything_results = self.nn_model(image, device=DEVICE, retina_masks=True,
                                               imgsz=self.seg_params.image_size, conf=self.seg_params.confidence,
                                               iou=self.seg_params.iou)
            prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)
            annotations = prompt_process._format_results(result=everything_results[0], filter=0)
            if filter_masks:
                annotations, _ = prompt_process.filter_masks(annotations)
            mask_array = [ann["segmentation"] for ann in annotations]  # is of type np_array
            if visualize:
                prompt_process.plot(annotations=prompt_process.everything_prompt(), output_path=f'segmentation{i}.jpg')

            self.mask_arrays.append(mask_array)

        return self.mask_arrays

    def segment_color_images_batch(self, filter_masks=True, visualize=False):
        DEVICE = self.device
        print("loaded NN model")
        color_images = self.color_images
        start_time = time.time()
        everything_results = self.nn_model(self.color_images, device=DEVICE, retina_masks=True,
                                           imgsz=self.seg_params.image_size, conf=self.seg_params.confidence,
                                           iou=self.seg_params.iou)
        # takes 95% of runtime up to here
        print("NN took ", time.time() - start_time, " s")
        prompt_process = FastSAMPrompt(self.color_images, everything_results, device=DEVICE)
        annotations1 = prompt_process._format_results(result=everything_results[0], filter=0)
        annotations2 = prompt_process._format_results(result=everything_results[1], filter=0)
        # takes 99% of runtime up to here (without filtering)
        if filter_masks:
            annotations1, _ = prompt_process.filter_masks(annotations1)
            annotations2, _ = prompt_process.filter_masks(annotations2)
        mask_array1 = [ann["segmentation"] for ann in annotations1]  # is of type np_array
        mask_array2 = [ann["segmentation"] for ann in annotations2]  # is of type np_array
        if visualize:
                prompt_process.plot(annotations=annotations1, output_path=f'segmentation1.jpg')
                prompt_process.plot(annotations=annotations2, output_path=f'segmentation2.jpg')

        self.mask_arrays = [mask_array1, mask_array2]

        return self.mask_arrays

    def generate_pointclouds_from_masks(self, visualize=False):
        postprocessing = 0
        pc_creating = 0
        rgbd_creating = 0
        point_cloud_arrays = [self.pc_array_1, self.pc_array_2]
        # ToDo: Find a way to generate way less pointclouds
        #  maybe by blacking the image where depth is no more interesting
        for i, (mask_array, depth_image, color_image, intrinsic) in enumerate(
                zip(self.mask_arrays, self.depth_images, self.color_images,
                    self.intrinsics)):
            for mask in mask_array:
                start_time = time.time()
                local_depth = o3d.geometry.Image(depth_image * mask)
                local_color = o3d.geometry.Image(color_image.astype(np.uint8))
                # Done: Check if creating pointcloud from depth image only is faster and feasible
                # It speeds up the function call by ca. 1s but makes it impossible to use colored ICP
                rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(local_color, local_depth,
                                                                              depth_scale=self.depth_scale,
                                                                              depth_trunc=self.max_depth,
                                                                              convert_rgb_to_intensity=False)
                if visualize:
                    o3d.visualization.draw_geometries([rgbd_img])

                pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=intrinsic)
                rgbd_time = time.time() - start_time
                rgbd_creating += rgbd_time
                # pc = open3d.cpu.pybind.geometry.PointCloud.create_from_depth_image(depth=local_depth,
                # intrinsic=intrinsic,
                # depth_scale=self.depth_scale,
                # depth_trunc=self.max_depth)

                pc_creation_time = time.time() - rgbd_time - start_time
                pc_creating += pc_creation_time
                pc = pc.uniform_down_sample(every_k_points=3)
                pc, _ = pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.6)
                # TODO: PC filtering is very costly and maks up 60% of the function call.
                #  However, the quality of the matches is shit without it
                # Statistical outlier rejection is however, way faster than radius outlier rejection
                # pc, _ = pc.remove_radius_outlier(nb_points=30, radius=0.05)
                pc_postprocess_time = time.time() - pc_creation_time - rgbd_time - start_time
                postprocessing += pc_postprocess_time

                # o3d.visualization.draw_geometries([pc], width=1280, height=720)
                if len(pc.points) > 100:  # delete all pointclouds with less than 40 points
                    point_cloud_arrays[i].append(pc)

        print("time for creating rgbd image is ", rgbd_creating)
        print("creating pointcloud took ", pc_creating)
        print("cleaning pointcloud took ", postprocessing)

    def project_pointclouds_to_global(self, visualize=True):
        for pc_array, transform in zip([self.pc_array_1, self.pc_array_2], self.transforms):
            for pc in pc_array:
                pc.transform(transform)
                if visualize:
                        o3d.visualization.draw_geometries([pc])
        return self.pc_array_1, self.pc_array_2

    def match_segmentations(self, voxel_size=0.05, threshold=0.0):
        correspondences, scores, indx1, indx2 = match_segmentations_3d(self.pc_array_1, self.pc_array_2, voxel_size=voxel_size,
                                                         threshold=threshold)
        for index in sorted(indx1, reverse=True):
            del self.pc_array_1[index]

        for index in sorted(indx2, reverse=True):
            del self.pc_array_2[index]

        self.final_pc_array = self.pc_array_1
        self.final_pc_array.extend(self.pc_array_2)
        

        return correspondences, scores, indx1, indx2

    def align_corresponding_objects(self, correspondences, scores, visualize=False, use_icp=True):
        # ToDo: (Here or in other function) -> take correspondences and create single pointcloud
        #  out of them for ROS publishing
        corresponding_pointclouds = []
        stitched_objects = []
        for pc_tuple, iou in zip(correspondences, scores):
            if use_icp:
                # align both pointclouds
                max_dist = 1 * np.linalg.norm(pc_tuple[0].get_center() - pc_tuple[1].get_center())
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
                    print("proceeding by overlaying point-clouds without transformation")

            corresponding_pointclouds.append(pc_tuple)
            # We use the open3d convienience operator "+" to combine two pointclouds
            stitched_objects.append((pc_tuple[0] + pc_tuple[1]).uniform_down_sample(every_k_points = 2))
        self.final_pc_array.extend(stitched_objects)
        # OPTIONAL TODO: reconstrct the whole mesh from segmented pointcloud
        if visualize:
            o3d.visualization.draw_geometries(stitched_objects)

        return corresponding_pointclouds, stitched_objects
