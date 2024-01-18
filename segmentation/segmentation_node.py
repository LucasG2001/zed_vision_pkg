#!/usr/bin/python3
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import numpy as np
from segmentation_matcher import SegmentationMatcher, SegmentationParameters
import open3d as o3d
import torch
import rospy
import pyzed.sl as sl
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import PlanningSceneWorld, PlanningScene
from tf2_msgs.msg import TFMessage
import time

from segmentation_utils.helper_functions import * 
from segmentation_utils.intrinsics_subscriber import intrinsic_subscriber
from segmentation_utils.image_subscriber import image_subscriber

# Get the current script's directory
script_directory = os.path.dirname(os.path.realpath(__file__))
# Construct the path to the transform.yaml file
yaml_path = os.path.join(script_directory, '..', 'scripts', 'transforms.yaml')

if __name__ == "__main__": # This is not a function but an if clause !!

    # "global" parameters
    rospy.init_node("segmentation_node")
    transforms_file = yaml_path

    # TODO: Read in intrinsics from zed node
    # published as sensor_msgs/CameraInfo
    # on topic /zed_multi/zed2i_long/zed_nodelet_front/left/camera_info or rear/left/camera_info respectively
    print("reading intrinsics")
    o3d_intrinsic1 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                       fx=533.77, fy=533.53,
                                                       cx=661.87, cy=351.29)

    o3d_intrinsic2 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                       fx=534.28, fy=534.28,
                                                       cx=666.59, cy=354.94)
    

    # TODO: scene and force_field publisher can be combined into one
    scene_publisher = rospy.Publisher("/collision_object", CollisionObject, queue_size=1) # adds object to the planning scene
    force_field_publisher = rospy.Publisher("/force_bboxes", PlanningSceneWorld, queue_size=1) # publishes planning scene to force field generation node
    transform_publisher = rospy.Publisher("/ee_transforms", TFMessage, queue_size=1)
    moveit_planning_scene_publisher = rospy.Publisher("/move_group/monitored_planning_scene", PlanningScene, queue_size=1)
    image_subscriber = image_subscriber()
    run_segmentation = False
    depth_images = []
    color_images = []
    rate = rospy.Rate(10)
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"  
    )
    print("using device ", DEVICE)

    #TODO add configuration of cameras in use for segmentation
       
    T_SC1 = load_transform_from_yaml(transforms_file, "H1")
    T_SC2 = load_transform_from_yaml(transforms_file, "H2")
    T_0S = load_transform_from_yaml(transforms_file, "T_0S")

    if T_SC1 is not None:
        print(f"Loaded H1 from YAML file:")
        print(T_SC1)
        print(f"Loaded H2 from YAML file:")
        print(T_SC2)

    H1 = T_0S @ T_SC1  # T_0S @ T_S_c1
    H2 = T_0S @ T_SC2  # T_0S @ T_S_c2


    print("starting loop")
    color_images = image_subscriber.get_images()[0]
    depth_images = image_subscriber.get_images()[1]
    segmentation_parameters = SegmentationParameters(1024, conf=0.4, iou=0.9)
    segmenter = SegmentationMatcher(segmentation_parameters, color_images, depth_images, min_dist=0.8, cutoff=1.9, model_path='FastSAM-s.pt', DEVICE=DEVICE, depth_scale=1.0)
    segmenter.set_camera_params([o3d_intrinsic1, o3d_intrinsic2], [H1, H2])
     # Set up logging
    iteration_times = []
    while not rospy.is_shutdown():
        print("looping")
        user_input = input("Enter 's' to segment the image. See the ZED-Ros node for a preview. Press 'x' to shut down. Press 'c' to clear scene ")
        if user_input.lower() == 'x':  # .lower() makes comparison case
            rospy.signal_shutdown("User requested shutdown")
        if user_input.lower() == 's':
            # segment only upon user input
            print("setting images")
            color_image1, color_image2 = image_subscriber.get_images()[0]   
            depth_image1, depth_image2 = image_subscriber.get_images()[1]
            show_input_images(color_image1, color_image2, depth_image1, depth_image2, save=True, dir=script_directory)
            segmenter.set_images([color_image1, color_image2], [depth_image1, depth_image2])
            #segmenter.generate_global_pointclouds(visualize=False)
            #segmenter.get_icp_transform(visualize=False)
            segmenter.get_image_mask()
            segmenter.preprocess_images(visualize=True) 
            # TEST!
            iteration_no = 0
            while(True):
                start = time.time()
                #mask_arrays = segmenter.segment_color_images(filter_masks=True, visualize=True)
                segmenter.preprocess_images(visualize=False) 
                segmenter.full_gpu_segment_and_mask(visualize=False)       
                # segmenter.segment_and_mask_images_gpu(visualize=False)
                print("Pointcloud generation at, ", time.time()-start, " seconds")
                segmenter.transform_pointclouds_icp(visualize=False)
                print("Alignment at, ", time.time()-start, " seconds")
                correspondences, scores, _, _ = segmenter.match_segmentations(voxel_size=0.04, threshold=0.05) 
                print("Corrrespondence match at, ", time.time()-start, " seconds")
                # Here we get the "stitched" objects matched by both cameras
                corresponding_pointclouds, matched_objects = segmenter.stitch_scene(correspondences, scores, visualize=False)
                # get all unique pointclouds
                print("final pointclouds at, ", time.time()-start, " seconds")
                pointclouds = segmenter.get_final_pointclouds()
                bounding_boxes = []
                for element in matched_objects:
                    element, _ =  element.remove_statistical_outlier(25, 0.5)
                    bbox = element.get_minimal_oriented_bounding_box(robust=True)
                    bbox.color = (1, 0, 0)  # open3d RED
                    bounding_boxes.append(bbox) # here bbox center is not 0 0 0
                
                print("Bounding boxes created at, ", time.time()-start, " seconds")

                #ToDo: publish objects to planning scene10
                collision_objects, force_field_planning_scene, transforms = create_planning_scene_object_from_bbox(bounding_boxes)
                for object in collision_objects:
                    scene_publisher.publish(object)
                    rospy.sleep(0.001)
                #print("published object to the planning scene")
                
                # transform axis aligned bboxes and corrresponding ee-transforms to the force field planner
                force_field_publisher.publish(force_field_planning_scene)
                #For debug
                """
                for object in force_field_planning_scene.collision_objects:
                    scene_publisher.publish(object)
                    rospy.sleep(0.05)
                """ 
                # End of debug
                transform_publisher.publish(transforms)
                matched_objects.extend(bounding_boxes)
                print(f"recognized and matched {len(bounding_boxes)} objects")
                #print("visualizing detected planning scene")
                # Visualize
                #o3d.visualization.draw_geometries(matched_objects)
                print("Everything took, ", time.time()-start, " seconds")

                
                # with open(current_dir + '/segmentation_log.txt', 'a') as file:
                #     file.write(f"Iteration {iteration_no} - Segmentation Time: {time.time()-start} seconds\n")
                # iteration_no += 1 

        if user_input.lower() == 'c':
            empty_scene = PlanningScene()
            empty_scene.is_diff = False
            moveit_planning_scene_publisher.publish(empty_scene)

        rate.sleep()
