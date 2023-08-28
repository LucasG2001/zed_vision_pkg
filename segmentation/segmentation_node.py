import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import numpy as np
import open3d.visualization
import cv2
from segmentation_matcher import SegmentationMatcher, SegmentationParameters
from segmentation_matching_helpers import *
import pickle
import time
import torch
import rospy
from sensor_msgs.msg import Image
import threading
import pyzed.sl as sl


def read_input():
    global user_input
    user_input = input("Enter 's' to segment the image. See the ZED-Ros node for a preview")

def img_callback(image_data):
    global color_image
    print("got a color image")
    color_image = image_data

def depth_callback(image_data):
    global depth_image
    print("got a depth image")
    depth_image = image_data

def homogenous_transform(R, t):
    homogeneous_matrix = np.eye(4, dtype=np.float64)
    homogeneous_matrix[0:3, 0:3] = R
    homogeneous_matrix[0:3, 3:4] = t

    return homogeneous_matrix

color_image = np.zeros([720,1280])
depth_image = np.zeros([720,1280])
user_input = 'f'

if __name__ == "__main__": # This is not a funciton but an if clause !!
    # "global" parameters
    rospy.init_node("segmentation_node")
    rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color/", Image, img_callback)
    rospy.Subscriber("/zed2i/zed_node/depth/depth_registered/", Image, img_callback)
    run_segmentation = False
    rate = rospy.Rate(1)
    model = FastSAM('FastSAM-x.pt')
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"  
    )
    # Start the input reading thread
    input_thread = threading.Thread(target=read_input)
    input_thread.start()

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

    # ToDo: subscribe to necessary ROS topics, such as images
    # rgb/image_rect_color
    # depth/depth_registered
    # prefix: /zed2i/zed_node/
    # placeholders
    print("settin gimages")
    depth_image1 = depth_image
    color_image1 = color_image
    depth_image2 = depth_image
    color_image2 = color_image
    # ToDo: Read in images from Ros topics (see callbacks)
    # convert color scale
    print("creating o3d images")
    # create o3d images
    o3d_depth_1 = o3d.geometry.Image(depth_image1.astype(np.uint16))
    o3d_color_1 = o3d.geometry.Image(color_image1.astype(np.uint8))
    # image 2
    o3d_depth_2 = o3d.geometry.Image(depth_image2.astype(np.uint16))
    o3d_color_2 = o3d.geometry.Image(color_image2.astype(np.uint8))
    # ToDo: Read in extrinsics from zed node
    print("reading intrinsics")
    o3d_intrinsic1 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                       fx=533.77, fy=535.53,
                                                       cx=661.87, cy=351.29)

    o3d_intrinsic2 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                       fx=523.68, fy=523.68,
                                                       cx=659.51, cy=365.34)

    print("starting loop")
    while not rospy.is_shutdown():
        print("looping")
        if segment_image == True:
            # segment only upon user input
            
            print("starting segmentation")
            segmentation_parameters = SegmentationParameters(736, conf=0.6, iou=0.9)
            segmenter = SegmentationMatcher(segmentation_parameters, cutoff=1.5, model_path='FastSAM-x.pt', DEVICE=DEVICE)
            segmenter.set_camera_params([o3d_intrinsic1, o3d_intrinsic2], [H1, H2])
            segmenter.set_images([color_image1, color_image2], [depth_image1, depth_image2])
            segmenter.preprocess_images(visualize=False)
            # mask_arrays = segmenter.segment_color_images(filter_masks=False)
            mask_arrays = segmenter.segment_color_images_batch(filter_masks=False)  # batch processing of two images saves meagre 0.3 seconds
            segmenter.generate_pointclouds_from_masks()
            global_pointclouds = segmenter.project_pointclouds_to_global()
            correspondences, scores = segmenter.match_segmentations(voxel_size=0.05, threshold=0.0)
            corresponding_pointclouds = segmenter.align_corresponding_objects(correspondences, scores, visualize=False)

            #ToDo: publish objects to planning scene
            segment_image = False

        rate.sleep()
 