#!/usr/bin/python3
########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


from geometry_msgs.msg import Point
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import rospy
import math
from visualization_msgs.msg import Marker, MarkerArray
from marker_viz import create_markers


def get_hand_keypoints(body):
    # initialize values
    left_hand_matrix = np.array(body.keypoint[16]).reshape([1, 3])
    right_hand_matrix = np.array(body.keypoint[17]).reshape([1, 3])
    for i in range(30, 50):
        keypoint_left = np.array(body.keypoint[i]).reshape([1, 3])
        keypoint_right = np.array(body.keypoint[i + 20]).reshape([1, 3])  # loops from 50 to 70 (69 is last)
        np.vstack((left_hand_matrix, keypoint_left))  # left hand
        np.vstack((right_hand_matrix, keypoint_right))  # left hand

    left_hand_pos = np.mean(left_hand_matrix, axis=0)
    right_hand_pos = np.mean(right_hand_matrix, axis=0)

    print("shape of hand positions is ", np.shape(left_hand_pos))
    return right_hand_pos, left_hand_pos


if __name__ == "__main__":

    T_0S = np.array([[-1, 0, 0, 0.358],
                     [0, 1, 0, 0.03],
                     [0, 0, -1, 0.006],
                     [0, 0, 0, 1]])
    """
      
    T_Y_up_w = np.array([[0, 0, 0, 0.0], # transform from Y up right handed cs to world cs
                     [0, 0, 0, -1.5922],
                     [0, 0, 0, 0.0],
                     [0, 0, 0, 1.0]])
    
    T_Y_up_w = np.eye(4,4)
     
    T_Y_dwn_Y_up = np.array([[1, 0, 0, 0.0], # transform  from default cs in zed to right handed y up
                         [0, -1, 0, 0.0],
                         [0, 0, -1, 0.0],
                         [0, 0, 0, 1.0]])
    
    T_Y_dwn_Y_up = np.eye(4,4)
    
    """

    T_SC = np.array([[0.59654493, 0.67077667, -0.44067313, 0.65222527],
                     [-0.80140306, 0.46812619, -0.37230498, 0.74315646],
                     [-0.04344286, 0.57525345, 0.81682078, -0.9906735],
                     [0., 0., 0., 1.]])

    Transform = T_0S @ T_SC

    print(Transform.dot(np.transpose([[0, 0, 0, 1]])))

    #  initalize node
    rospy.init_node("body_tracking_node")
    # initialze publisher for hand keypoint
    left_hand_publisher = rospy.Publisher('cartesian_impedance_controller/left_hand', Point, queue_size=10)
    right_hand_publisher = rospy.Publisher('cartesian_impedance_controller/right_hand', Point, queue_size=10)
    # Create a publisher for the "keypoint_marker" topic
    marker_publisher = rospy.Publisher('keypoint_marker', MarkerArray, queue_size=10)
    # Create a MarkerArray message object
    marker_array_msg = MarkerArray()

    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use 720 video mode
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.set_from_serial_number(32689769)

    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True  # Track people across images flow
    body_param.enable_body_fitting = True  # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_param.body_format = sl.BODY_FORMAT.BODY_70  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 35

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                       min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
        , display_resolution.height / camera_info.camera_configuration.resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,
                body_param.body_format)

    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    image = sl.Mat()

    left_hand_msg = Point()
    right_hand_msg = Point()
    # camera_pose = sl.Pose()
    # py_translation = sl.Translation()

    # init 
    right_wrist = np.array([0, 0, 0])
    left_wrist = np.array([0, 0, 0])

    while viewer.is_available():

        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # get camera position in world coordinate frame
            # rotation = camera_pose.get_rotation_vector()
            # translation = camera_pose.get_translation(py_translation.get())
            # print("camera is at ", translation, "\n")
            # print("and is rotated by ", rotation, "\n")
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            for element in bodies.body_list:
                # print("{} {}".format(element.id, element.position))
                if (len(bodies.body_list) > 1):
                    print("WARNING: more than one body detected!")
                print("----------------------")

                # update only if not nan, else use last value
                if (not math.isnan(element.keypoint[16][0])):
                    _, left_wrist = get_hand_keypoints(element)
                if (not math.isnan(element.keypoint[17][0])):
                    right_wrist, _ = get_hand_keypoints(element)

                # transform hand position to base frame
                right_wrist_transformed = np.matmul(Transform, np.append(right_wrist, [1], axis=0))[:3, ]
                left_wrist_transformed = np.matmul(Transform, np.append(left_wrist, [1], axis=0))[:3, ]
                print("left hand is at {}".format(left_wrist_transformed))
                print("right hand is at {}".format(right_wrist_transformed))
                left_hand_msg.x = left_wrist_transformed[0]
                left_hand_msg.y = left_wrist_transformed[1]
                left_hand_msg.z = left_wrist_transformed[2]
                right_hand_msg.x = right_wrist_transformed[0]
                right_hand_msg.y = right_wrist_transformed[1]
                right_hand_msg.z = right_wrist_transformed[2]

                # publish positions of the two hands
                left_hand_publisher.publish(left_hand_msg)
                right_hand_publisher.publish(right_hand_msg)
                marker_array_msg = create_markers(left_hand_msg, right_hand_msg)
                marker_publisher.publish(marker_array_msg)

            # Update GL view
            viewer.update_view(image, bodies)
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv, image_scale, bodies.body_list, body_param.enable_tracking,
                                body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            cv2.waitKey(10)
            rospy.sleep(0.01)

    viewer.exit()
    image.free(sl.MEM.CPU)
    zed.close()
