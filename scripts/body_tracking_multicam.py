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
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import rospy
import math
import yaml
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


start_cam = False
camera_number = 1

def load_transform_from_yaml(yaml_file, transform_name):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    transforms = data.get('transforms', {})
    selected_transform = transforms.get(transform_name)

    if selected_transform:
        return np.array(selected_transform) 
    else:
        print(f"Error: Transform {transform_name} not found in the YAML file.")
        return None
    
def get_hand_keypoints(body):
    # initialize values
    left_hand_matrix = np.array(body.keypoint[16]).reshape([1, 3])
    right_hand_matrix = np.array(body.keypoint[17]).reshape([1, 3])

    for i in range(30, 37, 2):
        keypoint_left = np.array(body.keypoint[i]).reshape([1, 3])
        keypoint_right = np.array(body.keypoint[i + 1]).reshape([1, 3])  # loops from 50 to 70 (69 is last)
        np.vstack((left_hand_matrix, keypoint_left))  # left hand
        np.vstack((right_hand_matrix, keypoint_right))  # left hand

    left_hand_pos = np.mean(left_hand_matrix, axis=0)
    right_hand_pos = np.mean(right_hand_matrix, axis=0)

    # print("shape of hand positions is ", np.shape(left_hand_pos))
    return right_hand_pos, left_hand_pos



def camera_start_callback(msg):
    global start_cam
    global camera_number
    start_cam = msg.data
    rospy.loginfo(f"Received camera start signal for camera {camera_number}: {start_cam}")


if __name__ == "__main__":
    #  initalize node
    rospy.init_node("body_tracking_node", anonymous=True)
    yaml_file = rospy.get_param("~yaml_file")
    transform_name = rospy.get_param("~transform_name")
    serial_number = rospy.get_param("~serial_number")
    visualize = rospy.get_param("~visualize")
    camera_number = rospy.get_param("~camera_number")

    rospy.sleep(1.0)
    # Add a subscriber to listen for this when not camera 1    
    rospy.Subscriber(f'camera_start_signal_{camera_number}', Bool, camera_start_callback)
    # Create Publisher to advise the next two camera nodes when to start a camera connection
    camera_start_publisher = rospy.Publisher(f'camera_start_signal_{str(int(camera_number) + 1)}', Bool, queue_size=1)
   
    T_SC = load_transform_from_yaml(yaml_file, transform_name)
    T_0S = load_transform_from_yaml(yaml_file, "T_0S")

    if T_SC is not None:
        print(f"Loaded {transform_name} from YAML file:")
        print(T_SC)

    Transform = T_0S @ T_SC
   
    # initialze publisher for hand keypoint
    left_hand_publisher = rospy.Publisher(f'/left_hand{serial_number}', Point, queue_size=1)
    right_hand_publisher = rospy.Publisher(f'/right_hand{serial_number}', Point, queue_size=1)
    # set up publishing of images for segmentation
    # Create a publisher for the Image message
    color_image_pub = rospy.Publisher(f'/color_image{serial_number}', Image, queue_size=1)
    depth_image_pub = rospy.Publisher(f'/depth_image{serial_number}', Image, queue_size=1)
    #Opencv Bridge
    bridge = CvBridge()

    #startup flag
    if int(camera_number) == 1:
        print("will start camera 1")
        start_cam = True
    else:
        start_cam = False
        print(f"camera {camera_number} waiting for previous node to establish camera connection")
      
        
    while not start_cam:
        rospy.sleep(0.1)


    print(f"Running Body Tracking sample on camera {camera_number} ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use 720 video mode
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.set_from_serial_number(int(serial_number)) #34783283
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    
    # Open the camera
    err = zed.open(init_params)
    print(err)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    print(f"camera {camera_number} has been opened")
    rospy.sleep(0.1)
    camera_start_publisher.publish(Bool(data=True))
    rospy.sleep(2.5)
    camera_start_publisher.unregister() # stop the publisher, as it is not needed anymore

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True  # Track people across images flow
    body_param.enable_body_fitting = True  # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_param.body_format = sl.BODY_FORMAT.BODY_38  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 0.4 #confidence threshold actually doesnt work
    #TODO @ Accenture: Does your confidence threshold actually work?
   

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280),
                                       min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
        , display_resolution.height / camera_info.camera_configuration.resolution.height]


    # Create ZED objects filled in the main loop
    # TODO: Maybe we need to create multiple sl.body()
    bodies = sl.Bodies()
    image = sl.Mat()
    depth_map = sl.Mat()


    left_hand_msg = Point()
    right_hand_msg = Point()
    # init 
    right_wrist = np.array([0, 0, 0])
    left_wrist = np.array([0, 0, 0])
    detected_body_list = []
    confidences = [] 

    print(f"detection loop is running on camera {camera_number}")
    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        detected_bodies = rospy.get_param('detected_body_list')
        # Gab an image
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth
        zed.retrieve_bodies(bodies, body_runtime_param)
        # Update OCV view
        image_left_ocv = image.get_data()
        depth_image_ocv = depth_map.get_data()
        # Convert the OpenCV image to an Image message
        color_img_msg = bridge.cv2_to_imgmsg(image_left_ocv, encoding="bgra8")
         # Create a new Image message with 32FC1 encoding
        depth_img_msg = CvBridge().cv2_to_imgmsg(depth_image_ocv, encoding="32FC1")
        # Publish the Image and Depth messages
        color_image_pub.publish(color_img_msg)
        depth_image_pub.publish(depth_img_msg)

        detected_body_list.clear()
        confidences.clear()
        for i, candidate in enumerate(bodies.body_list):
            if candidate.confidence > 60:
                detected_body_list.append(candidate)
                confidences.append(candidate.confidence)

        if len(confidences) > 0: # ==> len(detected_body_list) > 0
            max_value = max(confidences)
            max_confidence_index = confidences.index(max_value)
            element = detected_body_list[max_confidence_index]
            detected_body_list = [element]
            detected_bodies[camera_number -1] = True # set a 1 if this camera has detected a body
            rospy.set_param('detected_body_list', detected_bodies)
        else:
            detected_bodies[camera_number -1] = False # set a 0 if this camera has detected a body
            rospy.set_param('detected_body_list', detected_bodies)
            left_hand_msg.x = 0.0 #static offset coming out of nowhere
            left_hand_msg.y = 0.0
            left_hand_msg.z = 0.0
            right_hand_msg.x = 0.0 #static offset coming out of nowhere
            right_hand_msg.y = 0.0
            right_hand_msg.z = 0.0
            # publish positions of the two hands
            left_hand_publisher.publish(left_hand_msg)
            right_hand_publisher.publish(right_hand_msg)
            continue # do not publish rest of messages an do not viusalize -> "stutter" stems from this
        
        # update only if not nan, else use last value
        _, left_wrist = get_hand_keypoints(detected_body_list[0])
        right_wrist, _ = get_hand_keypoints(detected_body_list[0])
        # transform hand position to base frame
        right_wrist_transformed = np.matmul(Transform, np.append(right_wrist, [1], axis=0))[:3, ]
        left_wrist_transformed = np.matmul(Transform, np.append(left_wrist, [1], axis=0))[:3, ]
        ####
        left_hand_msg.x = left_wrist_transformed[0] - 0.0 #static offset coming out of nowhere
        left_hand_msg.y = left_wrist_transformed[1] - 0.0
        left_hand_msg.z = left_wrist_transformed[2] - 0.0
        right_hand_msg.x = right_wrist_transformed[0] - 0.0 #static offst coming out of nowhere
        right_hand_msg.y = right_wrist_transformed[1] - 0.0
        right_hand_msg.z = right_wrist_transformed[2] - 0.0
        # publish positions of the two hands
        left_hand_publisher.publish(left_hand_msg)
        right_hand_publisher.publish(right_hand_msg)

        if(False):
            cv_viewer.render_2D(image_left_ocv, image_scale, detected_body_list, body_param.enable_tracking,
                                body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            #print("confidence is ", detected_body_list[0].confidence)
            # print("lenght of detecetd bodies is ", len(detected_body_list))
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()
