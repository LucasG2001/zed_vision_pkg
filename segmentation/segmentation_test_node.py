#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

# Path to the script's directory
script_directory = os.path.dirname(os.path.abspath(__file__))
# Path to the image file
image_path1 = os.path.join(script_directory, 'images', 'color_img1.png')
depth_path1 = os.path.join(script_directory, 'images', 'depth_img1.png')

image_path2 = os.path.join(script_directory, 'images', 'color_img2.png')
depth_path2 = os.path.join(script_directory, 'images', 'depth_img2.png')

def image_publisher():
    # Initialize the ROS node
    rospy.init_node('image_publisher', anonymous=True)

    # Set the loop rate (in Hz)
    rate = rospy.Rate(15)  # 1 Hz

    # Initialize OpenCV and CvBridge
    cv_image1 = cv2.imread(image_path1)
    cv_depth_image1 = cv2.imread(depth_path1)
    cv_image2 = cv2.imread(image_path2)
    cv_depth_image2 = cv2.imread(depth_path2)
     # Convert the depth image to grayscale (single-channel)
    gray_image1 = cv2.cvtColor(cv_depth_image1, cv2.COLOR_BGR2GRAY)
    # Convert the grayscale image to 32FC1 encoding
    float_image1 = np.float32(gray_image1) / 500
    # Convert the depth image to grayscale (single-channel)
    gray_image2 = cv2.cvtColor(cv_depth_image2, cv2.COLOR_BGR2GRAY)
    # Convert the grayscale image to 32FC1 encoding
    float_image2 = np.float32(gray_image2) / 500

   

    bridge = CvBridge()

    # Create a publisher for the Image message
    image_pub1 = rospy.Publisher('/zed_multi/zed2i_short/zed_nodelet_rear/left/image_rect_color/', Image, queue_size=1)
    image_pub2 = rospy.Publisher('/zed_multi/zed2i_long/zed_nodelet_front/left/image_rect_color/', Image, queue_size=1)

    depth_pub1 = rospy.Publisher("/zed_multi/zed2i_short/zed_nodelet_rear/depth/depth_registered/", Image, queue_size=1)
    depth_pub2 = rospy.Publisher("/zed_multi/zed2i_long/zed_nodelet_front/depth/depth_registered/", Image,  queue_size=1)

    while not rospy.is_shutdown():
        # Convert the OpenCV image to an Image message
        img_msg = bridge.cv2_to_imgmsg(cv_image1, encoding="bgr8")
         # Create a new Image message with 32FC1 encoding
        depth_img_msg = CvBridge().cv2_to_imgmsg(float_image1, encoding="32FC1")
        # Publish the Image and Depth messages
        image_pub1.publish(img_msg)
        depth_pub1.publish(depth_img_msg)

        img_msg = bridge.cv2_to_imgmsg(cv_image2, encoding="bgr8")
         # Create a new Image message with 32FC1 encoding
        depth_img_msg = CvBridge().cv2_to_imgmsg(float_image2, encoding="32FC1")

        image_pub2.publish(img_msg)
        depth_pub2.publish(depth_img_msg)

        # Log a message (optional)
        rospy.loginfo("Image published to /your_image_topic")

        # Sleep to maintain the loop rate
        rate.sleep()

if __name__ == '__main__':
    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass
