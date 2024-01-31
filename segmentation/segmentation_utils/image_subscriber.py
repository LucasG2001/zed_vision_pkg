import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class image_subscriber():

    def __init__(self):

        self.bridge = CvBridge()        
        # TODO: Add second depth subscriber after finally using both ZED cams
        rospy.Subscriber("/depth_image34783283", Image, self.depth_callback, 1)
        rospy.Subscriber("/color_image34783283", Image, self.image_callback, 1)

        # ATTENTION: depending on the zed2i.yaml file for the zed configuration parameters the images will be downsampled to lower resolutions
        rospy.Subscriber("/depth_image32689769", Image, self.depth_callback, 0)
        rospy.Subscriber("/color_image32689769", Image, self.image_callback, 0)
        self.color_images = [0, 0]
        self.depth_images = [0, 0]
     
    
    def depth_callback(self, depth_data, index):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
            self.depth_images[index] = np.array(depth_image, dtype=np.float32)
            #print("created depth image")

        except CvBridgeError as e:
            print(e)
        
        
    def image_callback(self, img_data, index):
        try:
            color_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")  # use rgb8 for open3d color palette
        except CvBridgeError as e:
            print(e)
        self.color_images[index] = np.array(color_image, dtype=np.uint8)
        #print("created color image")

    def get_images(self):
        return self.color_images, self.depth_images