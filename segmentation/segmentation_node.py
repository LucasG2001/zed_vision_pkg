import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import numpy as np
import cv2
from segmentation_matcher import SegmentationMatcher, SegmentationParameters
from segmentation_matching_helpers import FastSAM
import open3d as o3d
import torch
import rospy
from sensor_msgs.msg import Image, CameraInfo
import pyzed.sl as sl
from cv_bridge import CvBridge, CvBridgeError
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation


def create_planning_scene_object_from_bbox(bbox, id = "1"):
    vertices = bbox.get_box_points() # o3d vector
    R = np.array(bbox.R) # Rotaiton Matrix of bounding box
    center = bbox.center
    sizes = bbox.extent
    quat_R = (Rotation.from_matrix(R)).as_quat()
    # we need to publish a pose and a size, to spawn a rectangle of size S at pose P in the moveit planning scene
    collision_object = CollisionObject()
    collision_object.header.frame_id = "panda_link0"

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = center 
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat_R

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = sizes
    # collision_object.pose.append(pose) # this is the pose to which everything else is defined relative?
    # TODO: get id's to be consotent throughout segmentations nad the planning scene
    collision_object.id = id
    collision_object.primitives.append(primitive)
    collision_object.primitive_poses.append(pose)
    collision_object.operation = CollisionObject.ADD

    return collision_object



class intrinsic_subscriber():
    def __init__(self):
        rospy.Subscriber("/zed_multi/zed2i_long/zed_nodelet_front/left/camera_info", CameraInfo, self.intrisics_callback, 0)
        rospy.Subscriber("/zed_multi/zed2i_long/zed_nodelet_rear/left/camera_info", CameraInfo, self.intrisics_callback, 1)
        o3d_intrinsic1 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                        fx=533.77, fy=535.53,
                                                       cx=661.87, cy=351.29)

        o3d_intrinsic2 = o3d.camera.PinholeCameraIntrinsic(width=1280, height=720,
                                                            fx=523.68, fy=523.68,
                                                            cx=659.51, cy=365.34)
        
        self.o3d_intrinsics = [o3d_intrinsic1, o3d_intrinsic2]

    
    def intrinsics_calback(self, data, index):
        intrinsics_array = data.K
        fx = intrinsics_array[0]
        cx = intrinsics_array[2]
        fy = intrinsics_array[4]
        cy = intrinsics_array[5]
        height = data.height
        width = data.width
        self.o3d_intrinsics[index].set_intrisic(width, height, fx, fy, cx, cy)
        
    def get_intrinsics(self):
        return self.o3d_intrinsics

class image_subscriber():

    def __init__(self):

        self.bridge = CvBridge()        
        # TODO: Add second depth subscriber after finally using both ZED cams
        rospy.Subscriber("/zed_multi/zed2i_long/zed_nodelet_front/depth/depth_registered/", Image, self.depth_callback, 0)
        rospy.Subscriber("/zed_multi/zed2i_long/zed_nodelet_front/left/image_rect_color/", Image, self.image_callback, 0)

        # ATTENTION: depending on the zed2i.yaml file for the zed configuration parameters the images will be downsampled to lower resolutions
        rospy.Subscriber("/zed_multi/zed2i_short/zed_nodelet_rear/depth/depth_registered/", Image, self.depth_callback, 1)
        rospy.Subscriber("/zed_multi/zed2i_short/zed_nodelet_rear/left/image_rect_color/", Image, self.image_callback, 1)
        self.color_images = [0, 0]
        self.depth_images = [0, 0]
     
    
    def depth_callback(self, depth_data, index):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
        except CvBridgeError as e:
            print(e)
        self.depth_images[index] = np.array(depth_image, dtype=np.float32)
        # print("created depth image")
        
    def image_callback(self, img_data, index):
        try:
            color_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.color_images[index] = np.array(color_image, dtype=np.uint8)
        # print("created color image")

    def get_images(self):
        return self.color_images, self.depth_images


def homogenous_transform(R, t):
    homogeneous_matrix = np.eye(4, dtype=np.float64)
    homogeneous_matrix[0:3, 0:3] = R
    homogeneous_matrix[0:3, 3:4] = t

    return homogeneous_matrix

if __name__ == "__main__": # This is not a function but an if clause !!
    # "global" parameters
    rospy.init_node("segmentation_node")
    scene_publisher = rospy.Publisher("/moveit_planning_scene", CollisionObject, queue_size=1)
    image_subscriber = image_subscriber()
    run_segmentation = False
    depth_images = []
    color_images = []
    rate = rospy.Rate(10)
    model = FastSAM('FastSAM-x.pt')
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"  
    )
    print("using device ", DEVICE)


    # TODO: read extrinsics from file or ROS parameter server
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

   
    # ToDo: Read in intrinsics from zed node
    # published as sensor_msgs/CameraInfo
    # on topic /zed_multi/zed2i_long/zed_nodelet_front/left/camera_info or rear/left/camera_info respectively
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
        user_input = input("Enter 's' to segment the image. See the ZED-Ros node for a preview. Press 'x' to shut down")
        if user_input.lower() == 'x':  # .lower() makes comparison case
            rospy.signal_shutdown("User requested shutdown")
        if user_input.lower() == 's':
            # segment only upon user input
            print("setting images")
            color_image1, color_image2 = image_subscriber.get_images()[0]
            cv2.imshow("color_image1", color_image1)
            cv2.waitKey(0)
            cv2.imshow("color_image2", color_image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            depth_image1, depth_image2 = image_subscriber.get_images()[1]
            cv2.imshow("depth1", depth_image1)
            cv2.waitKey(0)
            cv2.imshow("depth2", depth_image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # convert color scale
            print("creating o3d images")
            # create o3d images
            o3d_depth_1 = o3d.geometry.Image(depth_image1.astype(np.float32))
            o3d_color_1 = o3d.geometry.Image(color_image1.astype(np.uint8))
            # image 2
            o3d_depth_2 = o3d.geometry.Image(depth_image2.astype(np.float32))
            o3d_color_2 = o3d.geometry.Image(color_image2.astype(np.uint8))
           
            print("starting segmentation")
            segmentation_parameters = SegmentationParameters(736, conf=0.5, iou=0.9)
            segmenter = SegmentationMatcher(segmentation_parameters, cutoff=1.5, model_path='FastSAM-x.pt', DEVICE=DEVICE, depth_scale=1.0)
            segmenter.set_camera_params([o3d_intrinsic1, o3d_intrinsic2], [H1, H2])
            segmenter.set_images([color_image1, color_image2], [depth_image1, depth_image2])
            segmenter.preprocess_images(visualize=False)
            # TODO: by mounting a camera on the robot we could reconstruct the entire scene by matching multiple perspectives.
            # For this, we need to track the robots camera position and apply a iou search in n-dimensional space (curse of dimensionylity!!!)
            # We could thus preserve segmentation information.
            # This process may be sped up by using tracking
            mask_arrays = segmenter.segment_color_images_batch(filter_masks=False)  # batch processing of two images saves meagre 0.3 seconds
            segmenter.generate_pointclouds_from_masks()
            global_pointclouds = segmenter.project_pointclouds_to_global()
            # next step also deletes the corresponded poointclouds from general pintcloud array
            correspondences, scores, _, _ = segmenter.match_segmentations(voxel_size=0.05, threshold=0.0) 
            # Here we get the "stitched" objects matched by both cameras
            # TODO (IDEA) we could ICP the resultin pointclouds to find the bet matching geomtric primitives
            corresponding_pointclouds, matched_objects = segmenter.align_corresponding_objects(correspondences, scores, visualize=False)
            # get all unique pointclouds
            pointclouds = segmenter.get_final_pointclouds()
            bounding_boxes = []
            for element in pointclouds:
                element, _ =  element.remove_statistical_outlier(25, 0.5)
                bounding_boxes.append(element.get_minimal_oriented_bounding_box(robust=True))

            #ToDo: publish objects to planning scene
            collision_object = create_planning_scene_object_from_bbox(bounding_boxes[0])
            scene_publisher.publish(collision_object)
            print("published object to the planning scene")

        rate.sleep()
 