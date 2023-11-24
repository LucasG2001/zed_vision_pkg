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
from moveit_msgs.msg import PlanningSceneWorld, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

def homogenous_transform(R, t):
    homogeneous_matrix = np.eye(4, dtype=np.float64)
    homogeneous_matrix[0:3, 0:3] = R
    homogeneous_matrix[0:3, 3:4] = t

    return homogeneous_matrix

def inverse_transform(R, t):
    H_inv = np.eye(4, dtype=np.float64)
    H_inv[0:3, 0:3] = np.transpose(R)
    H_inv[0:3, 3] = -np.transpose(R) @ t

    return H_inv

def get_bboxes_for_force_field(bbox, primitive, R, t, index):
    aligned_collision_object = CollisionObject()
    aligned_collision_object.header.frame_id = "panda_link0"
    transform = TransformStamped()
    # transform oriented bounding box to axis aligned bounding box with center at (0,0,0)
    aligned_bounding_box = o3d.geometry.OrientedBoundingBox(bbox)
    inv_R = np.transpose(R)
    aligned_bounding_box.rotate(R=inv_R)
    aligned_bounding_box.translate(-inv_R @ t, relative=True) # now the bbox should be axis aligned and centered at origin
    orientation = np.array(aligned_bounding_box.R)
    # In this quaternion we save the necessary transform to align the bounding box
    transform_quat = Rotation.from_matrix(inv_R).as_quat()
    # fill transform
    position = aligned_bounding_box.center
    transform.transform.translation.x , transform.transform.translation.y, transform.transform.translation.z = -inv_R @ t # we want the bbox at 0 0 0 but need the corresponding transform
    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = transform_quat
    # fill pose
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position 
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = Rotation.from_matrix(orientation).as_quat() 
    # fill collision object
    aligned_collision_object.id = "aligned nr. " + str(index)
    aligned_collision_object.primitives.append(primitive)
    aligned_collision_object.primitive_poses.append(pose)
    aligned_collision_object.operation = CollisionObject.ADD
    
    return aligned_collision_object, transform

def add_mounting_table():
    """
    This function returns a collision object corresponding to the table the robot is mounted on
    we can return the mounting table both for the force field node and the planning scene, without transformation, since
    for the force generation we need only the dimensions anyway (and can assume the box to be at 0 0 0, just transforming the EE)
    """
    table = CollisionObject()
    table.header.frame_id = "panda_link0"
    transform = TransformStamped()
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = [0.45, 0.3, -0.02]  # pose of table center 
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = [0, 0, 0, 1]
    # fill collision object
    table.id = "mounting_table"
    # create corresponding primitive 
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = [0.9, 1.5, 0.04]
    table.primitives.append(primitive)
    table.primitive_poses.append(pose)
    table.operation = CollisionObject.ADD
    # create corresponding transform 
    transform.transform.translation.x , transform.transform.translation.y, transform.transform.translation.z = [-0.45, -0.3, 0.02]
    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = [0, 0, 0, 1]
    
    return table, transform

def create_planning_scene_object_from_bbox(bboxes, id = "1"):
    transforms_msg = TFMessage()
    force_field_planning_scene = PlanningSceneWorld()
    collision_objects = []
    for i, bbox in enumerate(bboxes):
        oriented_collision_object = CollisionObject()
        oriented_collision_object.header.frame_id = "panda_link0"
        R = np.array(bbox.R) # Rotation Matrix of bounding box
        center = bbox.center
        sizes = bbox.extent
        quat_R = (Rotation.from_matrix(R)).as_quat()
        # create corresponding primitive 
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = sizes
        # we need to publish a pose and a size, to spawn a rectangle of size S at pose P in the moveit planning scene
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = center # should generally not be 0 0 0 here
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat_R
        # TODO: get id's to be consistent throughout segmentations and the planning scene
        oriented_collision_object.id = str(i)
        oriented_collision_object.primitives.append(primitive)
        oriented_collision_object.primitive_poses.append(pose)
        oriented_collision_object.operation = CollisionObject.ADD
        collision_objects.append(oriented_collision_object)

        # fill axis aligned bounding boxes for Force field genreation
        aligned_bbox, ee_transform = get_bboxes_for_force_field(bbox, primitive, R, center, i)
        force_field_planning_scene.collision_objects.append(aligned_bbox)
        transforms_msg.transforms.append(ee_transform)
    
    # get the mounting table in any case
    table_object, table_transform = add_mounting_table()
    collision_objects.append(table_object)
    force_field_planning_scene.collision_objects.append(table_object)
    transforms_msg.transforms.append(table_transform)

        
    return collision_objects, force_field_planning_scene, transforms_msg



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
            color_image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")  # use rgb8 for open3d color palette
        except CvBridgeError as e:
            print(e)
        self.color_images[index] = np.array(color_image, dtype=np.uint8)
        # print("created color image")

    def get_images(self):
        return self.color_images, self.depth_images

# TODO: add argc, argv to choose levels of visualization (none, debug, user, visualize all)
if __name__ == "__main__": # This is not a function but an if clause !!
    # "global" parameters
    rospy.init_node("segmentation_node")
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
    # TODO: make reaidng extrinsics consistent with the camera in use
    T_0S = np.array([[-1, 0, 0, 0.41],  # Transformations from Robot base (0) to Checkerboard Frame (S)
                     [0, 1, 0, 0.0],
                     [0, 0, -1, 0.006],
                     [0, 0, 0, 1]])
    
    # camera higher up is camera 0
    # 0 is long/front and SN 32689769
    # 1 is short/rear
    rotations = {"camera0": np.array([[-0.74319018, -0.56269495,  0.36199826],
                                      [ 0.66867618, -0.64344379,  0.37262884],
                                      [ 0.02324917,  0.51899371,  0.85446182]]),

                 "camera1": np.array([[ 0.12160326,  0.87553127, -0.46760843],
                                      [-0.98941778,  0.06935403, -0.12744595],
                                      [-0.07915238,  0.47815794,  0.87469988]])}
    
                   

    translations = {"camera0": np.array([[-0.34671173], [-0.45521386], [-0.89451421]]),
                    "camera1": np.array([[0.54132945], [0.61558458], [-1.21148224]])}

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
        user_input = input("Enter 's' to segment the image. See the ZED-Ros node for a preview. Press 'x' to shut down. Press 'c' to clear scene ")
        if user_input.lower() == 'x':  # .lower() makes comparison case
            rospy.signal_shutdown("User requested shutdown")
        if user_input.lower() == 's':
            # segment only upon user input
            print("setting images")
            color_image1, color_image2 = image_subscriber.get_images()[0]
            cv2.imshow("color_image1", color_image1)
            cv2.imwrite("images/color1.png", color_image1)
            cv2.waitKey(0)
            cv2.imshow("color_image2", color_image2)
            cv2.imwrite("images/color2.png", color_image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            depth_image1, depth_image2 = image_subscriber.get_images()[1]
            cv2.imshow("depth1", depth_image1)
            cv2.imwrite("images/depth1.png", depth_image1)
            cv2.waitKey(0)
            cv2.imshow("depth2", depth_image2)
            cv2.imwrite("images/depth2.png", depth_image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("starting segmentation")
            segmentation_parameters = SegmentationParameters(736, conf=0.4, iou=0.9)
            segmenter = SegmentationMatcher(segmentation_parameters, cutoff=2.0, model_path='FastSAM-x.pt', DEVICE=DEVICE, depth_scale=1.0)
            segmenter.set_camera_params([o3d_intrinsic1, o3d_intrinsic2], [H1, H2])
            segmenter.set_images([color_image1, color_image2], [depth_image1, depth_image2])
            segmenter.preprocess_images(visualize=False)
            # TODO: by mounting a camera on the robot we could reconstruct the entire scene by matching multiple perspectives.
            # For this, we need to track the robots camera position and apply a iou search in n-dimensional space (curse of dimensionylity!!!)
            # We could thus preserve segmentation information.
            # This process may be sped up by using tracking
            mask_arrays = segmenter.segment_color_images(filter_masks=True, visualize=False)  # batch processing of two images saves meagre 0.3 seconds
            segmenter.generate_pointclouds_from_masks(visualize=False)
            global_pointclouds = segmenter.project_pointclouds_to_global(visualize=False)
            # next step also deletes the corresponded poointclouds from general pintcloud array
            correspondences, scores, _, _ = segmenter.match_segmentations(voxel_size=0.05, threshold=0.0) 
            # Here we get the "stitched" objects matched by both cameras
            # TODO (IDEA) we could ICP the resultin pointclouds to find the bet matching geomtric primitives
            corresponding_pointclouds, matched_objects = segmenter.align_corresponding_objects(correspondences, scores, 
                                                                                               visualize=True, use_icp=True)
            # get all unique pointclouds
            pointclouds = segmenter.get_final_pointclouds()
            bounding_boxes = []
            for element in matched_objects:
                element, _ =  element.remove_statistical_outlier(25, 0.5)
                bbox = element.get_minimal_oriented_bounding_box(robust=True)
                bbox.color = (1, 0, 0)  # open3d RED
                bounding_boxes.append(bbox) # here bbox center is not 0 0 0

            
            #ToDo: publish objects to planning scene
            collision_objects, force_field_planning_scene, transforms = create_planning_scene_object_from_bbox(bounding_boxes)
            for object in collision_objects:
                scene_publisher.publish(object)
                rospy.sleep(0.05)
            print("published object to the planning scene")
            # transform axis aligned bboxes and corrresponding ee-transforms to the force field planner
            force_field_publisher.publish(force_field_planning_scene)
            #For debug
            """
            for object in force_field_planning_scene.collision_objects:
                scene_publisher.publish(object)
                rospy.sleep(0.05)
            """ # End of debug
            transform_publisher.publish(transforms)
            matched_objects.extend(bounding_boxes)
            print("visualizing detected planning scene")
            o3d.visualization.draw_geometries(matched_objects)
        if user_input.lower() == 'c':
            empty_scene = PlanningScene()
            empty_scene.is_diff = False
            moveit_planning_scene_publisher.publish(empty_scene)

        rate.sleep()
 
 # Done: write a code which does the following steps: 
 # PYTHON
 # DONE
 # 1) take oriented bounding boxes and transforms them to be axis aligned. In praxis we just need the three coordinate 
 #    intervals (xmin, xmax; ymin, ymax; zmin, zmax) and to publish them together with the transform 
 # 2) publishes axis aligned bounding boxes to a "bbox" topic, and also publishes the corresponding 
 #    transforms to a "transform" topic or does both at the same time
 # ----------------------------------------------------------------------
# TODO: write a code which does the following steps:
 # C++ 
 # 1) Write a node that subscribes to the topic "ee-position"
 # 2) subscribe to the "bounding box" topic and/or to the "transform" topic
 # 3) create force field publisher
 # 4) for each bounding box, compute the nearest point on the box w.r.t to the end effector
 #  4.1) extract the three coordinate intervals in x, y, and z direction which delimit the bounding box
 #  4.2 transform the ee-position to be in the same frame as the axis aligned bounding boxes
 #  4.3) For each coordinate interval (x,y,z) do: (and use p' as the projected point)
 #          if ee[x] < max(x) && ee[x] > min(x)
 #              p' = ee[x]
 #          else if ee[x] > max(x):
 #              p' = max(x)
 #          else if ee[x] < min(x):
 #              p' = min(x)
 #  4.4) Compute F_i in dependence of p' and the end-effector
 # 5) Add up all F_i and publish them to a "Force Field" topic