import numpy as np
import open3d as o3d
import rospy
from sensor_msgs.msg import Image, CameraInfo
from moveit_msgs.msg import CollisionObject
from moveit_msgs.msg import PlanningSceneWorld, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import cv2
import yaml
import os

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
    
def homogenous_transform(R, t):
    homogeneous_matrix = np.eye(4, dtype=np.float64)
    homogeneous_matrix[0:3, 0:3] = R
    homogeneous_matrix[0:3, 3:4] = t

    return homogeneous_matrix

def inverse_transform(H):
    H_inv = np.eye(4, dtype=np.float64)
    H_inv[0:3, 0:3] = np.transpose(H[0:3, 3])
    H_inv[0:3, 3] = -np.transpose(H[0:3, 3]) @ H[0:3, 3] 
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

def show_input_images(color_image1, color_image2, depth_image1, depth_image2, save, dir):
        # color
       cv2.imshow("color_image1", color_image1)
       cv2.waitKey(0)
       cv2.imshow("color_image2", color_image2)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
        # depth
       cv2.imshow("depth1", depth_image1)
       cv2.waitKey(0)
       cv2.imshow("depth2", depth_image2)
       cv2.waitKey(0)
       cv2.destroyAllWindows()

       if (save):
        print("saving images")
        save_folder = "images"

        if not os.path.exists(os.path.join(dir, save_folder)):
            os.makedirs(os.path.join(dir, save_folder))
        # Construct the save path for color_image1
        save_path_color_image1 = 'home/sopho/catkin_ws/src/zed_vision/segmentation/images/color_image1.png'
        # Construccallabl
        save_path_color_image2 = 'home/sopho/catkin_ws/src/zed_vision/segmentation/images/color_image2.png'
        # Construccallabl
        save_path_depth_image1 = 'home/sopho/catkin_ws/src/zed_vision/segmentation/images/depth_image1.png'
        # Construccallabl
        save_path_depth_image2 ='home/sopho/catkin_ws/src/zed_vision/segmentation/images/depth_image2.png'

        # Save images
        cv2.imwrite(save_path_color_image1, color_image1)
        cv2.imwrite(save_path_color_image2, color_image2)
        cv2.imwrite(save_path_depth_image1, depth_image1)
        cv2.imwrite(save_path_depth_image2, depth_image2)