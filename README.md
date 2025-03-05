# zed_vision_pkg
ros package for computer vision/Zed tasks for Master Thesis

There are two main blocks body-tracking and segmentation.
1) Body Tracking - includes an exectuable which runs the hand detection and publishes them to ROS topics (body_tracking2.py)
2) segmentation - Folder includeing the FastSAM model, utilities, some experimental scripts and the class "SegmentationMatcher", which is used
   to segment a pair of images and produce the corresponding geometries and so on. The main node (and ROS-executable) is in the file segmentation_node.py

The launchfile body_tracking_multicam.launchallows you to track the the hand with three zed cameras at once. Use the corresponding YAML files to set all transforms and launch with: "roslaunch zed_vision body_tracking_multicam.launch"

## Running the package:

### Preparation:
- calibrate the three cameras (two at a time)
- Set the camera numbers, transforms and serial numbers in the YAML config file coherently
- set the transforms in body_tracking2 and segmentation_node.py to the correct camera transformations from the previous steps. Be careful to be consistent with the transforms and the serial number
- if needed you can change configurations (like image downsampling, depth quality etc.) in zed-ros-wrapper/params/common.yaml. This code was tested with no downsampling and depth_mode = NEURAL or ULTRA in 1280x720p
- Make sure the controller is running with a move_group node and that Rviz is open. (The segmentation will be published to the planning scene)
### launch steps

   1) roslaunch zed_vision body_tracking_multicam.launch -> Starting takes a while, because each three nodes are started, each running in the same terminal and running body tracking on one camera. Every subsequent node has to wait for the previous node's camera to open before starting, so that
   takes a while!
   The hand pose filter noe (hand_pose_filter) will be run as well. At the moment this (additional) node, takes the average of all recognized hand poses and publishes them (on "/right_hand"), if a hand is recognized. It does not count empty poses, i.e. the camera nodes transmit if a hand has been detected, and the average is run over all DETECTED hands.
   2) run roslaunch zed_vision segmentation_node.py -> once started press "s" when you're ready to segment the scene.
   2.1) Some windows will pop up. They show you (in this order): the captured rgb image, the detected depth images, the bgr images (for open3d) with blackened parts (outside of the "cutoff range").
         Since those images are blocking cv2.imshow calls, press "spacebar" to continue. When you skitpped through all images the NN should begin to load (see messages in terminal)
   2.2) After sgemntation it will show the combined 3D pointcloud in open3d, close this window when you have seen enough. The scene is now published to Rviz. Then another window pops up, showing the bounding boxes.
         Close all windows when you're done. THe node will keep running in the background, until you either press "x" to shut it down or "s" to segment again.
   NOTE: the segmentation is not hugely reliable, especially on feature-less or transparent surfces like white tables or glasses. To combat this, add some clutter and/or increase the lighting. In brighter conditions the             segmentation generally works better. Rarely, it can be completely off, in that case delete the planning scene and run a new segmentaiton.
   NOTE: At the moment the 3D-model of the mounting table is automatically added. In this repo we obviously use our table, so be careful if your setup deviates significantly. You can change the paremeters in the
         function segmentation_node.py -> add_mounting_table()
   3) If you want to activate the force field, inspect the planning scene and correct any unwanted obstacels. For instance, delete the bounding box representing the robot, as it will block all movement. You can also delete any other artifacts or unwanted or unnecessary objects (for instance outside of the actual worksapce). DO NOT PUBLISH it YET!!
   4) When you are ready, run <rosrun goal_state_publisher force field>. Once the node is up, pblish your planning scene. From now on, all the bounding boxes in the planning scene generate a force field at their correspondnig
      real word locations when the end efector moves near it.
   5) Move around and test! Ideally you can test this first in free-floating mode (with 0 stifness) and manually guide the EE around. Then you can feel the force fields for yourself. Use the left hand to guide, since it  will evade your right hand, or try to push it into an object like a jedi :D (it should get blocked)

   Note:
   -Presently, only a few points on the franka hand geometry are avoided, so you might still collide with the sides of the Franka Gripper or the robot!

