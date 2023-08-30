# zed_vision_pkg
ros package for computer vision/Zed tasks for Master Thesis

There are two main blocks body-tracking and segmentation.
1) Body Tracking - includes an exectuable which runs the hand detection and publishes them to ROS topics (body_tracking2.py)
2) segmentation - Folder includeing the FastSAM model, utilities, some experimental scripts and the class "SegmentationMatcher", which is used
   to segment a pair of images and produce the corresponding geometries and so on. The main node (and ROS-executable) is in the file segmentation_node.py

There are no further executables in this package at the moment.

Package Requirements:

