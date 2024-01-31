#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
import pandas as pd
import numpy as np
import os
from custom_msgs.msg import HandPose


# Create /hand_data folder if it doesn't exist
folder_path = 'hand_data'
# Get the parent directory of the currently executed script
script_directory = os.path.dirname(os.path.abspath(__file__))
print(script_directory)
os.makedirs(folder_path, exist_ok=True)

# Create container class for HOlolens
class HololensHand:
    def __init__(self):
      self.position = None
      self.orientation = None
      self.isTracked = False
      self.isUpdated = True


class HandAveragerNode:
    def __init__(self):
        rospy.init_node('hand_averager_node', anonymous=True)
        
        # filter parameter
        self.filter_param = 0.1
        # TODO: filter not on callback reception but at the end - else causes massiv ealisasing at 30 HZ publishing
        # frequency of bodytracking nodes
        self.hololens_hand = HololensHand()
        self.camera_list = ["32689769", "38580376", "34783283"]

        # Publisher
        self.left_hand_pub = rospy.Publisher('cartesian_impedance_controller/left_hand', Point, queue_size=1)
        self.right_hand_pub = rospy.Publisher('cartesian_impedance_controller/right_hand', Point, queue_size=1)

        # Variables to store received points
        self.left_hand_points = {'cam1': Point(), 'cam2': Point(), 'cam3': Point()}
        self.right_hand_points = {'cam1': Point(), 'cam2': Point(), 'cam3': Point()}

        # Subscribers
        for i in range(1, 4):
            rospy.Subscriber(f'/left_hand{self.camera_list[i-1]}', Point, self.left_point_callback, callback_args=f'cam{i}')
            rospy.Subscriber(f'/right_hand{self.camera_list[i-1]}', Point, self.right_point_callback, callback_args=f'cam{i}')
        # create subscriber for hololens
        rospy.Subscriber('hl_hand_pose', HandPose, self.hololens_callback)

    def left_point_callback(self, data: Point(), cam):
        # add filetring with EMA
        self.left_hand_points[cam].x = self.filter_param * data.x + (1-self.filter_param) * self.left_hand_points[cam].x
        self.left_hand_points[cam].y = self.filter_param * data.y + (1-self.filter_param) * self.left_hand_points[cam].y
        self.left_hand_points[cam].z = self.filter_param * data.z + (1-self.filter_param) * self.left_hand_points[cam].z

        print(f"got message {data} on left hand from camera {cam}")

    def right_point_callback(self, data: Point(), cam):
        # add filetring with EMA
        self.right_hand_points[cam].x = self.filter_param * data.x + (1-self.filter_param) * self.right_hand_points[cam].x
        self.right_hand_points[cam].y = self.filter_param * data.y + (1-self.filter_param) * self.right_hand_points[cam].y
        self.right_hand_points[cam].z = self.filter_param * data.z + (1-self.filter_param) * self.right_hand_points[cam].z
        #print(f"got message {data} on right hand from camera {cam}")
    
    # TODO: Integrate HOlolens in Hand Pose
    def hololens_callback(self, msg: HandPose):
        self.hololens_hand.position = msg.position
        self.hololens_hand.orientation = msg.orientation
        self.hololens_hand.isTracked = msg.isTracked
        self.hololens_hand.isUpdated = msg.isUpdated
        

    def calculate_and_publish_average(self, n_bodies):
        # Calculate the average point for the specified hand
        left_avg = Point()
        right_avg = Point()
       
        num_points = n_bodies
        
        # handle special case
        if n_bodies == 0:
            print("no bodies were detected yet")
            left = np.array([0, 0, 0])
            right = np.array([0, 0, 0])
            return left, right
        
        i= 0
        for (key1,left_point), (key2, right_point) in zip(self.left_hand_points.items(), self.right_hand_points.items()):
            left_avg.x += left_point.x
            left_avg.y += left_point.y
            left_avg.z += left_point.z

            right_avg.x += right_point.x
            right_avg.y += right_point.y
            right_avg.z += right_point.z

            print(f"left hand of camera {i} at {left_point}")
            print("_______________________________")
            i += 1

        # add hololens tracking for right hand
        if self.hololens_hand.isTracked:
            right_avg.x += self.hololens_hand.position.x
            right_avg.y += self.hololens_hand.position.y
            right_avg.z += self.hololens_hand.position.z

            right_avg.x /= num_points + 1
            right_avg.y /= num_points + 1
            right_avg.z /= num_points + 1

        else:
            right_avg.x /= num_points
            right_avg.y /= num_points
            right_avg.z /= num_points

        left_avg.x /= num_points
        left_avg.y /= num_points
        left_avg.z /= num_points

        # Publish the average point for the specified hand
        self.left_hand_pub.publish(left_avg)
        self.right_hand_pub.publish(right_avg)
        # arrays for logging
        left = np.array([left_avg.x, left_avg.y, left_avg.z])
        right = np.array([right_avg.x, right_avg.y, right_avg.z])
        
        print(f"right hand is at {right_avg}")
        print(" ")
        print(f"left hand is at {left_avg}")
        print(" ")
        print(f"no points is {num_points}")
        print("___________________________________")

        return left, right


if __name__ == '__main__':
    try:
        hand_node = HandAveragerNode()
        rate = rospy.Rate(200)
        print("Established hand pose node")

        # Initialize data for logging
        right_hand_timeseries = np.empty((0, 3))
        left_hand_timeseries = np.empty((0, 3))

        # Initialize detected bodies
        detected_bodies = [0, 0, 0]  # initialize parameter
        rospy.set_param('detected_body_list', detected_bodies)

        # Create CSV files if they don't exist
        left_csv_path = os.path.join(script_directory, 'left_hand_data.csv')
        right_csv_path = os.path.join(script_directory, 'right_hand_data.csv')
        print(left_csv_path)

        
        pd.DataFrame(columns=['x', 'y', 'z']).to_csv(left_csv_path, index=False)

        if not os.path.isfile(right_csv_path):
            pd.DataFrame(columns=['x', 'y', 'z']).to_csv(right_csv_path, index=False)

        while not rospy.is_shutdown():
            detected_bodies = rospy.get_param('detected_body_list')
            nr_bodies = np.sum(np.array(detected_bodies))
            if nr_bodies == 0:
                continue
            left_hand, right_hand = hand_node.calculate_and_publish_average(nr_bodies)

            # Append new values to CSV files
            pd.DataFrame([left_hand], columns=['x', 'y', 'z']).to_csv(left_csv_path, mode='a', header=not os.path.isfile(left_csv_path), index=False)
            pd.DataFrame([right_hand], columns=['x', 'y', 'z']).to_csv(right_csv_path, mode='a', header=not os.path.isfile(right_csv_path), index=False)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass