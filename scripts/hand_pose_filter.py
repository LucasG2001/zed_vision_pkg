#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
import pandas as pd
import numpy as np
import os

# Create /hand_data folder if it doesn't exist
folder_path = 'hand_data'
os.makedirs(folder_path, exist_ok=True)

class HandAveragerNode:
    def __init__(self):
        rospy.init_node('hand_averager_node', anonymous=True)

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

    def left_point_callback(self, data: Point(), cam):
        self.left_hand_points[cam] = data
        #print(f"got message {data} on left hand from camera {cam}")

    def right_point_callback(self, data: Point(), cam):
        self.right_hand_points[cam] = data
        #print(f"got message {data} on right hand from camera {cam}")
        

    def calculate_and_publish_average(self, n_bodies):
        # Calculate the average point for the specified hand
        left_avg = Point()
        right_avg = Point()
        num_points = n_bodies
        
        # handle special case
        if n_bodies == 0:
            print("no bodies were detected yet")
            return 
        
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

        right_avg.x /= num_points
        right_avg.y /= num_points
        right_avg.z /= num_points

        left_avg.x /= num_points
        left_avg.y /= num_points
        left_avg.z /= num_points

        # Publish the average point for the specified hand
        self.left_hand_pub.publish(left_avg)
        self.right_hand_pub.publish(right_avg)
        
        print(f"right hand is at {right_avg}")
        print(" ")
        print(f"left hand is at {left_avg}")
        print(" ")
        print(f"no points is {num_points}")
        print("___________________________________")


if __name__ == '__main__':
    try:
        hand_node = HandAveragerNode()
        rate = rospy.Rate(15)
        print("established hand pose node")
        #initialize data for logging
        #right_hand_timeseries = np.empty((0,3))
        #left_hand_timeseries = np.empty((0,3))
        # initialize detected bodies 
        detected_bodies = [0, 0, 0]  # initialize parameter
        rospy.set_param('detected_body_list', detected_bodies)
        while not rospy.is_shutdown():
            detected_bodies = rospy.get_param('detected_body_list')
            nr_bodies = np.sum(np.array(detected_bodies))
            hand_node.calculate_and_publish_average(nr_bodies)
            #hand_node.right_hand_points.clear()
            #hand_node.left_hand_points.clear()
            # print(hand_node.right_hand_points)
            #right_hand_timeseries = np.vstack((right_hand_timeseries, right_hand))        
            #left_hand_timeseries = np.vstack((left_hand_timeseries, left_hand))
            rate.sleep()
        # create dataframe for plotting
        # TODO: Create Logging that actually works, the problem is that upon shutdown this code is not executed anymore
        #right_hand_df = pd.DataFrame(right_hand_timeseries, columns=['x', 'y', 'z'])
        #left_hand_df = pd.DataFrame(left_hand_timeseries, columns=['x', 'y', 'z'])
        ## Save DataFrames to CSV files
        #left_hand_df.to_csv(os.path.join(folder_path, 'left_hand_data.csv'))
        #right_hand_df.to_csv(os.path.join(folder_path, 'right_hand_data.csv'))

    except rospy.ROSInterruptException:
        pass
