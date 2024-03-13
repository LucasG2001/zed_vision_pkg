#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point, Pose
import pandas as pd
import numpy as np
import os
from custom_msgs.msg import HandPose
import time
import math


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
        self.filter_param = 0.9999
        # TODO: filter not on callback reception but at the end - else causes massiv ealisasing at 30 HZ publishing
        # frequency of bodytracking nodes
        self.hololens_hand = HololensHand()
        self.camera_list = ["32689769", "34783283"]

        # Publisher
        self.left_hand_pub = rospy.Publisher('cartesian_impedance_controller/left_hand', Pose, queue_size=1)
        self.right_hand_pub = rospy.Publisher('cartesian_impedance_controller/right_hand', Pose, queue_size=1)

        # Variables to store received points
        self.left_hand_points = {'cam1': Point(), 'cam2': Point()}
        self.right_hand_points = {'cam1': Point(), 'cam2': Point()}
        
        self.vleft = np.array([0, 0, 0])
        self.vright = np.array([0, 0, 0])
        self.right_estimate = np.array([0, 0, 0])
        self.left_estimate = np.array([0, 0, 0])
        self.frequency = 80
        self.dt = 1.0/self.frequency

        #Variable to stor filtered avg data
        self.last_left = np.array([0, 0, 0])
        self.last_right = np.array([0, 0, 0])

        # Subscribers
        for i in range(1, 3):
            rospy.Subscriber(f'/left_hand{self.camera_list[i-1]}', Point, self.left_point_callback, callback_args=f'cam{i}')
            rospy.Subscriber(f'/right_hand{self.camera_list[i-1]}', Point, self.right_point_callback, callback_args=f'cam{i}')
        # create subscriber for hololens
        rospy.Subscriber('/hl_hand_pose', HandPose, self.hololens_callback)

    def left_point_callback(self, data: Point, cam):
        # add filetring with EMA
        if not (math.isnan(data.x)):
            self.left_hand_points[cam].x = data.x 
            self.left_hand_points[cam].y = data.y 
            self.left_hand_points[cam].z = data.z 

        print(f"got message {data} on left hand from camera {cam}")

    def right_point_callback(self, data: Point, cam):
        # add filetring with EMA
        if not (math.isnan(data.x)):
            self.right_hand_points[cam].x = data.x 
            self.right_hand_points[cam].y = data.y
            self.right_hand_points[cam].z =  data.z
        #print(f"got message {data} on right hand from camera {cam}")
    
    # TODO: Integrate HOlolens in Hand Pose
    def hololens_callback(self, msg: HandPose):
        self.hololens_hand.position = msg.position
        self.hololens_hand.orientation = msg.orientation
        self.hololens_hand.isTracked = msg.isTracked
        self.hololens_hand.isUpdated = msg.isUpdated
        print("received hololens message")
        

    def calculate_and_publish_average(self, n_bodies):
        # Calculate the average point for the specified hand
        left_avg = Pose()
        right_avg = Pose()
        left_avg.position.x = 0.0
        left_avg.position.y = 0.0
        left_avg.position.z = 0.0
        left_avg.orientation.x = 1.0
        left_avg.orientation.y = 0.0
        left_avg.orientation.z = 0.0
        left_avg.orientation.w = 0.0

        right_avg.position.x  = 0.0
        right_avg.position.y  = 0.0
        right_avg.position.z  = 0.0
        right_avg.orientation.x = 1.0
        right_avg.orientation.y = 0.0
        right_avg.orientation.z = 0.0
        right_avg.orientation.w = 0.0

        num_points = n_bodies
        
        # handle special case
        if n_bodies == 0:
            print("no bodies were detected yet")
            left = np.array([0, 0, 0])
            right = np.array([0, 0, 0])
            return left, right
        
        i= 0
        for (key1,left_point), (key2, right_point) in zip(self.left_hand_points.items(), self.right_hand_points.items()):
            left_avg.position.x += left_point.x
            left_avg.position.y += left_point.y
            left_avg.position.z += left_point.z

            right_avg.position.x += right_point.x
            right_avg.position.y += right_point.y
            right_avg.position.z += right_point.z

            print(f"left hand of camera {i} at {left_point}")
            print("_______________________________")
            i += 1

        # add hololens tracking for right hand
        if self.hololens_hand.isTracked:
            right_avg.position.x += 6 * self.hololens_hand.position.x
            right_avg.position.y += 6 * self.hololens_hand.position.y
            right_avg.position.z += 6 * self.hololens_hand.position.z

            right_avg.position.x /= (num_points + 6)
            right_avg.position.y /= (num_points + 6)
            right_avg.position.z /= (num_points + 6)

            right_avg.orientation = self.hololens_hand.orientation
            print(f"right average orientation is{right_avg.orientation.x, right_avg.orientation.y, right_avg.orientation.z, right_avg.orientation.w}")

        else:
            right_avg.position.x /= num_points
            right_avg.position.y /= num_points
            right_avg.position.z /= num_points

        left_avg.position.x /= num_points
        left_avg.position.y /= num_points
        left_avg.position.z /= num_points
        # Here we have final average from measurement
        # now we add velocity
        # arrays for logging and other operations
        left = np.array([left_avg.position.x, left_avg.position.y, left_avg.position.z])
        right = np.array([right_avg.position.x, right_avg.position.y, right_avg.position.z])

        kalman_gain = 0.3

        self.vleft = self.vleft * 0.9 + 0.1 * (left - self.last_left) / self.dt
        self.vright = self.vright * 0.9 + 0.1 * (right -self.last_right) / self.dt

        left_avg.position.x =  (1-kalman_gain) * left[0] + kalman_gain * self.left_estimate[0]
        left_avg.position.y =  (1-kalman_gain) * left[1] + kalman_gain * self.left_estimate[1]
        left_avg.position.z =  (1-kalman_gain) * left[2] + kalman_gain * self.left_estimate[2]
        right_avg.position.x = (1-kalman_gain) * right[0] + kalman_gain * self.right_estimate[0]
        right_avg.position.y = (1-kalman_gain) * right[1] + kalman_gain * self.right_estimate[1]
        right_avg.position.z = (1-kalman_gain) * right[2] + kalman_gain * self.right_estimate[2]

        self.last_left = (1 - kalman_gain) * left + kalman_gain * self.left_estimate
        self.last_right = (1 - kalman_gain) * right + kalman_gain * self.right_estimate

        self.left_estimate = left + self.vleft * self.dt
        self.right_estimate = right + self.vright * self.dt

        # Publish the average point for the specified hand
        self.left_hand_pub.publish(left_avg)
        self.right_hand_pub.publish(right_avg)
        
        print(f"right hand is at {right_avg}")
        print(" ")
        print(f"left hand is at {left_avg}")
        print(" ")
        print(f"no points is {num_points}")
        print("___________________________________")

        return self.last_left, self.last_right


if __name__ == '__main__':
    try:
        hand_node = HandAveragerNode()
        rate = rospy.Rate(hand_node.frequency)
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
        pd.DataFrame(columns=['x', 'y', 'z']).to_csv(right_csv_path, index=False)

        while not rospy.is_shutdown():
            start = time.time()
            detected_bodies = rospy.get_param('detected_body_list')
            nr_bodies = np.sum(np.array(detected_bodies))
            if nr_bodies == 0:
                continue
            left_hand, right_hand = hand_node.calculate_and_publish_average(nr_bodies)
            print(hand_node.vright)
            #print(f"loop without csv print took {time.time()-start}") # takes about 0.001 s
            # Append new values to CSV files
            pd.DataFrame([left_hand], columns=['x', 'y', 'z']).to_csv(left_csv_path, mode='a', header=not os.path.isfile(left_csv_path), index=False)
            pd.DataFrame([right_hand], columns=['x', 'y', 'z']).to_csv(right_csv_path, mode='a', header=not os.path.isfile(right_csv_path), index=False)
            #print(f"loop took {time.time()-start}") # takes about 0.003 seconds up to here
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
