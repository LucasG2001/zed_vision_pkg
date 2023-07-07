import rospy    
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def create_markers(left_hand, right_hand, reference_frame="map"):
    # Create a MarkerArray message object
    marker_array_msg = MarkerArray()
    # Create two Marker message object
    left_marker = Marker()
    right_marker = Marker()
    


    left_marker.header.frame_id = reference_frame # Specify the frame in which the marker will be displayed
    left_marker.id = 1
    left_marker.type = Marker.SPHERE  # Set the marker type to points
    left_marker.action = Marker.ADD  # Set the action to add the marker
    # Define the color of the marker
    left_marker.color.r = 1.0  # Red
    left_marker.color.g = 0.0  # Green
    left_marker.color.b = 0.0  # Blue
    left_marker.color.a = 1.0  # Alpha (transparency)

    # Define the scale of the marker
    left_marker.scale.x = 0.1  # Size of the points
    left_marker.scale.y = 0.1
    left_marker.scale.z = 0.1
    
   
    right_marker.id = 2
    right_marker.header.frame_id = reference_frame # Specify the frame in which the marker will be displayed
    right_marker.type = Marker.SPHERE  # Set the marker type to pointsleft_marker
    right_marker.action = Marker.ADD  # Set the action to add the marker
    # Define the color of the marker
    right_marker.color.r = 0.0  # Red
    right_marker.color.g = 0.0  # Green
    right_marker.color.b = 1.0  # Blue
    right_marker.color.a = 1.0  # Alpha (transparency)

    # Define the scale of the marker
    right_marker.scale.x = 0.1  # Size of the points
    right_marker.scale.y = 0.1
    right_marker.scale.z = 0.1

    # insert poses 0 for left 1 for right
    left_marker.pose.position = left_hand
    right_marker.pose.position = right_hand

    # Add the markers to the MarkerArray
    marker_array_msg.markers.append(left_marker)
    marker_array_msg.markers.append(right_marker)

    return marker_array_msg