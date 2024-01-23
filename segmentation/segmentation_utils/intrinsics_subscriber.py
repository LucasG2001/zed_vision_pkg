
import open3d as o3d
import rospy
from sensor_msgs.msg import CameraInfo

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