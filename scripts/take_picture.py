import cv2
from pyzed import sl
import numpy as np

# Initialize ZED camera
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Set the depth mode to neural
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units 

zed = sl.Camera()
err = zed.open(init_params)

if err != sl.ERROR_CODE.SUCCESS:
    print(f"ZED initialization error: {str(err)}")
else:
    color_image = sl.Mat()
    depth_image = sl.Mat()

    # Capture image
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(color_image, sl.VIEW.LEFT)  # Retrieve left color image
        zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)  # Retrieve depth map

        # Save color image
        cv2.imwrite('color_image.png', color_image.get_data())  # Save as uint16 grayscale image
        print('Color image saved.')

        # Scale depth map and save as uint16 grayscale image
        depth_data = depth_image.get_data()  # Depth map data in millimeters
        max_depth = np.max(depth_data)
        scaled_depth = (depth_data).astype('uint16')
        cv2.imwrite('depth_image.png', scaled_depth)  # Save as uint16 grayscale image
        print('Depth image saved.')

    # Close the camera
    zed.close()
