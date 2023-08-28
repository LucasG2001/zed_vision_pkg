import cv2
import sys
import os

os.add_dll_directory(r"C:\Users\Willi\AppData\Local\Programs\Python\Python310\Lib\site-packages\pyzed")
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import numpy as np
import signal
import time
import threading

zed_list = []
left_list = []
depth_list = []
timestamp_list = []
thread_list = []
depth_images = []
color_images = []
viewer_list = []
stop_signal = False


def signal_handler(signal, frame):
    global stop_signal
    stop_signal = True
    time.sleep(0.5)
    exit()


def extract_image_data(cam, depth_for_display, depth_image, color_image, viewer):
    cam.retrieve_image(depth_for_display, sl.VIEW.DEPTH)  # only for display purposes
    cam.retrieve_measure(depth_image, sl.MEASURE.DEPTH)  # This is the data
    cam.retrieve_image(color_image, sl.VIEW.LEFT)  # color image
    # Use get_data() to get the numpy array
    image_depth_ocv = depth_for_display.get_data()
    # !!!CAUTION!!! depth image will be near black if not multiplied by 1 or 10 thousand
    # since uint16 range is up to 65'535, even 1k looks black
    depth_image_ocv = depth_image.get_data() * 10000
    depth_image_ocv = depth_image_ocv.astype(np.uint16)
    color_image_ocv = color_image.get_data().astype(np.uint8)
    # Display the depth view from the numpy array
    if viewer.save_data:
        print("images extracted")
        viewer.save_data = False
        return color_image_ocv, depth_image_ocv
    else:
        return 0, 0


def grab_run(index):
    global stop_signal
    global zed_list
    global timestamp_list
    global left_list
    global depth_list
    global viewer_list
    camera_model = zed_list[index].get_camera_information().camera_model
    res = sl.Resolution()
    res.width = 1280
    res.height = 720
    viewer_list[index].init(1, sys.argv, camera_model, res)
    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)  # color image
            zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.DEPTH)  # depth image
            # depth_images[index] = depth_list[index].get_data()
            # color_images[index] = left_list[index].get_data()
            timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        time.sleep(0.001)  # 1ms
    zed_list[index].close()

    # Display the depth view from the numpy array
    """
       if viewer.save_data:
        print("images extracted")
        viewer.save_data = False
        return color_image_ocv, depth_image_ocv
    else:
        return 0, 0
    """



def main():
    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global timestamp_list
    global thread_list
    global color_images
    global depth_images
    global viewer_list
    signal.signal(signal.SIGINT, signal_handler)

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30  # The framerate is lowered to avoid any USB3 bandwidth issues
    init.coordinate_units = sl.UNIT.METER  # Set coordinate units
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # set camera resolution
    # List and open cameras
    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()
    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append("ZED {}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))
        zed_list.append(sl.Camera())
        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        viewer_list.append(gl.GLViewer())
        timestamp_list.append(0)
        last_ts_list.append(0)
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        index = index + 1

    """
      viewer1 = gl.GLViewer()
    viewer2 = gl.GLViewer()
    viewers = [viewer1, viewer2]
    """
    # Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()
            camera_model = zed_list[index].get_camera_information().camera_model
            # viewers[index].init(1, sys.argv, camera_model, res)
            # ToDo: write camera intrinsics to file
            zed_params = zed_list[index].get_camera_information().calibration_parameters
            print("ZED Camera Intrinsics:")
            print(f"Focal Length (fx): {zed_params.left_cam.fx}")
            print(f"Focal Length (fy): {zed_params.left_cam.fy}")
            print(f"Principal Point (cx): {zed_params.left_cam.cx}")
            print(f"Principal Point (cy): {zed_params.left_cam.cy}")

    depth_for_display = sl.Mat()
    depth_image = sl.Mat()
    color_image = sl.Mat()
    zed1 = zed_list[0]
    zed2 = zed_list[1]
    # Display camera images
    # Display camera images
    key = ''
    while key != 113:  # for 'q' key
        for index in range(0, len(zed_list)):
            if zed_list[index].is_opened():
                if timestamp_list[index] > last_ts_list[index]:
                    cv2.imshow(name_list[0], left_list[0].get_data())
                    x = round(depth_list[index].get_width() / 2)
                    y = round(depth_list[index].get_height() / 2)
                    err, depth_value = depth_list[index].get_value(x, y)
                    last_ts_list[index] = timestamp_list[index]
        key = cv2.waitKey(2)
    cv2.destroyAllWindows()

    # Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()

    print("\nFINISH")


if __name__ == "__main__":
    main()
