import cv2
import sys
import numpy as np
import pickle

sys.path.insert(0, "/pyORBSLAM2/src/build")
import ORBSLAM2 as os2
from time import time, sleep
import pickle
import pyrealsense2 as rs

from PIL import Image

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

flag_visualize_gridmap = False
# ret, frame = cap.read()
# print(frame.shape)
# exit()
print("Initializing SLAM...")
slam_obj = os2.SLAM()
# slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "../logitec.yaml", "mono", not flag_visualize_gridmap)
slam_obj.init("/slamdoom/tmp/orbslam2/Vocabulary/ORBvoc.txt", "../realsense.yaml", "rgbd", False)
print("SLAM was successfully initialized!")


def get_id_color(i):
    np.random.seed(i)
    return np.asarray([
        np.random.randint(0, 255, dtype=np.uint8),
        np.random.randint(0, 255, dtype=np.uint8),
        np.random.randint(0, 255, dtype=np.uint8)
    ])

i_frame = 0
while True:
    i_frame += 1

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())

    x = slam_obj.visualize_cape(frame, depth_image.astype(np.float32))
    for i in np.unique(x):
        if i!=0:
            mask1 = x == i
            color = get_id_color(i)
            # print(color.shape, frame.shape)
            frame[mask1, :] = (frame[mask1, :] + color) // 2
            # print(frame.shape, frame.dtype)
            # exit()

    print("len(np.unique(x))", len(np.unique(x)))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # print(x.shape, x.dtype)
    # print(np.unique(x))
    # # Image.fromarray(frame).show()
    # break


del slam_obj