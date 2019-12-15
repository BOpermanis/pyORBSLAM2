import cv2
import numpy as np
import ORBSLAM2 as os2
from time import time, sleep
import pickle
from gridmap import to_gridmap, DisplayMap

import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

flag_visualize_gridmap = True
# ret, frame = cap.read()
# print(frame.shape)
# exit()
print("Initializing SLAM...")
slam_obj = os2.SLAM()
# slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "../logitec.yaml", "mono", not flag_visualize_gridmap)
slam_obj.init("/slamdoom/tmp/orbslam2/Vocabulary/ORBvoc.txt", "../realsense.yaml", "rgbd", True)
print("SLAM was successfully initialized!")
if flag_visualize_gridmap:
    displayer = DisplayMap()
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

    slam_obj.track_rgbd(frame, depth_image, time())

    if flag_visualize_gridmap:
        kps = slam_obj.get_feature_kps()
        displayer.new_frame(frame, kps, slam_obj.tracking_state() == 2)

        if i_frame % 100 == 0:
            pts = slam_obj.getmap()
            if pts is not None:
                displayer.new_map(pts)

