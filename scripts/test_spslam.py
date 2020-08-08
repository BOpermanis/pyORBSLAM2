import cv2
import sys
import numpy as np

sys.path.insert(0, "/pyORBSLAM2/src_spslam/build")
import ORBSLAM2 as os2
from time import time, sleep
import pickle

import pyrealsense2 as rs

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
slam = os2.SLAM()
# slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "../logitec.yaml", "mono", not flag_visualize_gridmap)
slam.init("/slamdoom/tmp/orbslam2/Vocabulary/ORBvoc.txt", "/SP-SLAM/Examples/RGB-D/realsense.yaml", "rgbd", True)
print("SLAM was successfully initialized!")

i_frame = 0
while True:
    i_frame += 1

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data()).copy()
    frame = np.asanyarray(color_frame.get_data()).copy()

    slam.track_rgbd(frame, depth_image, time())
    if i_frame > 100:
        slam.prepare_dump()
        kf_ids_from_mps = slam.get_kf_ids_from_mps()
        kf_ids = slam.get_kf_ids()
        plane_ids = slam.get_plane_ids()
        mp_3dpts = slam.get_mp_3dpts()
        kf_3dpts = slam.get_kf_3dpts()
        plane_ids_from_boundary_pts = slam.get_plane_ids_from_boundary_pts()
        plane_params = slam.get_plane_params()
        boundary_pts = slam.get_boundary_pts()
        boundary_updates = slam.get_boundary_update_sizes()
        with open("/home/slam_data/data_sets/spspalm_map.pickle", 'wb') as conn:
            pickle.dump((kf_ids_from_mps, kf_ids, plane_ids, mp_3dpts, kf_3dpts, plane_ids_from_boundary_pts, plane_params, boundary_pts, boundary_updates), conn)
        break

del slam

