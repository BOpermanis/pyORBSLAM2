import cv2
import sys
import numpy as np
import pickle

sys.path.insert(0, "/pyORBSLAM2/src/build")
import ORBSLAM2 as os2
from slam.utils import convert_depth_frame_to_pointcloud
from time import time, sleep
import pickle
import pyrealsense2 as rs

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg = pipeline.start(config)
profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics()

flag_visualize_gridmap = False
# ret, frame = cap.read()
# print(frame.shape)
# exit()
print("Initializing SLAM...")
slam_obj = os2.SLAM()
# slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "../logitec.yaml", "mono", not flag_visualize_gridmap)
slam_obj.init("/slamdoom/tmp/orbslam2/Vocabulary/ORBvoc.txt", "../realsense.yaml", "rgbd", True)
print("SLAM was successfully initialized!")


i_frame = 0
depth_images = []
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
    depth_images.append(depth_image)
    sleep(0.2)
    if i_frame > 50:
        slam_obj.prepare_dump()
        kf_ids_from_mps = slam_obj.get_kf_ids_from_mps()
        kf_ids = slam_obj.get_kf_ids()
        mp_3dpts = slam_obj.get_mp_3dpts()
        kf_3dpts = slam_obj.get_kf_3dpts()
        kf_ids_from_planes = slam_obj.get_kf_ids_from_planes()
        plane_params = slam_obj.get_plane_params()
        frame_ids = slam_obj.get_frame_ids()
        plane_segs = slam_obj.get_plane_segs()

        print("plane_params.shape[0] / 4", plane_params.shape[0] / 4)
        with open("/home/slam_data/data_sets1/temp.pickle", 'wb') as conn:
            s = set(frame_ids.flatten())
            depth_images = [convert_depth_frame_to_pointcloud(a, intr, flag_drop_zero_depth=False) for i, a in enumerate(depth_images) if i in s]
            pickle.dump((kf_ids_from_mps, kf_ids, mp_3dpts, kf_3dpts, kf_ids_from_planes, plane_params, frame_ids, depth_images, plane_segs), conn)
        print(11111111111111111111111)
        break

del slam_obj