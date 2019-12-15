import cv2
import numpy as np
import sys
# sys.path.insert(0, "/slamdoom/tmp/orbslam2/include/build/")
import ORBSLAM2 as os2
from time import time, sleep
import pickle
from gridmap import to_gridmap, DisplayMap


def test_SLAM_init():
    # "/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt"
    # "/slamdoom/libs/orbslam2/Examples/RGB-D/TUM1.yaml"

    flag_visualize_gridmap = False
    cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # print(frame.shape)
    # exit()
    print("Initializing SLAM...")
    slam_obj = os2.SLAM()
    # slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "../logitec.yaml", "mono", not flag_visualize_gridmap)
    slam_obj.init("/slamdoom/tmp/orbslam2/Vocabulary/ORBvoc.txt", "../logitec.yaml", "mono", True)
    print("SLAM was successfully initialized!")
    if flag_visualize_gridmap:
        displayer = DisplayMap()
    i_frame = 0
    while True:
        i_frame += 1
        ret, frame = cap.read()
        slam_obj.track_mono(frame, time())

        if flag_visualize_gridmap:
            kps = slam_obj.get_feature_kps()
            displayer.new_frame(frame, kps, slam_obj.tracking_state() == 2)

            if i_frame % 100 == 0:
                pts = slam_obj.getmap()
                if pts is not None:
                    displayer.new_map(pts)
        # if i_frame > 1000:
        #     break


if __name__ == "__main__":
    test_SLAM_init()