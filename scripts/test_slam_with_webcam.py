import cv2
import numpy as np
import sys
# sys.path.insert(0, "/slamdoom/tmp/orbslam2/include/build/")
import ORBSLAM2 as os2
from time import time, sleep
import pickle
from scripts.gridmap import to_gridmap, DisplayMap


def test_SLAM_init():
    # "/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt"
    # "/slamdoom/libs/orbslam2/Examples/RGB-D/TUM1.yaml"

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # print(frame.shape)
    # exit()
    print("Initializing SLAM...")
    slam_obj = os2.SLAM()
    slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "../logitec.yaml", "mono", False)
    print("SLAM was successfully initialized!")
    input("Press key to continue...")

    displayer = DisplayMap(max_side=500)
    i_frame = 0
    while True:
        ret, frame = cap.read()
        slam_obj.track_mono(frame, time())

        kps = slam_obj.get_feature_kps()
        if kps is not None:
            print(kps.shape, np.min(kps), np.max(kps))

        if i_frame % 200 == 0:
            pts = slam_obj.getmap()
            if pts is not None:
                gmap = to_gridmap(pts)
                displayer.new_map(gmap)

    print("SLAM was successfully continued!")
    input("Press key to finish...")

if __name__ == "__main__":
    test_SLAM_init()