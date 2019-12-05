import cv2
import numpy as np
import sys
# sys.path.insert(0, "/slamdoom/tmp/orbslam2/include/build/")
import ORBSLAM2 as os2
from time import time, sleep
import pickle

def test_SLAM_init():
    # "/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt"
    # "/slamdoom/libs/orbslam2/Examples/RGB-D/TUM1.yaml"

    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    print(frame.shape)
    exit()
    print("Initializing SLAM...")
    slam_obj = os2.SLAM()
    slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "../logitec.yaml", "mono", False)
    print("SLAM was successfully initialized!")
    input("Press key to continue...")

    i_frame = 0
    while True:
        ret, frame = cap.read()
        slam_obj.track_mono(frame, time())
        sleep(0.5)
        print(i_frame)


    out = slam_obj.getmap()

    with open("/home/slam_data/data_sets/out.pickle", "wb") as conn:
        pickle.dump(out, conn)

    print("SLAM was successfully continued!")
    input("Press key to finish...")
    pass

if __name__ == "__main__":
    test_SLAM_init()