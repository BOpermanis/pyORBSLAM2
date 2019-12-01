import cv2
import numpy as np
import ORBSLAM2 as os2
from time import sleep
import time
from glob import glob
from time import sleep
import pickle


def test_SLAM_init():
    # "/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt"
    # "/slamdoom/libs/orbslam2/Examples/RGB-D/TUM1.yaml"
    print("Initializing SLAM...")
    slam_obj = os2.SLAM()
    slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "/slamdoom/libs/orbslam2/Examples/Stereo/EuRoC.yaml", "stereo", True)
    print("SLAM was successfully initialized!")
    input("Press key to continue...")
    fs0 = sorted(glob("/home/slam_data/mav0/cam0/data/*"))
    fs1 = sorted(glob("/home/slam_data/mav0/cam1/data/*"))
    for i_frame, (f0, f1) in enumerate(zip(fs0, fs1)):

        print("frame {} from {}".format(i_frame, len(fs1)))
        img0 = cv2.imread(f0)
        img1 = cv2.imread(f1)
        slam_obj.track_stereo(img1, img0, time.time())

        # if i_frame == 150:
        #     # sleep(10)
        #     out = slam_obj.getmap()
        #     with open("/home/slam_data/data_sets/out.pickle", "wb") as conn:
        #         pickle.dump(out, conn)
        #     print(out.shape, out.shape[0] / 6)
        #     exit()
    out = slam_obj.getmap()
    with open("/home/slam_data/data_sets/out.pickle", "wb") as conn:
        pickle.dump(out, conn)
    print("SLAM was successfully continued!")
    input("Press key to finish...")
    pass

if __name__ == "__main__":
    test_SLAM_init()