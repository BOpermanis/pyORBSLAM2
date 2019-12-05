import cv2
import numpy as np
import sys
# sys.path.insert(0, "/slamdoom/tmp/orbslam2/include/build/")
import ORBSLAM2 as os2
from time import sleep
import time
from glob import glob
import pickle
# print("syccesfull import")
# exit()
def test_SLAM_init():
    # "/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt"
    # "/slamdoom/libs/orbslam2/Examples/RGB-D/TUM1.yaml"
    print("Initializing SLAM...")
    slam_obj = os2.SLAM()
    slam_obj.init("/slamdoom/libs/orbslam2/Vocabulary/ORBvoc.txt", "/slamdoom/libs/orbslam2/Examples/Monocular/EuRoC.yaml", "mono", True)
    print("SLAM was successfully initialized!")
    input("Press key to continue...")
    fs = sorted(glob("/home/slam_data/mav0/cam0/data/*"))
    for i_frame, f in enumerate(fs):

        print("frame {} from {}".format(i_frame, len(fs)))
        array = cv2.imread(f)
        slam_obj.track_mono(array, time.time())
        # sleep(1)
        #
        # if i_frame == 300:
        #     # sleep(10)
        #     print(1111)
        #     out = slam_obj.getmap()
        #     print(2222)
        #     with open("/home/slam_data/data_sets/out.pickle", "wb") as conn:
        #         pickle.dump(out, conn)
        #     print(out, out.shape, out.shape[0] / 6)
        #     exit()
    print(1)
    out = slam_obj.getmap()
    print(2)
    with open("/home/slam_data/data_sets/out.pickle", "wb") as conn:
        pickle.dump(out, conn)

    print("SLAM was successfully continued!")
    input("Press key to finish...")
    pass

if __name__ == "__main__":
    test_SLAM_init()