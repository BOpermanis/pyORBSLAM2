import numpy as np
import cv2
from matplotlib import pyplot as plt
from glob import glob
from pprint import pprint

# fsl = sorted(glob("/home/slam_data/mav0/cam0/data/*"))
# fsr = sorted(glob("/home/slam_data/mav0/cam1/data/*"))

fsl = sorted(glob("/home/slam_data/data_sets/kitti/dataset/sequences/03/image_0/*"))
fsr = sorted(glob("/home/slam_data/data_sets/kitti/dataset/sequences/03/image_1/*"))


# pprint([v for v in dir(cv2) if "mode_hh" in v.lower()])
# print(cv2.STEREO_SGBM_MODE_HH4)
# exit()

stereo = cv2.StereoSGBM_create(
    numDisparities=16
    ,P1=50
    ,P2=200
    ,mode=cv2.STEREO_SGBM_MODE_HH4
    ,disp12MaxDiff=0
    ,speckleWindowSize=100
    ,speckleRange=2
    ,uniquenessRatio=15
)

min_i = 1
prev_disparity = None
for fl, fr in zip(fsl[min_i:], fsr[min_i:]):
    imgl = cv2.imread(fl, 0)
    imgr = cv2.imread(fr, 0)

    h, w = imgl.shape[:2]
    imgl = cv2.resize(imgl, (int(h * 1.1), h))
    imgr = cv2.resize(imgr, (int(h * 1.1), h))
    # r = 0.2
    # a, b = int(w * r), int(w * (1.0 - r))
    # imgl = imgl[:, a:b]
    # imgr = imgr[:, a:b]

    disparity = stereo.compute(imgl, imgr)

    # disparity1 = stereo.compute(imgl, imgr)
    # if prev_disparity is not None:
    #     disparity = (prev_disparity + disparity1) // 2
    #     prev_disparity = disparity1
    # else:
    #     disparity = disparity1
    #     prev_disparity = disparity1

    m = np.min(disparity)
    M = np.max(disparity)
    print(m, M)
    disparity = (255 * (disparity - m) / (M - m)).astype(np.uint8)
    # plt.imshow(disparity, 'gray')
    # plt.show()
    # exit()
    # Display the resulting frame
    img = np.concatenate([imgl, imgr, disparity], axis=1)
    r = 1.0
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()