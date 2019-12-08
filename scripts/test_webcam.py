import cv2
from PIL import Image
import numpy as np
cap = cv2.VideoCapture(0)

for i in range(100):
    ret, frame = cap.read()
    cv2.imwrite("/home/slam_data/data_sets/temp/frame{}.jpg".format(i), frame)
    print(i, frame.shape)
# frame = np.zeros((100, 100, 3), dtype=np.uint8)
# Image.fromarray(frame).show()