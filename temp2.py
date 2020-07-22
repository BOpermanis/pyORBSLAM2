import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from pprint import pprint

print(cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# exit()
a = pd.read_csv("/home/slam_data/data_sets/seg_output.csv", header=None)
a = np.asarray(a)

a = a[:, :-1]
print(a.shape)
m, M = np.min(a), np.max(a)
a = 255 * (a - m) / (M - m)
a = a.astype(np.uint8)
# a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

d = 10
ker = np.ones((d, d))
a = cv2.dilate(a, ker, 1)
a = cv2.erode(a, ker, 1)

ret, thresh = cv2.threshold(a, 127, 255, 0)
img, contours, hierarchy = cv2.findContours(thresh, -1, cv2.CHAIN_APPROX_SIMPLE)

contour = contours[np.argmax([len(c) for c in contours])]

epsilon = 0.1*cv2.arcLength(contour,True)
approx = cv2.approxPolyDP(contour,epsilon,True)
# print(contour.shape)
# print(approx.shape)
# exit()
# for c, h in zip(contours, hierarchy[0, :, :]):
#
#     print(h)
#     print(c.shape)
a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
cv2.drawContours(a, [contour], -1, (0,255,0), 3)
cv2.drawContours(a, [approx], -1, (255,0,0), 3)
print(len(contours))
plt.imshow(a)
plt.show()
