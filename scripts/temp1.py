import pickle
import numpy as np
from gridmap import to_gridmap, DisplayMap
import cv2

with open("/home/slam_data/data_sets/pts.pickle", "rb") as conn:
    pts = pickle.load(conn)

gridmap = to_gridmap(pts)

r = 0.5
img = cv2.resize(gridmap, (int(gridmap.shape[1] * r), int(gridmap.shape[0] * r)))
# Image.fromarray(img).show()
while True:
    cv2.imshow('Frame', gridmap)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# pts = np.reshape(pts, (-1, 7))
# allpts = np.concatenate([pts[:, :3], pts[:, 3:6]])
# l = list(pts[:, 6])
# s = set(l)
#


