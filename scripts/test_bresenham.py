import pickle
import numpy as np
import matplotlib.pyplot as plt
from bresenham import bresenham
from PIL import Image
import cv2

with open("/home/slam_data/data_sets/out.pickle", "rb") as conn:
    out = pickle.load(conn)

out = np.reshape(out, (-1, 6))

inds = [0, 2]

allpts = np.concatenate([out[:, :3][:, inds], out[:, 3:][:, inds]])

n = int(allpts.shape[0] / 2)
w = 64

xm, xM = np.min(allpts[:, 0]), np.max(allpts[:, 0])
ym, yM = np.min(allpts[:, 1]), np.max(allpts[:, 1])

dx = xM - xm
dy = yM - ym

h = int(w * dy / dx)

allpts[:, 0] = w * (allpts[:, 0] - xm) / dx
allpts[:, 1] = h * (allpts[:, 1] - ym) / dy
allpts = allpts.astype(np.int32)

# xm, xM = np.min(allpts[:, 0]), np.max(allpts[:, 0])
# ym, yM = np.min(allpts[:, 1]), np.max(allpts[:, 1])
# print(xm, xM)
# print(ym, yM)
# exit()
arr_occ = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
arr_vis = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
arr_free = np.zeros(shape=(h + 1, w + 1), dtype=np.float32)

mps = allpts[:n, :]
kfs = allpts[n:, :]

for mp, kf in zip(mps, kfs):

    for x, y in bresenham(mp[0], mp[1], kf[0], kf[1]):
        arr_vis[y, x] += 1

    # arr_vis[mp[1], mp[0]] -= 1
    arr_occ[mp[1], mp[0]] += 1

d = 2
arr_vis = cv2.blur(arr_vis, (d, d)) * (d * d)
arr_occ = cv2.blur(arr_occ, (d, d)) * (d * d)

mask = arr_vis > 0

arr_free[mask] = 1 - arr_occ[mask] / arr_vis[mask]
arr_free[np.logical_not(mask)] = 0.5

img = np.full(shape=arr_free.shape, fill_value=127, dtype=np.uint8)

img[arr_free < 0.4] = 0
img[arr_free > 0.6] = 255

r = 20
img = cv2.resize(img, (int(img.shape[1] *r), int(img.shape[0] *r)))
Image.fromarray(img).show()

# x = mps
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()