import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def to_pt(l):
    # return tuple(map(float, l.split(" - ")[0].replace("(", "").split(",")))
    return l.split(" - ")[0].replace("(", "").split(",")

with open("/home/slam_data/data_sets/pcl_plane_pts.txt", "r") as conn:
    pts = conn.readlines()
    pts = [tuple(map(float, to_pt(l))) for l in pts if l!="\n"]
pts = np.asarray(pts)

coefs = np.asarray([0.0338028, 0.930114, 0.365712, -1417.56])

pixels = np.asarray([[422, 220], [310, 478], [638, 459], [638, 240], [476, 220]])
pixels = np.concatenate([pixels, np.ones((pixels.shape[0], 1))], axis=1)

R = np.asarray([
    [5.7592685448804468e+02, 0., 3.1515026356388171e+02],
    [0., 5.7640791601093247e+02, 2.3058580662101753e+02],
    [0., 0., 1.]
])

R1 = np.linalg.pinv(R)
print(R.shape)
print(pixels.shape)
pixels1 = np.matmul(pixels, R1.T)
print(pixels1.shape)

# a = pixels
# plt.scatter(a[:, 0], a[:, 1])
# plt.show()

a = coefs[:3]
thetas = [coefs[3] / np.dot(a, b) for b in pixels1]
pts1 = np.vstack([x * t for x, t in zip(pixels1, thetas)])
# print(pts1.shape)
# print(pts.shape)


ds = np.average(np.abs(np.asarray([
    np.dot(x, a) - coefs[3] for x in pts
])))

ds1 = np.average(np.abs(np.asarray([
    np.dot(x, a) - coefs[3] for x in pts1
])))

print(ds, ds1)

# vals1, vecs1 = np.linalg.eig(np.cov(pts1.T))
# vals, vecs = np.linalg.eig(np.cov(pts.T))



