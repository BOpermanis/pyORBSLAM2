import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def to_pt(l):
    # return tuple(map(float, l.split(" - ")[0].replace("(", "").split(",")))
    return l.split(" - ")[0].replace("(", "").split(",")

with open("/home/slam_data/data_sets/pcl_plane_pts.txt", "r") as conn:
    pts1 = conn.readlines()
    pts1 = [tuple(map(float, to_pt(l))) for l in pts1 if l != "\n"]
pts1 = np.asarray(pts1)

coefs1 = np.asarray([0.0413848, 0.936528, 0.348142, -1354.07])


a = coefs1[:3]
ds1 = np.average(np.abs(np.asarray([
    np.dot(x, a) - coefs1[3] for x in pts1
])))

avg = np.average(pts1, axis=0)
ds2 = np.average(np.abs(np.asarray([
    np.dot(x+avg, a) - coefs1[3] for x in pts1
])))

ds3 = np.average(np.abs(np.asarray([
    np.dot(x-avg, a) - coefs1[3] for x in pts1
])))

print("dist of avg", np.dot(avg, a) - coefs1[3])
print("ds1", ds1)
print("ds2", ds2)
print("ds3", ds3)


cov = np.cov(pts1.T)

pprint(np.linalg.eig(cov))