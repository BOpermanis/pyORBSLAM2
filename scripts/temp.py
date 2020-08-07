import pickle
import numpy as np
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
from slam.utils import visualize2d

with open("/home/slam_data/data_sets1/temp.pickle", 'rb') as conn:
    kf_ids_from_mps, kf_ids, mp_3dpts, kf_3dpts, kf_ids_from_planes, plane_params, frame_ids, clouds = pickle.load(conn)


# print(kf_ids)
# print(frame_ids)
# exit()

kf_ids_from_mps = kf_ids_from_mps[:, 0]
kf_ids = kf_ids[:, 0]

s = set(kf_ids)
s1 = set(kf_ids_from_mps)

print(kf_ids_from_mps[:4])
print(kf_ids[:4])
print(len(kf_ids_from_mps), len(set(kf_ids_from_mps)))
print(len(kf_ids), len(set(kf_ids)))
print(kf_ids.dtype, kf_ids_from_mps.dtype)
print(mp_3dpts.dtype, kf_3dpts.dtype)

print(len(s.intersection(s1)))

# print(plane_params.shape)

plane_params = plane_params.reshape((-1, 4))
print(plane_params.shape)

# pprint(plane_params)

# scalars = []
# for v1, v2 in combinations(plane_params[:,:3], 2):
#     scalars.append(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
#
# plt.hist(scalars)
# plt.show()
# exit()
i1 = 0
i2 = 2
dict_kfs = {i: [p[0][i1], p[0][i2]] for i, p in zip(kf_ids, kf_3dpts)}

pts = []
for i, p in zip(kf_ids_from_mps, mp_3dpts):
    pts.append([p[0][i1], p[0][i2]] + dict_kfs[i])

x1, y1, x2, y2 = zip(*pts)

visualize2d(x1, y1, x2, y2)