import pickle
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

with open("/home/slam_data/data_sets1/temp.pickle", 'rb') as conn:
    kf_ids_from_mps, kf_ids, mp_3dpts, kf_3dpts, kf_ids_from_planes, plane_params = pickle.load(conn)

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

print(np.linalg.norm(plane_params[:,:3], axis=1))

dict_kfs = {i: p for i, p in zip(kf_ids, kf_3dpts)}

pts = []
for i, p in zip(kf_ids_from_mps, mp_3dpts):
    pts.append(list(p[0][:2]) + list(dict_kfs[i][0][:2]))

x1, y1, x2, y2 = zip(*pts)
plt.plot(x1, y1, x2, y2)
# plt.plot(x1, x2, y1, y2)
# plt.scatter(x2, y2, color="yellow", s=100)
# plt.scatter(x2, y2, marker='o')
plt.show()



