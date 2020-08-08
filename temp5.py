import pickle
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from slam.utils import visualizeCountour, average_plane_gridmap

with open("/home/slam_data/data_sets/spspalm_map.pickle", 'rb') as conn:
    kf_ids_from_mps, kf_ids, plane_ids, mp_3dpts, kf_3dpts, plane_ids_from_boundary_pts, plane_params, boundary_pts, boundary_updates = pickle.load(conn)

plane_params = plane_params.reshape((-1, 4))
boundary_pts = boundary_pts[:, 0, :]
plane_ids_from_boundary_pts = plane_ids_from_boundary_pts[:, 0]
boundary_updates = boundary_updates[:, 0, :]
# a1 = -1
# for a in plane_ids_from_boundary_pts:
#     if a !=a1:
#         print(a)
#     a1 = a
#
print(Counter(plane_ids_from_boundary_pts))
# exit()
for ii in [2]: #set(plane_ids_from_boundary_pts):
    xs1 = boundary_pts[plane_ids_from_boundary_pts == ii, :]
    updates = boundary_updates[boundary_updates[:, 0] == ii, 1]
    # print("xs.shape", xs1.shape, np.sum(updates))
    # print("len(updates)", len(updates))
    # print("np.sum(updates)", np.sum(updates))
    xs = xs1[:updates[0], :]

    # ys = xs[:, ]
    # cov = np.cov(xs.T)
    # a = np.linalg.eig(cov)
    # print(a[0].shape, a[1].shape)
    # print(a[0])
    # ys = PCA(n_components=2).fit_transform(xs)
    # print(ys.shape)

    # visualizeCountour(xs[:, 0], xs[:, 1])
    average_plane_gridmap(xs1[:, (0, 2)], updates)

    exit()
