import pickle
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from slam.utils import visualizeCountour, average_plane_gridmap, projectPtsOnPlane

with open("/home/slam_data/data_sets/spspalm_map.pickle", 'rb') as conn:
    kf_ids_from_mps, kf_ids, plane_ids, mp_3dpts, kf_3dpts, plane_ids_from_boundary_pts, plane_params, boundary_pts, boundary_updates = pickle.load(conn)

plane_ids = plane_ids[:, 0]
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
print(plane_params.shape, plane_ids)
print(plane_params[:, 3])




print()
# exit()

ii = Counter(plane_ids_from_boundary_pts).most_common(1)[0][0]

xs1 = boundary_pts[plane_ids_from_boundary_pts == ii, :]
updates = boundary_updates[boundary_updates[:, 0] == ii, 1]

xs2 = projectPtsOnPlane(xs1, plane_params[plane_ids == ii][0])

xs3 = PCA(n_components=2).fit_transform(xs2)

average_plane_gridmap(xs3, updates)
