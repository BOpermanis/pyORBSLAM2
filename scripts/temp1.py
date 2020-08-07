import pickle
import numpy as np
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
from slam.utils import visualize2d, to_gridmap
from collections import defaultdict
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class Plane:
    def __init__(self, e, d, margin2d, scaler, pca_comps):
        self.e = e
        self.d = d
        self.margin2d = margin2d
        self.scaler = scaler
        self.pca_comps = pca_comps


with open("/home/slam_data/data_sets1/temp.pickle", 'rb') as conn:
    kf_ids_from_mps, kf_ids, mp_3dpts, kf_3dpts, kf_ids_from_planes, plane_params, frame_ids, clouds, plane_segs = pickle.load(conn)

kf_ids = kf_ids.flatten()
frame_ids = frame_ids.flatten()
kf_ids_from_mps = kf_ids_from_mps.flatten()
kf_ids_from_planes = kf_ids_from_planes.flatten()

plane_segs = plane_segs.reshape((-1, 480, 640))


clouds = clouds.reshape((-1, len(kf_ids), 4)).transpose([1, 0, 2])
# print(clouds.shape)
# # print(clouds[:2, :])
# exit()
clouds = clouds[:, :, :3]
# print(clouds.shape)
# exit()

plane_params = plane_params.reshape((-1, 4))

dict_kfs = {i: pt for i, pt in zip(kf_ids, kf_3dpts)}
dict_params = defaultdict(list)
for id_kf, param in zip(kf_ids_from_planes, plane_params):
    dict_params[id_kf].append(param)

dict_mps = defaultdict(list)
for id_kf, mp in zip(kf_ids_from_mps, mp_3dpts):
    dict_mps[id_kf].append(mp)


pca = PCA(n_components=2)
scaler = StandardScaler()

for i, (id_kf, cloud) in enumerate(zip(kf_ids, clouds)):
    print(11111111)
    mps = np.concatenate(dict_mps[id_kf])
    scaler.fit_transform(mps)
    m0, s0 = scaler.mean_, scaler.scale_

    scaler.fit_transform(cloud)
    m1, s1 = scaler.mean_, scaler.scale_

    d = m1 - m0
    nd = np.linalg.norm(d)
    de = d / nd
    d1 = np.abs(np.dot(s0, de))
    d2 = np.abs(np.dot(s1, de))
    print(d1 < nd, d2 < nd, d1 , nd, d2)
    continue
    params = dict_params[id_kf]
    seg = plane_segs[i, :, :].flatten()
    masks = [seg == j + 1 for j in range(len(params))]
    planes = []
    for mask in masks:
        scaler = StandardScaler()
        cloud1 = cloud[mask]

        cloud1 = scaler.fit_transform(cloud1)
        pca.fit(cloud1)

        cloud1 = pca.transform(cloud1)
        hull = ConvexHull(cloud1)
        e = np.cross(pca.components_[0, :], pca.components_[1, :])
        d = -np.dot(e, scaler.mean_)

        # plt.plot(cloud1[:, 0], cloud1[:, 1], '.')
        inds = set()
        for simplex in hull.simplices:
            inds.update(simplex)
        #     plt.plot(cloud1[simplex, 0], cloud1[simplex, 1], 'k-')
        # plt.show()
        # exit()
        inds = list(inds)
        margin2d = cloud1[inds,:]
        margin3d = np.matmul(margin2d, pca.components_)
        for i in range(3):
            margin3d[:, i] *= scaler.scale_[i]
            margin3d[:, i] += scaler.mean_[i]

        planes.append(Plane(e, d, margin2d, scaler, pca.components_))

    mps = np.concatenate(dict_mps[id_kf])
    kf = kf_3dpts[i, :][0, :]
    for plane in planes:
        eb = np.dot(kf, plane.e)
        d = plane.d
        # d = -np.dot(kf, plane.e)
        ea = np.matmul(mps, plane.e)
        lambdas = (d - eb) / (ea - eb)
        mask = np.logical_and(0 <= lambdas, lambdas < 1)
        if np.sum(mask) > 0:
            print(np.average(mask))
            # TODO izmanto convex hull
            kf1 = np.stack([kf] * np.sum(mask))
            mps[mask, :] = kf1 * lambdas[mask] + mps[mask, :] * (1 - lambdas[mask])
        dict_mps[id_kf] = mps

    # exit()
kfs, mps = [], []
for id_kf, mps1 in dict_mps.items():
    for mp in mps1:
        mps.append(mp)
        kfs.append(kf_3dpts[id_kf])

mps = np.stack(mps)
kfs = np.concatenate(kfs)

print(mps.shape, kfs.shape)
img = to_gridmap(kfs, mps)
plt.imshow(img)
plt.show()