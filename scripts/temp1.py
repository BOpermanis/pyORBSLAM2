import pickle
import numpy as np
from pprint import pprint
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
from slam.utils import visualize2d
from collections import defaultdict
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.decomposition import PCA

with open("/home/slam_data/data_sets1/temp.pickle", 'rb') as conn:
    kf_ids_from_mps, kf_ids, mp_3dpts, kf_3dpts, kf_ids_from_planes, plane_params, frame_ids, clouds, plane_segs = pickle.load(conn)

kf_ids = kf_ids.flatten()
frame_ids = frame_ids.flatten()
kf_ids_from_mps = kf_ids_from_mps.flatten()
kf_ids_from_planes = kf_ids_from_planes.flatten()

plane_segs = plane_segs.reshape((-1, 480, 640))

plane_params = plane_params.reshape((-1, 4))

dict_params = defaultdict(list)
for id_kf, param in zip(kf_ids_from_planes, plane_params):
    dict_params[id_kf].append(param)

pca = PCA(n_components=2)

for i, (id_kf, cloud) in enumerate(zip(kf_ids, clouds)):

    params = dict_params[id_kf]
    seg = plane_segs[i, :, :].flatten()

    masks = [seg == j + 1 for j in range(len(params))]

    for mask in masks:
        cloud1 = cloud[mask]
        # print(np.linalg.eigvals(np.cov(cloud1.T)))
        pca.fit(cloud1)
        print(pca.components_)
        cloud1 = pca.transform(cloud1)
        hull = ConvexHull(cloud1)

        plt.plot(cloud1[:, 0], cloud1[:, 1], '.')
        inds = set()
        for simplex in hull.simplices:
            inds.update(simplex)
        #     plt.plot(cloud1[simplex, 0], cloud1[simplex, 1], 'k-')
        # plt.show()
        inds = list(inds)
        cloud1 = cloud1[inds,:]
        print(cloud1.shape)
        cloud1 = np.matmul(cloud1, pca.components_)
        print(cloud1.shape)
        exit()


    exit()