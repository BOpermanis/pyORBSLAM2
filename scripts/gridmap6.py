import numpy as np
from bresenham import bresenham
import cv2
# from multiprocessing import Process, Queue
# from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
# import matplotlib.pyplot as plt
from collections import Counter
from pprint import pprint
from PIL import Image
from time import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.cluster import AgglomerativeClustering

w_occ = 2.0
thresh_floor = -0.3


def heron_vectorized(a, b, c):
    d_ab = np.linalg.norm(a - b, axis=1)
    d_ac = np.linalg.norm(a - c, axis=1)
    d_bc = np.linalg.norm(c - b, axis=1)
    s = (d_ab + d_ac + d_bc) / 2
    x = s * (s - d_ab) * (s - d_ac) * (s - d_bc)
    return np.where(x < 0.0, 0.0, np.sqrt(x))


def dist2line_vectorized(p, l1, l2):
    S = heron_vectorized(p, l1, l2)
    l = np.linalg.norm(l1 - l2, axis=1)
    return 2 * S / l


def visualize_gridmap(gridmap, r=0.5):
    gridmap = cv2.resize(gridmap, (int(gridmap.shape[1] * r), int(gridmap.shape[0] * r)))

    print("gridmap.shape", gridmap.shape)
    Image.fromarray(gridmap).show()
    return gridmap


def heron(a, b, c):
    d_ab = np.linalg.norm(a - b)
    d_ac = np.linalg.norm(a - c)
    d_bc = np.linalg.norm(c - b)
    s = (d_ab + d_ac + d_bc) / 2
    x = s * (s - d_ab) * (s - d_ac) * (s - d_bc)
    if x < 0.0:
        return 0.0
    return np.sqrt(x)


def dist2line(p, l1, l2):
    S = heron(p, l1, l2)
    l = np.linalg.norm(l1 - l2)
    return 2 * S / l


def to_gridmap2(pts):
    pts = np.reshape(pts, (-1, 7)).astype(np.float32)
    allpts = np.concatenate([pts[:, :3], pts[:, 3:6]])

    w = 32
    n = int(allpts.shape[0] / 2)
    xm, xM = np.min(allpts[:, 0]), np.max(allpts[:, 0])
    ym, yM = np.min(allpts[:, 2]), np.max(allpts[:, 2])
    dx = xM - xm
    dy = yM - ym
    h = int(w * dy / dx)

    allpts[:, 0] = w * (allpts[:, 0] - xm) / dx
    allpts[:, 2] = h * (allpts[:, 2] - ym) / dy
    allpts = allpts.astype(np.int32)

    arr_occ = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
    arr_vis = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
    arr_free = np.zeros(shape=(h + 1, w + 1), dtype=np.float32)

    mps = allpts[:n, :]
    kfs = allpts[n:, :]

    for mp, kf in zip(mps, kfs):
        for x, y in bresenham(mp[0], mp[2], kf[0], kf[2]):
            arr_vis[y, x] += 1
        if mp[1] > thresh_floor:
            arr_occ[mp[2], mp[0]] += w_occ

    d = 2
    arr_vis = cv2.blur(arr_vis, (d, d)) * (d * d)
    arr_occ = cv2.blur(arr_occ, (d, d)) * (d * d)

    mask = arr_vis > 0

    arr_free[mask] = 1 - arr_occ[mask] / arr_vis[mask]
    arr_free[np.logical_not(mask)] = 0.5

    img = np.full(shape=arr_free.shape, fill_value=127, dtype=np.uint8)

    img[arr_free < 0.2] = 0
    img[arr_free > 0.8] = 255

    r = 20
    img = np.flipud(img)
    return cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))


def get_kp_mp_range(mps, kfs):
    allpts = np.concatenate([mps, kfs], axis=0)
    if allpts.shape[1] > 2:
        raise ValueError
    xm, xM = np.min(allpts[:, 0]), np.max(allpts[:, 0])
    ym, yM = np.min(allpts[:, 1]), np.max(allpts[:, 1])
    dx = xM - xm
    dy = yM - ym
    return xm, xM, ym, yM, dx, dy


def pts_float2pixel(mps2d, kfs2d, w, h, xm, xM, ym, yM):
    if mps2d.shape[1] == 3:
        mps2d = mps2d[:, (0, 2)]
        kfs2d = kfs2d[:, (0, 2)]
    n = mps2d.shape[0]
    allpts = np.concatenate([mps2d, kfs2d], axis=0)

    dx = xM - xm
    dy = yM - ym
    allpts[:, 0] = w * (allpts[:, 0] - xm) / dx
    allpts[:, 1] = h * (allpts[:, 1] - ym) / dy
    allpts = allpts.astype(np.int32)

    mps2d = allpts[:n, :]
    kfs2d = allpts[n:, :]
    return mps2d, kfs2d


def make_only_gridmap(mps2d, kfs2d, w, h, xm, xM, ym, yM, tresh_lower=0.2, log=None, flag_return_px_pts=False):
    if mps2d.dtype == np.float32:
        mps2d, kfs2d = pts_float2pixel(mps2d, kfs2d, w, h, xm, xM, ym, yM)

    if True: #arr_occ is None:
        arr_occ = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
        arr_vis = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
        arr_free = np.zeros(shape=(h + 1, w + 1), dtype=np.float32)
        img = np.full(shape=arr_free.shape, fill_value=127, dtype=np.uint8)

    for mp, kf in zip(mps2d, kfs2d):
        for x, y in bresenham(mp[0], mp[1], kf[0], kf[1]):
            arr_vis[y, x] += 1
        if mp[1] > thresh_floor:
            arr_occ[mp[1], mp[0]] += w_occ

    d = 2
    arr_vis = np.max([arr_vis, cv2.blur(arr_vis, (d, d)) * (d * d)], axis=0)
    arr_occ = np.max([arr_occ, cv2.blur(arr_occ, (d, d)) * (d * d)], axis=0)

    mask = arr_vis > 0

    arr_free[mask] = 1 - arr_occ[mask] / arr_vis[mask]
    arr_free[np.logical_not(mask)] = 0.5

    img[arr_free < tresh_lower] = 0
    img[arr_free > 0.8] = 255
    if flag_return_px_pts:
        return img, mps2d, kfs2d
    return img


def make_simple_gridmap(mps2d, kfs2d, w, h, xm, xM, ym, yM, tresh_lower=0.2, log=None, n=None):
    # visualize_gridmap(img, 10)
    d_max = np.sqrt((xM - xm) ** 2 + (yM - ym) ** 2) * 0.05

    cluster = AgglomerativeClustering(n_clusters=None,
                                      affinity='euclidean',
                                      linkage='single',
                                      distance_threshold=d_max)

    cls = cluster.fit_predict(kfs2d)

    cls_uni, cnts = np.unique(cls, return_counts=True)
    if len(cls_uni) > 1:
        avgs = [np.average(kfs2d[cls == cl, :], axis=0) for cl in cls_uni]

        list_cl_condesation = []
        if n is None:
            n = mps2d.shape[0]

        for i0, (cl0, cnt) in enumerate(zip(cls_uni, cnts)):
            if cnt / n < 0.05:
                avg0 = avgs[i0]
                i = np.argsort([np.linalg.norm(avg0 - avg) for avg in avgs])[1]
                list_cl_condesation.append((cl0, cls_uni[i]))

        for cl_small, cl_big in list_cl_condesation:
            cls[cls == cl_small] = cl_big
            cls_uni = np.unique(cls)

    # exit()

    img, mps2d, kfs2d = make_only_gridmap(mps2d, kfs2d, w, h, xm, xM, ym, yM,
                                          tresh_lower=tresh_lower, log=log, flag_return_px_pts=True)

    list_sub_cluster_masks = []
    if len(cls) > 1:
        for c in cls_uni:
            img1 = make_only_gridmap(mps2d[cls == c], kfs2d[cls == c], w, h, xm, xM, ym, yM)
            thresh = (img1 != 127).astype(np.uint8)
            list_sub_cluster_masks.append((c, thresh))
    else:
        thresh = (img != 127).astype(np.uint8)
        list_sub_cluster_masks.append((cls_uni[0], thresh))

    per_cluster_inds = []
    mask_on_wall = img[mps2d[:, 1], mps2d[:, 0]] == 0

    for c, thresh in list_sub_cluster_masks:
        mask_cluster = cls == c
        inds_on_wall = np.where(np.logical_and(mask_cluster, mask_on_wall))[0]
        inds = np.where(mask_cluster)[0]
        per_cluster_inds.append((inds, inds_on_wall, thresh))
        # Image.fromarray(thresh * 255).show()

    return per_cluster_inds


def to_gridmap(pts):
    pts = np.reshape(pts, (-1, 7)).astype(np.float32)
    allpts = np.concatenate([pts[:, :3], pts[:, 3:6]])

    w = 32
    n = int(allpts.shape[0] / 2)
    mps = allpts[:n, :]
    kfs = allpts[n:, :]

    mps_height = mps[:, 1]
    kfs_height = kfs[:, 1]
    mps2d = mps[:, (0, 2)]
    kfs2d = kfs[:, (0, 2)]
    ds = mps2d - kfs2d

    angles = np.angle(ds[:, 0] + 1j * ds[:, 1]) * 180 / np.pi
    # plt.hist(angles)
    # plt.show()
    # exit()

    distance_threshold = 60
    cluster = AgglomerativeClustering(n_clusters=None,
                                      affinity='euclidean',
                                      linkage='complete',
                                      distance_threshold=distance_threshold)

    cls = cluster.fit_predict(np.expand_dims(angles, 1))

    cls_uni = np.unique(cls)

    inds_cl_uni_sort = sorted(cls_uni, key=lambda ii: np.average(angles[cls == ii]))

    i_small, i_big = inds_cl_uni_sort[0], inds_cl_uni_sort[-1]
    dm = distance_matrix(np.expand_dims(angles[cls == i_small], 1),
                         np.expand_dims(angles[cls == i_big] - 360, 1))

    if np.max(dm) <= distance_threshold:
        cls[cls == i_small] = i_big
        cls_uni = np.unique(cls)

    cls_uni, cnts = np.unique(cls, return_counts=True)
    avgs = np.asarray([np.average(angles[cls == cl]) for cl in cls_uni])
    list_cl_condesation = []
    for i0, (cl0, cnt) in enumerate(zip(cls_uni, cnts)):
        if cnt / n < 0.05:
            avg0 = avgs[i0]
            i = np.argsort([np.abs(avg0 - avg) for avg in avgs])[1]
            list_cl_condesation.append((cl0, cls_uni[i]))

    for cl_small, cl_big in list_cl_condesation:
        cls[cls == cl_small] = cl_big
        cls_uni = np.unique(cls)

    xm, xM, ym, yM, dx_whole_map, dy_whole_map = get_kp_mp_range(mps2d, kfs2d)
    h = int(w * dy_whole_map / dx_whole_map)

    clusters = []
    mps2d_pix, kfs2d_pix = pts_float2pixel(mps2d, kfs2d, w, h, xm, xM, ym, yM)
    for i_cl, cl in enumerate(cls_uni):
        inds_cl = np.where(cls == cl)[0]
        kfs2d1 = kfs2d[inds_cl, :]
        mps2d1 = mps2d[inds_cl, :]

        per_cluster_inds = make_simple_gridmap(mps2d1, kfs2d1, w, h, xm, xM, ym, yM, tresh_lower=0.2, n=n)

        for inds, inds_on_wall, mask in per_cluster_inds:
            inds = inds_cl[inds]
            inds_on_wall = inds_cl[inds_on_wall]
            clusters.append([inds, inds_on_wall, mask])

            # kfs2d3 = kfs2d_pix[inds, :]
            # mps2d3 = mps2d_pix[inds, :]
            # mask_debug = make_only_gridmap(mps2d3, kfs2d3, w, h, xm, xM, ym, yM, tresh_lower=0.2)
            # mask_debug = (mask_debug != 127).astype(np.uint8)
            # mm = np.concatenate([mask, mask_debug])
            # Image.fromarray(mm * 255).show()
            # print(np.sum(mask_debug != mask))
            # print(mask_debug.shape, mask.shape)

    # exit()

    # print("len(clusters)", len(clusters))
    # print(mps2d.shape, np.sum([len(v[0]) for v in clusters]))
    # exit()
    # for i, (_, _, mask1) in enumerate(clusters):
    #     for j, (_, _, mask2) in enumerate(clusters):
    #         if i > j:
    #             s = np.sum(mask1 * mask2)
    #             if s > 0:
    #                 a = np.sum(mask1 + mask2 > 0)
    #                 iou = s / a
    #                 print(iou)
    # exit()

    # print(np.sum([len(v[0]) for v in clusters]))
    # print(np.sum([len(v[1]) for v in clusters]))
    # print(cls_uni)
    # print(len(cls))
    # print(mps2d.shape)
    # exit()

    dict_kf2mps = {}
    for i_mp, k in enumerate(kfs2d):
        k = tuple(k)
        if k in dict_kf2mps:
            dict_kf2mps[k].append(i_mp)
        else:
            dict_kf2mps[k] = [i_mp]

    dict_mp2kf = {}
    kfs2d_uni = []
    for i_kf, (kf, mps) in enumerate(dict_kf2mps.items()):
        kfs2d_uni.append(kf)
        for i_mp in mps:
            dict_mp2kf[i_mp] = i_kf

    kfs2d_uni = np.asarray(kfs2d_uni)

    def optimize_cluster(inds1, inds2, log=None):

        ###### CONDITION
        inds1, inds_on_wall1, mask1 = inds1
        inds2, inds_on_wall2, mask2 = inds2

        if not np.any(np.logical_and(mask1, mask2)):
            return mask1, mask2, 0.0

        mask1_erode = cv2.erode(mask1, np.ones((3, 3), dtype=np.uint8))
        mask2_erode = cv2.erode(mask2, np.ones((3, 3), dtype=np.uint8))

        mps2d_pix1 = mps2d_pix[inds_on_wall1]
        mps2d_pix2 = mps2d_pix[inds_on_wall2]

        num_error1 = np.sum(mask2_erode[mps2d_pix1[:, 1], mps2d_pix1[:, 0]])
        num_error2 = np.sum(mask1_erode[mps2d_pix2[:, 1], mps2d_pix2[:, 0]])

        total_error = (num_error1 + num_error2) / (mps2d_pix1.shape[0] + mps2d_pix2.shape[0])

        # print(num_error1)
        # print(num_error2)
        # print()
        #
        # exit()
        if num_error1 / mps2d_pix1.shape[0] < 0.01 and num_error2 / mps2d_pix2.shape[0] < 0.01:
            return mask1, mask2, total_error

        mps2d1 = mps2d[inds_on_wall1]
        mps2d2 = mps2d[inds_on_wall2]

        kfs2d1 = np.stack([kfs2d_uni[dict_mp2kf[i]] for i in inds_on_wall1])
        kfs2d2 = np.stack([kfs2d_uni[dict_mp2kf[i]] for i in inds_on_wall2])

        ds1 = mps2d1 - kfs2d1
        ds2 = mps2d2 - kfs2d2

        d_norms1 = np.expand_dims(np.linalg.norm(ds1, axis=1), 1)
        d_norms2 = np.expand_dims(np.linalg.norm(ds2, axis=1), 1)

        nrms = np.matmul(d_norms1, d_norms2.T)
        dots = np.matmul(ds1, ds2.T)
        dots = np.where(nrms > 0.0, dots / nrms, 0.0)

        inds_overlap = np.where(dots < -0.5)

        if not len(inds_overlap) > 0:
            s = np.sum(mask1 * mask2)
            a = np.sum(mask1 + mask2 > 0)
            return mask1, mask2, total_error
            # print(s / a, np.min(dots), np.max(dots))

        x1 = np.where(mask1 > 0)
        x2 = np.where(mask2 > 0)

        x1 = np.stack([x1[0], x1[1]], axis=1)
        x2 = np.stack([x2[0], x2[1]], axis=1)

        # plt.scatter(x1[:, 0], x1[:, 1], c="red")
        # plt.scatter(x2[:, 0], x2[:, 1], c="green")
        # plt.show()

        ###### GRADIENT
        x = np.concatenate([x1, x2], axis=0)
        y = np.asarray([0]*x1.shape[0] + [1]*x2.shape[0])

        model = lda()
        model.fit(x, y)
        r = np.fliplr(model.coef_)
        if np.sum(r) == 0:
            raise ValueError
        r = r / np.linalg.norm(r)
        # r = np.asarray([[1.0, 0.0]]).astype(np.float32)

        lambda1 = 0.05

        avg1 = np.average(x1, axis=0)
        avg2 = np.average(x2, axis=0)

        d12 = avg2 - avg1
        norm_d12 = np.linalg.norm(d12)

        ##### UPDATE
        switch = (np.dot(r, d12) > 0).astype(np.float32)*2 - 1

        mps2d[inds_on_wall1] += switch * r * lambda1
        mps2d[inds_on_wall2] -= switch * r * lambda1
        kfs2d[inds_on_wall1] += switch * r * lambda1
        kfs2d[inds_on_wall2] -= switch * r * lambda1

        mps2d_pix1, kfs2d_pix1 = pts_float2pixel(mps2d[inds_on_wall1], kfs2d[inds_on_wall1], w, h, xm, xM, ym, yM)
        mps2d_pix2, kfs2d_pix2 = pts_float2pixel(mps2d[inds_on_wall2], kfs2d[inds_on_wall2], w, h, xm, xM, ym, yM)

        mps2d_pix1[:, 0] = np.clip(mps2d_pix1[:, 0], a_min=0, a_max=w)
        mps2d_pix2[:, 0] = np.clip(mps2d_pix2[:, 0], a_min=0, a_max=w)
        kfs2d_pix1[:, 0] = np.clip(kfs2d_pix1[:, 0], a_min=0, a_max=w)
        kfs2d_pix2[:, 0] = np.clip(kfs2d_pix2[:, 0], a_min=0, a_max=w)

        mps2d_pix1[:, 1] = np.clip(mps2d_pix1[:, 1], a_min=0, a_max=h)
        mps2d_pix2[:, 1] = np.clip(mps2d_pix2[:, 1], a_min=0, a_max=h)
        kfs2d_pix1[:, 1] = np.clip(kfs2d_pix1[:, 1], a_min=0, a_max=h)
        kfs2d_pix2[:, 1] = np.clip(kfs2d_pix2[:, 1], a_min=0, a_max=h)

        for i in inds_on_wall1:
            kfs2d_uni[dict_mp2kf[i]] = kfs2d[i]

        for i in inds_on_wall2:
            kfs2d_uni[dict_mp2kf[i]] = kfs2d[i]

        mps2d_pix[inds_on_wall1] = mps2d_pix1
        mps2d_pix[inds_on_wall2] = mps2d_pix2

        mask1 = make_only_gridmap(mps2d_pix1, kfs2d_pix1, w, h, xm, xM, ym, yM)
        mask2 = make_only_gridmap(mps2d_pix2, kfs2d_pix2, w, h, xm, xM, ym, yM)
        mask1 = (mask1 != 127).astype(np.uint8)
        mask2 = (mask2 != 127).astype(np.uint8)

        return mask1, mask2, total_error

    print([len(v[0]) for v in clusters])
    print([len(v[1]) for v in clusters])

    for nr_iter in range(300):
        total_error = 0.0
        for i1, inds1 in enumerate(clusters):
            for i2, inds2 in enumerate(clusters):
                if i1 < i2:
                    log = {}
                    mask1, mask2, error = optimize_cluster(inds1, inds2, log=log)
                    total_error += error
                    clusters[i1][2] = mask1
                    clusters[i2][2] = mask2
        if total_error < 0.0001:
            break
        print("{}) total_error {}".format(nr_iter, total_error))

    kfs2d = np.stack([kfs2d_uni[dict_mp2kf[i]] for i in range(mps2d.shape[0])])

    mps = np.stack([mps2d[:, 0], mps_height, mps2d[:, 1]]).T#.astype(np.int32)
    kfs = np.stack([kfs2d[:, 0], kfs_height, kfs2d[:, 1]]).T#.astype(np.int32)

    allpts = np.concatenate([mps, kfs], axis=0)

    xm, xM = np.min(allpts[:, 0]), np.max(allpts[:, 0])
    ym, yM = np.min(allpts[:, 2]), np.max(allpts[:, 2])

    print("xm, xM = ", xm, xM, ", ym, yM = ", ym, yM)
    dx = xM - xm
    dy = yM - ym

    h = int(w * dy / dx)

    allpts[:, 0] = w * (allpts[:, 0] - xm) / dx
    allpts[:, 2] = h * (allpts[:, 2] - ym) / dy
    allpts = allpts.astype(np.int32)

    mps = allpts[:n, :]
    kfs = allpts[n:, :]

    arr_occ = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
    arr_vis = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
    arr_free = np.zeros(shape=(h + 1, w + 1), dtype=np.float32)

    for mp, kf in zip(mps, kfs):
        for x, y in bresenham(mp[0], mp[2], kf[0], kf[2]):
            arr_vis[y, x] += 1
        if mp[1] > thresh_floor:
            arr_occ[mp[2], mp[0]] += w_occ

    d = 2
    arr_vis = cv2.blur(arr_vis, (d, d)) * (d * d)
    arr_occ = cv2.blur(arr_occ, (d, d)) * (d * d)

    mask = arr_vis > 0

    arr_free[mask] = 1 - arr_occ[mask] / arr_vis[mask]
    arr_free[np.logical_not(mask)] = 0.5

    img = np.full(shape=arr_free.shape, fill_value=127, dtype=np.uint8)

    img[arr_free < 0.2] = 0
    img[arr_free > 0.8] = 255

    r = 20
    img = np.flipud(img)
    return cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))