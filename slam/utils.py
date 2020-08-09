import cv2
from bresenham import bresenham
# from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
# from multiprocessing import Process, Queue
from threading import Thread as Process
from queue import Queue
from collections import Counter

w_occ = 2.0
thresh_floor = -0.3


def average_plane_gridmap(xs, updates):

    mx, Mx = np.min(xs[:, 0]), np.max(xs[:, 0])
    my, My = np.min(xs[:, 1]), np.max(xs[:, 1])

    width = 500
    height = 500

    xs[:, 0] = width * (xs[:, 0] - mx) / (Mx - mx)
    xs[:, 1] = width * (xs[:, 1] - my) / (My - my)

    a = 0
    imgs = []
    for b in np.cumsum(updates):
        xs1 = xs[a:b, :]
        # print(xs1.shape, a, b)
        xs1 = np.concatenate([xs1, xs1[:1, :]]).astype(int)
        img = np.zeros(shape=(height, width))
        cv2.drawContours(img, [np.expand_dims(xs1, 1)], -1, 1, -1)
        imgs.append(img)
        a = b

    img = np.sum(np.stack(imgs), axis=0) > 3
    return (img * 255).astype(np.uint8)
    # plt.imshow(img)
    # plt.show()


def projectPtsOnPlane(xs1, coef):
    # finding vector on a plane
    v = - coef[3] / np.sum(np.square(coef[:3]))
    vec = v * coef[:3]

    # compute dists from point to plane
    vecs = np.stack([vec] * xs1.shape[0])
    dists = np.apply_along_axis(lambda x: np.dot(x, coef[:3]), 1, vecs - xs1)

    # using direction of plane normal find projection
    ns = np.stack([coef[:3]] * xs1.shape[0])
    return xs1 + (ns.T * dists).T


def worker_grid(queue_pts, queue):
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    while True:

        # print(111111111111111111)
        plane_ids, plane_params, boundary_pts, plane_ids_from_boundary_pts, boundary_updates = queue_pts.get()
        # print(222222222222222222)
        if boundary_updates.shape[0] > 10:
            ii = set(plane_ids[np.abs(np.dot(plane_params[:, :3], np.asarray([0.0, 1.0, 0.0]))) > 0.8])
            ii = Counter([a for a in plane_ids_from_boundary_pts if a in ii]).most_common(1)[0][0]
            xs = boundary_pts[plane_ids_from_boundary_pts == ii, :]
            updates = boundary_updates[boundary_updates[:, 0] == ii, 1]
            coef = plane_params[plane_ids == ii][0]
            xs = projectPtsOnPlane(xs, coef)
            # xs = pca.fit_transform(xs).astype(int)
            xs = xs[:, (0, 2)]
            gridmap = average_plane_gridmap(xs, updates)
            queue.put(gridmap)



def worker_display(queue):
    i = 0
    while True:
        i += 1
        h, w = 500, 500
        gridmap = queue.get()
        gridmap = np.stack([gridmap] * 3, axis=2)
        h1, w1 = gridmap.shape[:2]
        r = h1 / w1
        r = np.clip(r, a_min=0.33, a_max=3.0)
        h1 = h
        w1 = int(h1 / r)
        gridmap = cv2.resize(gridmap, (w1, h1))

        cv2.imshow('gridmap', gridmap)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


class DisplayMap:
    def __init__(self, w=None, h=None, max_side=None):
        self.w = w
        self.h = h
        self.max_side = max_side
        self.queue = Queue(maxsize=20)
        self.queue_pts = Queue(maxsize=20)

        threads = [
            Process(target=worker_display, args=(self.queue, )),
            Process(target=worker_grid, args=(self.queue_pts, self.queue))
        ]

        for t in threads:
            t.daemon = True
            t.start()

    def add_data(self, slam):
        slam.prepare_dump()
        plane_ids = slam.get_plane_ids()
        plane_ids_from_boundary_pts = slam.get_plane_ids_from_boundary_pts()
        plane_params = slam.get_plane_params()
        boundary_pts = slam.get_boundary_pts()
        boundary_updates = slam.get_boundary_update_sizes()

        plane_ids = plane_ids[:, 0]
        plane_params = plane_params.reshape((-1, 4))
        boundary_pts = boundary_pts[:, 0, :]
        plane_ids_from_boundary_pts = plane_ids_from_boundary_pts[:, 0]
        boundary_updates = boundary_updates[:, 0, :]

        self.queue_pts.put((
            plane_ids, plane_params, boundary_pts, plane_ids_from_boundary_pts, boundary_updates
        ))


def convert_depth_frame_to_pointcloud(depth_image, intr, flag_drop_zero_depth=True):
    height, width = depth_image.shape
    u, v = np.meshgrid(
        np.linspace(0, width - 1, width, dtype=np.int16),
        np.linspace(0, height - 1, height, dtype=np.int16))
    u = u.flatten()
    v = v.flatten()

    x = (u - intr.ppx) / intr.fx
    y = (v - intr.ppy) / intr.fy
    z = depth_image.flatten() / 1000
    x = np.multiply(x, z)
    y = np.multiply(y, z)
    points3d_all = np.stack([x, y, z], axis=1)

    if flag_drop_zero_depth:
        mask = np.nonzero(z)
        return points3d_all[mask, :]
    return points3d_all


def visualize2d(x1s, x2s, y1s, y2s):
    import matplotlib.pyplot as plt
    x = list(x1s) + list(x2s)
    y = list(y1s) + list(y2s)

    mx, Mx = np.min(x), np.max(x)
    my, My = np.min(y), np.max(y)

    width = 500
    height = 500
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    for x1, y1, x2, y2 in zip(x1s, x2s, y1s, y2s):
        x1, x2 = map(lambda x: int(width * (x - mx) / (Mx - mx)), (x1, x2))
        y1, y2 = map(lambda y: int(height * (y - my) / (My - my)), (y1, y2))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(img)
    plt.show()


def visualizeCountour(xs, ys):
    import matplotlib.pyplot as plt
    mx, Mx = np.min(xs), np.max(xs)
    my, My = np.min(ys), np.max(ys)

    width = 500
    height = 500
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    for i in range(xs.shape[0]):
        a = i
        b = 0 if i + 1 == xs.shape[0] else i + 1
        x1, x2 = xs[a], xs[b]
        y1, y2 = ys[a], ys[b]
        x1, x2 = map(lambda x: int(width * (x - mx) / (Mx - mx)), (x1, x2))
        y1, y2 = map(lambda y: int(height * (y - my) / (My - my)), (y1, y2))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    plt.imshow(img)
    plt.show()


def to_gridmap(kfs, mps):
    allpts = np.concatenate([mps, kfs])

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
