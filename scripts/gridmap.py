import numpy as np
from bresenham import bresenham
import cv2
from threading import Thread
from queue import Queue

def to_gridmap(pts):
    pts = np.reshape(pts, (-1, 6))
    inds = [0, 2]
    allpts = np.concatenate([pts[:, :3][:, inds], pts[:, 3:][:, inds]])
    n = int(allpts.shape[0] / 2)
    w = 64
    xm, xM = np.min(allpts[:, 0]), np.max(allpts[:, 0])
    ym, yM = np.min(allpts[:, 1]), np.max(allpts[:, 1])
    dx = xM - xm
    dy = yM - ym
    h = int(w * dy / dx)
    allpts[:, 0] = w * (allpts[:, 0] - xm) / dx
    allpts[:, 1] = h * (allpts[:, 1] - ym) / dy
    allpts = allpts.astype(np.int32)
    arr_occ = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
    arr_vis = np.zeros(shape=(h + 1, w + 1), dtype=np.int32)
    arr_free = np.zeros(shape=(h + 1, w + 1), dtype=np.float32)
    mps = allpts[:n, :]
    kfs = allpts[n:, :]

    for mp, kf in zip(mps, kfs):
        for x, y in bresenham(mp[0], mp[1], kf[0], kf[1]):
            arr_vis[y, x] += 1

    # arr_vis[mp[1], mp[0]] -= 1
    arr_occ[mp[1], mp[0]] += 1
    d = 2
    arr_vis = cv2.blur(arr_vis, (d, d)) * (d * d)
    arr_occ = cv2.blur(arr_occ, (d, d)) * (d * d)

    mask = arr_vis > 0

    arr_free[mask] = 1 - arr_occ[mask] / arr_vis[mask]
    arr_free[np.logical_not(mask)] = 0.5

    img = np.full(shape=arr_free.shape, fill_value=127, dtype=np.uint8)

    img[arr_free < 0.4] = 0
    img[arr_free > 0.6] = 255

    r = 20
    return cv2.resize(img, (int(img.shape[1] *r), int(img.shape[0] *r)))


class DisplayMap:
    def __init__(self, w=None, h=None, max_side=None):
        self.w = w
        self.h = h
        self.max_side = max_side
        self.queue = Queue()
        thread = Thread(target=self._worker_display)
        thread.setDaemon(True)
        thread.start()

    def _worker_display(self):
        gridmap = np.zeros((100, 100, 3), dtype=np.uint8)
        
        while True:
            if self.queue.qsize() > 0:
                gridmap = self.queue.get()
                self.queue.task_done()
            if self.w is not None:
                gridmap = cv2.resize(gridmap, (self.w, self.h))
            if self.max_side is not None:
                if np.prod(gridmap.shape) > 0:
                    h, w = gridmap.shape[:2]
                    r = w / h
                    if h > w:
                        h = self.max_side
                        w = int(h * r)
                    else:
                        w = self.max_side
                        h = int(w / r)
                    gridmap = cv2.resize(gridmap, (w, h))
            cv2.imshow('gridmap', gridmap)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def new_map(self, gridmap):
        self.queue.put(gridmap)

if __name__ == "__main__":
    # from PIL import Image
    import pickle
    import matplotlib.pyplot as plt
    from time import sleep

    with open("/home/slam_data/data_sets/out.pickle", "rb") as conn:
        pts = pickle.load(conn)

    img = to_gridmap(pts)
    # plt.imshow(img)

    model = DisplayMap()
    model.new_map(img)
    sleep(10)

    # while True:
    #     cv2.imshow('Frame', img)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         break
    # Image.fromarray(img).show()
    