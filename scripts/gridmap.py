import numpy as np
from bresenham import bresenham
import cv2
from multiprocessing import Process, Queue
# import matplotlib.pyplot as plt


def to_gridmap(pts, w_occ=2.0, thresh_floor=-0.3):
    pts = np.reshape(pts, (-1, 6))
    allpts = np.concatenate([pts[:, :3], pts[:, 3:]])

    # x = pts[:, 3:][:, 1].reshape(-1, 1)
    # # x = pts[:, :3][:, 1].reshape(-1, 1)
    # plt.hist(x)
    # plt.show()
    # exit()

    w = 64
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
    return cv2.resize(img, (int(img.shape[1] *r), int(img.shape[0] *r)))


def worker_grid(queue_pts, queue):
    while True:
        pts = queue_pts.get()
        gridmap = to_gridmap(pts)
        queue.put(gridmap)
        

def worker_display(queue, queue_frames):
    gridmap = np.zeros((100, 100, 3), dtype=np.uint8)
    i = 0
    while True:
        i += 1
        frame, kps, is_tracking_ok = queue_frames.get()
        if kps is not None:
            for kp in kps[:, 0, :].astype(np.int32):
                if is_tracking_ok:
                    cv2.circle(frame, tuple(kp), 4, (0, 255, 0), 5)
                else:
                    cv2.circle(frame, tuple(kp), 2, (0, 255, 0), 1)

        h, w = frame.shape[:2]

        if queue.qsize() > 0:
            gridmap = queue.get()
            gridmap = np.stack([gridmap]*3, axis=2)
            h1, w1 = gridmap.shape[:2]
            r = h1 / w1
            r = np.clip(r, a_min=0.33, a_max=3.0)
            h1 = h
            w1 = int(h1 / r)
            gridmap = cv2.resize(gridmap, (w1, h1))

        if np.any(gridmap.shape[:2] != frame.shape[:2]):
            gridmap = cv2.resize(gridmap, (frame.shape[1], frame.shape[0]))

        frame1 = np.concatenate([frame, gridmap], axis=1)
        cv2.imshow('gridmap', frame1)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


class DisplayMap:
    def __init__(self, w=None, h=None, max_side=None):
        self.w = w
        self.h = h
        self.max_side = max_side
        self.queue = Queue(maxsize=20)
        self.queue_frames = Queue(maxsize=20)
        self.queue_pts = Queue(maxsize=20)
        
        threads = [
            Process(target=worker_display, args=(self.queue, self.queue_frames)),
            Process(target=worker_grid, args=(self.queue_pts, self.queue))
        ]
        for t in threads:
            t.daemon = True
            t.start()
    
    def new_frame(self, frame, kps, is_tracking_ok):
        self.queue_frames.put((frame, kps, is_tracking_ok))
    
    def new_map(self, pts):
        self.queue_pts.put(pts)


if __name__ == "__main__":
    from PIL import Image
    import pickle
    import matplotlib.pyplot as plt
    from time import sleep

    with open("/home/slam_data/data_sets/out.pickle", "rb") as conn:
        pts = pickle.load(conn)

    img = to_gridmap(pts, w_occ=2.0, thresh_floor=-0.1)
    # plt.imshow(img)

    model = DisplayMap()
    i = 0
    while True:
        i += 1
        model.new_frame(np.zeros((480, 640, 3), dtype=np.uint8), None, True)
        if i % 50 == 0:
            model.new_map(pts)

    exit()
    r = 0.5
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    # Image.fromarray(img).show()
    while True:
        cv2.imshow('Frame', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Image.fromarray(img).show()
    