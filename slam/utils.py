import numpy as np
import cv2

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
