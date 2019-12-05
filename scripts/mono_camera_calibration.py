import numpy as np
import cv2
import glob
from time import time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(1)


newcameramtx, roi = None, None

state = 0 # 0: vizualizee gridu, nesaglabaa, 1: vizualizee un saglabaa, 2: undistoro


t0 = time()
while True:
    if state == 1:
        t0 = time()

    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if state in (0, 1):
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        # If found, add object points, image points (after refining them)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (8, 6), (-1, -1), criteria)
            if state == 1:
                objpoints.append(objp)
                imgpoints.append(corners2)
                state = 0

            img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        print("len(objpoints)", len(objpoints))

    if len(objpoints) > 10 and state in (0, 1):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[:2], None, None)
        print(mtx)
        # distortion coefs k1, k2, p1, p2, k3
        print(dist)
        exit()

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        print("total error: " + str(mean_error / len(objpoints)))
        state = 2

    if state == 2:
        h, w = img.shape[:2]
        if newcameramtx is None:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # undistort
        img = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]

    cv2.imshow('img',img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

    if time() - t0 > 0.0 and state == 0:
        state = 1


cv2.destroyAllWindows()


