import cv2
import numpy as np
import glob

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
obj_points = []
img_points = []
images = glob.glob("/Users/haihui_gao/Documents/my_scripts/cv_test/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                    (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        cv2.waitKey(1)
_, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)

Camera_intrinsic = {"mtx": mtx, "dist": dist, }
# print(Camera_intrinsic)