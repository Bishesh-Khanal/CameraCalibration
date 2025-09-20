import cv2
import numpy as np
import glob

checkerboard_size = (9, 6)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('images/*.png')
print(f"Found {len(images)} images")

gray_shape = None
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not read image {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(
        gray,
        checkerboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)

        cv2.drawChessboardCorners(img, checkerboard_size, corners_refined, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(300)
        print(f"Found corners in {fname}")
    else:
        print(f"No corners found in {fname}")

cv2.destroyAllWindows()

if len(objpoints) > 0:
    print(f"Calibrating using {len(objpoints)} valid images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    print("\nCamera matrix:\n", mtx)
    print("\nDistortion coefficients:\n", dist)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("Mean reprojection error:", mean_error/len(objpoints))

    np.savez("calibration_data.npz", camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
    print("Calibration data saved to calibration_data.npz")

    img = cv2.imread(images[0])
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, new_mtx)
    cv2.imshow('Undistorted', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No valid checkerboard images found! Cannot calibrate.")