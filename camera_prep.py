#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import glob

def calibrate(target_file_path, col, row):
    # Calibrate the camera
    # initialize object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # the object points are the points in 3D space of the corners
    # z is zero for all points since its on a flat surface and the
    # x,y coords are simply row/col values of the corners starting at
    # 0,0 and going to 6,9 since it is a 6 by 9 grid
    # remember that in Numpy the first coordinate is columns, 2nd is rows
    # like matrix math, not cartesian/graphical coordinates
    # http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
    # target_file_path: target path and file info for calibration images ex: 'camera_cal/ca*.jpg'
    # col: the number of inner corner columns not chessboard columns
    # row: the number of inner corner columns not chessboard columns
    objp = np.zeros((row*col,3), np.float32)
    # use np grid function to assign coordinates to the array
    objp[:,:2] = np.mgrid[0:col, 0:row].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    # glob is a utility for reading in files with similar names
    cal_images = glob.glob(target_file_path)
    images = []
    num_image = 0
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(cal_images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        num_image += 1
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (col,row), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (col,row), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            # plot has a limit on the number of images it can display
            if num_image <= 12:
                images.append(img)
    return images, objpoints, imgpoints

def get_dist_mtx(objpoints, imgpoints, image, save=True, disp=True):
    # Get camera and distortion matrix given object points and image points
    # key return values are the camera matrix(mtx), & the distortion coefficients(dist)
    # objpoints, imgpoints: return values from camera calibration
    # image: image to test distortion correction on
    # save, disp: flags on whether to save and display the image & cal results
    img_size = (image.shape[1], image.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    # undistort the image
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    if save:
        # save the undistored image
        cv2.imwrite('camera_cal/test_undist_calibrate1.jpg',dst)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump( dist_pickle, open( "camera_cal/cam_cal_pickle.p", "wb" ) )
    if disp:
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image', fontsize=30)
    return dst, mtx, dist
