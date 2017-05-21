#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from image_enhance import hls_threshold, hls, mag_thresh, dir_threshold, abs_sobel_thresh
from lanes import locate_lanes, inverse_perspective, get_avg_radius, get_off_center

def pipeline(img, mtx, dist, M_perspective, Minv_perspective, left_lane=None, right_lane=None):
    # undistort the image with the calibration data
    # TODO: may need to read this from the pickle file in final program
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    #images = []
    undist_image = cv2.undistort(img, mtx, dist, None, mtx)
    #images.append(image)
    # calculate the thresholded s-channel
    s_binary = hls_threshold(undist_image, 2, (150, 255))
    #images.append(s_binary)
    # use the s channel of the image as the primary input to gradients
    image = hls(undist_image)[:,:,2]
    #images.append(image)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(20, 100))
    # 0.5 1.1
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    gradient_sx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    gradient_sy = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    combined_gradient = np.zeros_like(mag_binary)
    combined_gradient[(gradient_sx == 1) & (gradient_sy == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # combine the thresholded s-channel and the combined gradients
    combined = np.zeros_like(s_binary)
    combined[(combined_gradient == 1) | (s_binary == 1)] = 1
    #images.append(combined)
    # perspective transform the image
    top_down = cv2.warpPerspective(combined, M_perspective, dsize=image.shape[::-1], flags=cv2.INTER_LINEAR)
    #images.append(top_down)
    left_lane, right_lane = locate_lanes(top_down, left_lane, right_lane, plot = False)
    curve_rad, rad_string = get_avg_radius(left_lane, right_lane)
    off_ctr, off_ctr_string = get_off_center(left_lane.x_start, right_lane.x_start, top_down.shape[1])
    result = inverse_perspective(top_down, undist_image, Minv_perspective, left_lane, right_lane, rad_string, off_ctr_string)
    #print("result shape:", result.shape)
    #plot_images(images, gray=True)
    return result, left_lane, right_lane
