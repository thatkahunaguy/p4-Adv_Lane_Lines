#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def set_perspective(image):
    # add source points found from inspector starting top left corner
    # and going clockwise
    # note these were from manual inspection of straight_lines1.jpg

    src_pts = np.array([(568, 470),
                        (716, 470),
                        (1099, 720),
                        (216, 720)], dtype=np.int32)
    src = np.float32(src_pts)
    offset = 275
    img_size = image.shape
    # note need to form image size to a 2D tuple not 3D with width first
    img_size = image.shape[:-1]
    img_size = img_size[::-1]
    # print(img_size)
    dst = np.float32([[offset, 0], [img_size[0]-offset,0],
                         [img_size[0]-offset, img_size[1]],
                         [offset, img_size[1]]])
    # print("src: ", src)
    # print("dst: ", dst)
    return src, dst, img_size

def perspective_xform(image, src, dst, plot=True):
    # Transform perspective and return the transform and inverse transform matrices

    # src_pts = np.float32([[,],[,],[,],[,]]) source points
    # dst = np.float32([[,],[,],[,],[,]]) destination points
    # returns:
    #   M, the perspective transform matrix
    #   M_inv, the inverse perspective transform matrix (to transform back)
    # get the perspective matrix
    M_perspective = cv2.getPerspectiveTransform(src, dst)
    Minv_perspective = cv2.getPerspectiveTransform(dst, src)
    return M_perspective, Minv_perspective

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient(either x or y direction per orient)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sx = sy = 0
    if orient == 'x':
        sx = 1
    else:
        sy = 1
    sobel = cv2.Sobel(img, cv2.CV_64F, sx, sy, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    #    is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # Apply threshold
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    # Apply threshold
    mag_binary[(scaled_sobel >= mag_thresh[0]) &
               (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of
    # the gradient
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(sobel_x)
    # Apply threshold
    dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return dir_binary

def hls(image):
    # convert from rgb read by mpimg to hsv colorspace
    # h_channel = hsv[:,:,0], l is 1, s is 2 and most useful
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

def threshold(image, min, max):
    # threshold an image setting values <= min & >= maxto 0, between min & max to 1
    thresh_image = np.zeros_like(image)
    thresh_image[(image >= min) & (image <= max)] = 1
    return thresh_image

def hls_threshold(image, channel=2, thresh=(130, 255)):
    # Calculate threshold of an hls channel on an rgb image
    # default is saturation (s-channel) thresholded at 170,255
    # channel:
    #   0: hue
    #   1: light
    #   2: saturation
    #
    # get the hls tranform of the rgb image
    # TODO: refactor as likely faster to hls outside and pass in the
    # desired channel to minimize the number of rgb to hls transforms
    hls_3_channel = hls(image)
    # threshold the image on the requested channel
    hls_image = threshold(hls_3_channel[:,:,channel], thresh[0], thresh[1])
    return hls_image
