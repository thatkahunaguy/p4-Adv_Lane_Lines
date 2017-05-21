#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from Line import Line

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin, left_lane, right_lane):
    # set a tolerance for the minimum convolution signal at which to create a new x value for a layer
    conv_tolerance = 30

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template
    if left_lane == None:
        # Histogram of bottom half(sum of pixels across x by half) of image to find peaks left and right
        # can use a different ratio - trades off between filtering noise but potentially
        # mislocating start due to lane curvatuere with larger ratio
        l_sum = np.sum(image[int(image.shape[0]/2):,:int(image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(image[int(image.shape[0]/2):,int(image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    else:
        # start with the lane positions from the last frame
        # TODO: should probably start search here vs just setting
        l_center = left_lane.x_start
        r_center = right_lane.x_start

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_max_sig_index = np.argmax(conv_signal[l_min_index:l_max_index])+ l_min_index-offset
        # if the left # of points in window is greate than tolerance, update l_center, else keep the same
        #print("layer: ", level)
        #print("conv_signal", conv_signal)
        #print("left center, min, max, argmax", l_center, l_min_index, l_max_index, l_max_sig_index)
        left_pixels = conv_signal[int(l_max_sig_index)]
        if left_pixels > conv_tolerance:
            l_center = l_max_sig_index
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_max_sig_index = np.argmax(conv_signal[r_min_index:r_max_index])+ r_min_index-offset
        # if the right # of points in window is greate than tolerance, update l_center, else keep the same
        right_pixels = conv_signal[int(r_max_sig_index)]
        if right_pixels > conv_tolerance:
            r_center = r_max_sig_index
        # Add what we found for that layer
        #print("right center, min, max, argmax", r_center, r_min_index, r_max_index, r_max_sig_index)
        #print("left & right conv signals:", left_pixels, right_pixels )
        window_centroids.append((l_center,r_center))
    return window_centroids

def draw_centroids(image, window_centroids, window_width, window_height):
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        # Go through each level and draw the windows
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,image,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,image,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        # TODO: could make this conditional, assuming image is single channel [0,1] pixel value due to s-channel use
        image_255 = np.multiply(image, 255.).astype(np.uint8)
        zero_channel = np.zeros_like(image_255) # create a zero color channel matching top_down
        warpage = np.array(cv2.merge((image_255,zero_channel,zero_channel)),np.uint8) # making the original road pixels red 3 color channels
        # TODO: check way to draw lane - addWeighted seems a failure if you have white pixels as any addition to
        # pixel values in this matrix will simply saturate and stay white which is not the intention
        # separating to different channels doesn't work either as the 2 channels combine for different colors
        # than intended in some areas.
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((image,image,image)),np.uint8)
    return output

def fit_centroids(window_centroids, window_height, image_height, left_lane, right_lane):
    # return left and right fit given window_centroids(leftx, rightx) and centroid window_height
    num_y_values = len(window_centroids)
    y_values = []
    left_x = []
    right_x = []
    for i in range(0, num_y_values):
        y_values.append(image_height - (window_height*i))
        left_x.append(window_centroids[i][0])
        right_x.append(window_centroids[i][1])
    #print("y:", y_values)
    #print("left_x:", left_x)
    #print("right_x:", right_x)
    #left_fit = np.polyfit(y_values, left_x, 2)
    #right_fit = np.polyfit(y_values, right_x, 2)
    if left_lane == None:
        # initialize the lanes
        left_lane = Line(left_x, y_values, 720)
        right_lane = Line(right_x, y_values, 720)
    else:
        # add this fit to the lanes
        left_lane.fit_line(left_x, y_values)
        right_lane.fit_line(right_x, y_values)
    #print("left fit:", left_fit)
    #print("right fit:", right_fit)
    # Fit new polynomials to x,y in world space
    # We can't simply scale curvature from pixel space since the y & x scales are different
    #left_fit_cr = np.polyfit(np.asarray(y_values)*ym_per_pix, np.asarray(left_x)*xm_per_pix, 2)
    #right_fit_cr = np.polyfit(np.asarray(y_values)*ym_per_pix, np.asarray(right_x)*xm_per_pix, 2)
    return left_lane, right_lane

def locate_lanes(top_down, left_lane=None, right_lane=None, plot=True):
    # locate the lane lines
    # top_down is the thresholded top_down image of the lanes
    # window settings
    window_width = 50
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 50 # How much to slide left and right for searching - started with 100
    # Define conversions in x and y from pixels space to meters
    # These are based on assumptions defined in the lesson since we don't have real world measurements
    # We can't simply scale from pixel space since the y & x scales are different
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    window_centroids = find_window_centroids(top_down, window_width, window_height, margin, left_lane, right_lane)
    left_lane, right_lane = fit_centroids(window_centroids, window_height,
                                                                   top_down.shape[0], left_lane, right_lane)
    # Generate x and y values for plotting
    # we fit x as a function of y
    #ploty = np.linspace(0, top_down.shape[0]-1, top_down.shape[0] )
    #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    if plot:
        output = draw_centroids(top_down, window_centroids, window_width, window_height)
        # Display the final results
        plt.imshow(output)
        # plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(left_lane.get_avg_fit(), color='yellow')
        plt.plot(right_lane.get_avg_fit(), color='yellow')
        plt.xlim(0, top_down.shape[1])
        plt.ylim(top_down.shape[0], 0)
        plt.title('window fitting results')
        plt.show()
    return left_lane, right_lane

def get_off_center(lane1x, lane2x, img_width, xm_per_pix=3.7/700):
    # calculate the pixel center of the front of the vehicle (ie bottom of image x coord)
    # xm_per_pix=3.7/700 meters per pixel
    # TODO: is this correct or did we force center with the perspective change src to dst points?
    img_ctr = int(img_width/2)
    pixel_ctr = int((lane2x - lane1x)/2) + lane1x
    if pixel_ctr < img_ctr:
        direc = "left"
    else:
        direc = "right"
    #print("pixel center: ", pixel_ctr, "off center by:", pixel_ctr - top_down.shape[1]/2, "pixels")
    m_ctr = pixel_ctr*xm_per_pix
    off_ctr_string = "{:.2f} m {} of center".format(np.abs(m_ctr - (xm_per_pix * img_ctr)), direc)
    return m_ctr, off_ctr_string

def get_avg_radius(lane1, lane2):
    lane1.set_radius()
    lane2.set_radius()
    avg_radius = (lane1.radius + lane2.radius)/2.
    if avg_radius > 10000:
        rad_string = "Curve Radius: Straight(>10km)"
    else:
        rad_string = "Curve Radius: {:.0f} m".format(avg_radius)
    return avg_radius, rad_string

def inverse_perspective(image, orig_image, Minv_perspective, left_lane, right_lane, text1="", text2=""):
    # Create the perspective transformed image with the lane drawn
    # image: the perspective transformed image
    # orig_image: the original unmodified image
    # left_fitx: x points for left lane
    # right_fitx: x points for right lane
    # ploty: y points for the lanes
    # text: text to add to the image

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.gen_points()]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.gen_points()])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv_perspective, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(orig_image, 1, newwarp, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # note had to add the line type and increase thickness to make the text readable
    cv2.putText(result, text1, (int(result.shape[1]/5), int(result.shape[0]/6) - 40), font, 2, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(result, text2, (int(result.shape[1]/5), int(result.shape[0]/6) + 40), font, 2, (255,255,255), 3, cv2.LINE_AA)
    return result
