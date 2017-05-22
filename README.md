# **Advanced Lane Finding - Project #4** 

### John Glancy

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**Included Files**
1) Project Writeup: [p4_lane_detect_writeup.html](/p4_lane_detect_writeup.html)
2) [p4_lane_detect.ipynb](/p4_lane_detect.ipynb): Jupyter notebook with the project
3) Python Files Imported to Notebook in 2:
[LaneTracker.py](/LaneTracker.py): object to track lane info across frames		
[image_enhance.py](/image_enhance.py): hls, gradient, & thresholding routines
[plot_images.py](/plot_images.py): helper for plotting
[Line.py](/Line.py): object to track information for each lane	
[lanes.py](/lanes.py): routines to locate the lanes	
[camera_prep.py](/camera_prep.py): routines to calibrate the camera
[pipeline.py](/pipeline.py): processing pipeline for images
4) [output images folder](/output_images): example output images(described in detail in notebook 2)
5) [test_image_output](/test_image_output) folder: pipeline output for all test images
6) [project_output_video.mp4](/project_output_video.mp4): output video

**Project Discussion**
While this project identified several new techniques for lane finding, the implementation still is relatively simple and has areas where it is likely to fail or could be improved.
* Sensitivity to image bright/saturated spots: while the thresholding and gradient combinations on the saturation channel are fairly robust creating a binary image where lanes can be identified, there are still conditions where this approach is likely to fail or enounter issues including rain, fog, and roads with image conditions/saturated spots which the thresholding & gradient combinations don't properly filter.  
* Obstruction by other vehicles: the images and video processed in this project have a clear lane in front of the vehicle which is not realistic.  Obstruction by other vehicles in the same lane would likely cause issues in perspective transformation and algorithm for identifying lane pixels from that perspective transform.
* While it wouldn't solve these specific issues a simple region of interest mask similar to the one we used in the initial project is something I'd add with a bit more time.  Additionally, given a bit more time I'd refactor some of the code for more efficency and remove some of the commented debug lines and improve overall commenting to simply clean up the code.