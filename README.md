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
1) Project Writeup: [P4_lane_detect_writeup.html](/P4_lane_detect_writeup.html)
2) [P4_lane_detect_writeup.ipynb](/P4_lane_detect_writeup.ipynb): Jupyter notebook with the project
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
