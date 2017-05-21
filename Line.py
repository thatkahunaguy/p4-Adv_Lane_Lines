
import numpy as np
from collections import deque

# define a line class to hold important parameters
class Line():
    def __init__(self, x, y, h):
        # x,y: x & y points to fit for the line
        # h: height of the frame for generating the line
        self.h = h
        # was the line detected in the last iteration? Needed if x_start??
        self.detected = False
        # x_value of lane at bottom of frame if detected in prior frame
        self.x_start = None
        #average 2nd order fit coefficients over the last 5 iterations
        self.coeff = deque(maxlen=5)
        #radius of curvature of the line in meters
        self.radius = None
        # string describing radius of curvature
        self.rad_string = None
        self.fit_line(x, y)

    def fit_line(self, x, y):
        #print("fitting line to:")
        #print("x:", x)
        #print("y:", y)
        #print("fit_line_coeff:",np.polyfit(y, x, 2))
        self.coeff.append(np.polyfit(y, x, 2))

    def gen_points(self):
        ploty = np.linspace(0, self.h-1, self.h )
        # use the avg fit over the last 5 coefficients
        avg_fit = self.get_avg_fit()
        #print("avg_fit:", avg_fit)
        # update the x starting position for the line
        # TODO: if I'm doing line checks to see if there's a problem does this location for update
        # make sense? probably not and it needs to be sanity checkd to ensure is is still in the image
        x = avg_fit[0]*ploty**2 + avg_fit[1]*ploty + avg_fit[2]
        self.x_start  = x[self.h-1]
        #print("x_start:", self.x_start)
        return x, ploty

    def get_avg_fit(self):
        #print("getting avg fit from:", self.coeff)
        return np.mean(np.array(self.coeff), axis=0)

    def set_radius(self):
        ''' return the curve radius given y values and left & right
            fit in real world units
            ploty: y values (meters)
            left_fit_cr & right_fit_cr: left/right curve radius fits(2nd order) in meters'''
        # Define conversions in x and y from pixels space to meters
        # These are based on assumptions defined in the lesson since we don't have real world measurements
        # We can't simply scale from pixel space since the y & x scales are different
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        x, y = self.gen_points()
        # might need to use np.asarray as I did in the code or change the return from gen_points
        fit_in_m = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        # np.polyfit(np.asarray(y_values)*ym_per_pix, np.asarray(left_x)*xm_per_pix, 2)
        #print(left_curve_rad, right_curve_rad)
        self.radius = ((1 + (2*fit_in_m[0]*self.h*ym_per_pix + fit_in_m[1])**2)**1.5) / np.absolute(2*fit_in_m[0])
        # set the radius string straight if >2000m radius, else to the radius
        if self.radius > 10000:
            self.rad_string = "Curve Radius: Straight(>10km)"
        else:
            self.rad_string = "Curve Radius: {:.0f} m".format(self.radius)
