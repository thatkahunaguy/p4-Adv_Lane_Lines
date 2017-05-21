from pipeline import pipeline

class LaneTracker(object):
    """
    Track the lane in a series of consecutive frames.
    """

    def __init__(self, mtx, dist, M_perspective, Minv_perspective):
        """
        Initialize a tracker object.

        """
        self.first_frame = True
        self.left_lane = None
        self.right_lane = None
        self.frame = 1
        self.mtx = mtx
        self.dist = dist
        self.M_perspective = M_perspective
        self.Minv_perspective = Minv_perspective

    def process_image(self, image):
        # NOTE: The output you return should be a color image (3 channel)
        # for processing video below
        # TODO: put your pipeline here,
        # you should return the final output (image with lines are drawn on lanes)
        # get image dimensions for slope filter and masking
        if self.first_frame:
            result, self.left_lane, self.right_lane = pipeline(image, self.mtx,
                        self.dist, self.M_perspective, self.Minv_perspective)
            self.first_frame = False
        else:
            result, self.left_lane, self.right_lane = pipeline(image, self.mtx,
                        self.dist, self.M_perspective, self.Minv_perspective,
                        self.left_lane, self.right_lane)
        # plt.imshow(result)
        self.frame += 1
        #print("**FRAME #:", self.frame)
        return result
