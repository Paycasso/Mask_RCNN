import numpy as np


class BBox(object):
    '''
    Given a set of cornerpoints (list of 4 2D position vectors), computes a tight bounding box around them
    '''
    def __init__(self, cornerpoints):
        self.left = np.min(cornerpoints[:, 0])
        self.right = np.max(cornerpoints[:, 0])
        self.upper = np.max(cornerpoints[:, 1])
        self.lower = np.min(cornerpoints[:, 1])

        self.width = self.right - self.left
        self.height = self.lower - self.upper

        self.coords = np.array([[self.left, self.upper],
                                [self.right, self.upper],
                                [self.right, self.lower],
                                [self.left, self.lower]])