import numpy as np


def _to_homogenous_2D(nparr):
    # homogenous = np.c_[nparr, np.ones(4)]
    homogenous = np.c_[nparr, np.ones(len(nparr))]
    return homogenous

def _from_homogenous_2D(nparr):
    # outarr = np.zeros((4, 2))
    outarr = np.zeros((len(nparr), 2))
    outarr[:, 0] = nparr[:, 0] / nparr[:, 2]
    outarr[:, 1] = nparr[:, 1] / nparr[:, 2]
    return outarr

def transform_cornerpoints_2D(matrix, coordinate_list):
    # turn the coords array into homogeneous coordinates, to do matrix operations on them
    coords_homogeneous = _to_homogenous_2D(coordinate_list)
    # transpose the cornerpoints arrays so that they can be multiplied by matrices
    coords_homogeneous = np.transpose(coords_homogeneous)
    # perform dot
    coords_homogeneous = np.dot(matrix, coords_homogeneous)
    # un-transpose
    coords_homogeneous = np.transpose(coords_homogeneous)
    # convert from homogeneous to 2D
    return _from_homogenous_2D(coords_homogeneous)

def rectangle_containing_quadrilateral(quadrilateral_cornerpoints):
    leftmost = np.min(quadrilateral_cornerpoints[:,0])
    topmost = np.min(quadrilateral_cornerpoints[:,1])
    rightmost = np.max(quadrilateral_cornerpoints[:,0])
    bottommost = np.max(quadrilateral_cornerpoints[:,1])

    box = np.array([[leftmost, topmost], [rightmost, topmost], [rightmost, bottommost], [leftmost, bottommost]])
    return box