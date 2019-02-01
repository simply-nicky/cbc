from __future__ import print_function

import numpy as np
from scipy.linalg import expm
import math
from timeit import default_timer as timer

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / math.sqrt(np.dot(axis, axis)) * theta))


v = np.array([3, 5, 0])
axis = np.array([4, 4, 1])
theta = 1.2 

start = timer()
print([np.dot(rotation_matrix(axis, theta), v) for _ in range(1000)][0], 'Elapsed time: %f' % (timer() - start), sep='\n')
start = timer()
print([np.dot(M(axis, theta), v) for _ in range(1000)][0], 'Elapsed time: %f' % (timer() - start), sep='\n')