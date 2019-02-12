"""
File: utilfuncs.py (Python 2.X and 3.X)

Utility functions for convergent beam diffraction project.
"""
from __future__ import print_function

import os, numpy as np, numba as nb
from math import sqrt, cos, sin
from timeit import default_timer as timer
import matplotlib.pyplot as plt

@nb.njit(nb.complex128[:,:](nb.complex128[:,:], nb.complex128[:,:]), fastmath=True)
def couterdot(A, B):
    a = A.shape[0]
    b, c = B.shape
    C = np.empty((a, b), dtype=np.complex128)
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    for i in range(a):
        for j in range(b):
            dC = np.complex128(0.0)
            for k in range(c):
                dC += A[i,k] * B[j,k]
            C[i,j] = dC
    return C

@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:]), fastmath=True)
def outerdot(A, B):
    a = A.shape[0]
    b, c = B.shape
    C = np.empty((a, b), dtype=np.float64)
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    for i in range(a):
        for j in range(b):
            dC = 0.0
            for k in range(c):
                dC += A[i,k] * B[j,k]
            C[i,j] = dC
    return C

def make_filename(path, filename, i=2):
    name, ext = os.path.splitext(filename)
    newname = name + "_{:d}".format(i) + ext
    if not os.path.isfile(os.path.join(path, filename)):
        return filename    
    elif os.path.isfile(os.path.join(path, newname)):
        return make_filename(path, filename, i + 1)
    else:
        return newname

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    axis - rotation axis
    theta - rotation angle
    """
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def verbose_call(v, func, *args):
    """
    Call function func with given arguments args verbose if v is True and silent if V is False.

    v - verbosity flag
    func - a function to call
    args - tuple of arguments for func

    Returns results of function func.
    """
    if v:
        print('Parsing argument(s):', *args, sep='\n')
        start = timer()
        res = func(*args)
        print('%s is done\nEstimated time: %f' % (func.__name__, (timer() - start)))
        return res
    else:
        return func(*args)

def search_rec(path, ext='hdf5', filelist=None):
    """
    Search recursively in sub folders of given path for files with extension ext.

    path - a path to search
    ext - file extension to search

    Returns list of paths.
    """
    if not os.path.isdir(path): 
        raise  ValueError("the path is invalid")
    if filelist == None:
        filelist = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(ext):
                filelist.append(os.path.join(root, f))
    return filelist

class AxesSeq(object):
    """
    Creates a series of axes in a figure where only one is displayed at any given time. Which plot is displayed is controlled by the arrow keys.
    """
    def __init__(self, size):
        self.fig = plt.figure()
        self.axes = [self.fig.add_subplot(1,1,1, label=i, visible=False) for i in range(size)]
        self.index = 0
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def __iter__(self):
        return iter(self.axes)

    def on_keypress(self, event):
        if event.key == 'right':
            self.next_plot()
        elif event.key == 'left':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self.index < len(self.axes) - 1:
            self.axes[self.index].set_visible(False)
            self.axes[self.index+1].set_visible(True)
            self.index += 1

    def prev_plot(self):
        if self.index > 0:
            self.axes[self.index].set_visible(False)
            self.axes[self.index-1].set_visible(True)
            self.index -= 1

    def show(self):
        self.axes[self.index].set_visible(True)
        plt.show()