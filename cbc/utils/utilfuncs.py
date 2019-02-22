"""
File: utilfuncs.py (Python 2.X and 3.X)

Utility functions for convergent beam diffraction project.
"""
from __future__ import print_function

import os, numpy as np, numba as nb
from math import sqrt, cos, sin, exp
from timeit import default_timer as timer
import matplotlib.pyplot as plt

@nb.njit(nb.complex128[:,:](nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64), fastmath=True)
def phase(ks, xs, ys, zs, wavelength):
    a = ks.shape[0]
    b = xs.size
    res = np.empty((a, b), dtype=np.complex128)
    ks = np.ascontiguousarray(ks)
    xs = np.ascontiguousarray(xs)
    ys = np.ascontiguousarray(ys)
    zs = np.ascontiguousarray(zs)
    for i in range(a):
        for j in range(b):
            _ph = ks[i,0] * xs[j] + ks[i,1] * ys[j] + ks[i,2] * zs[j]
            res[i,j] = np.complex128(cos(2 * np.pi / wavelength * _ph) + sin(2 * np.pi / wavelength * _ph) * 1j)
    return res

@nb.njit(nb.complex128[:,:,:](nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64), fastmath=True)
def phase_conv(kos, kjs, xs, ys, zs, wavelength):
    a = kos.shape[0]
    b = xs.shape[0]
    c = kjs.shape[0]
    res = np.empty((a, b, c), dtype=np.complex128)
    kos = np.ascontiguousarray(kos)
    kjs = np.ascontiguousarray(kjs)
    xs = np.ascontiguousarray(xs)
    ys = np.ascontiguousarray(ys)
    zs = np.ascontiguousarray(zs)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                _ph = (kos[i,0] - kjs[k,0]) * xs[j] + (kos[i,1] - kjs[k,1]) * ys[j] + (kos[i,2] - kjs[k,2]) * zs[j]
                res[i,j,k] = np.complex128(cos(2 * np.pi / wavelength * _ph) + sin(2 * np.pi / wavelength * _ph) * 1j)
    return res

@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]), fastmath=True)
def asf_sum(ss, acoeffs, bcoeffs):
    a = ss.size
    b = acoeffs.size
    asfs = np.empty(a, dtype=np.float64)
    ss = np.ascontiguousarray(ss)
    acoeffs = np.ascontiguousarray(acoeffs)
    bcoeffs = np.ascontiguousarray(bcoeffs)
    for i in range(a):
        dasf = 0.0
        for j in range(b):
            dasf += acoeffs[j] * exp(-ss[i] * ss[i] * bcoeffs[j])
        asfs[i] = dasf
    return asfs

@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:]), fastmath=True)
def q_abs(kout, kin):
    a = kout.shape[0]
    b, c = kin.shape
    qs = np.empty((a, b), dtype=np.float64)
    kout = np.ascontiguousarray(kout)
    kin = np.ascontiguousarray(kin)
    for i in range(a):
        for j in range(b):
            dq = 0.0
            for k in range(c):
                dq += (kout[i,k] - kin[j,k])**2
            qs[i,j] = sqrt(dq)
    return qs

@nb.njit(nb.float64[:,:,:](nb.float64[:,:], nb.float64[:,:], nb.float64[:,:]), fastmath=True)
def q_abs_conv(kout, kis, kjs):
    a = kout.shape[0]
    b = kis.shape[0]
    c, d = kjs.shape
    qs = np.empty((a, b, c), dtype=np.float64)
    kout = np.ascontiguousarray(kout)
    kis = np.ascontiguousarray(kis)
    kjs = np.ascontiguousarray(kjs)
    for i in range(a):
        for j in range(b):
            for k in range(c):
                dq = 0.0
                for l in range(d):
                    dq += (kout[i,l] - kis[j,l] - kjs[k,l])**2
                qs[i,j,k] = sqrt(dq)
    return qs

@nb.njit(nb.types.UniTuple(nb.float64[:], 3)(nb.float64[:,:], nb.float64[:], nb.float64[:], nb.float64[:]), fastmath=True)
def rotate(m, xs, ys, zs):
    a = xs.size
    XS = np.empty(a, dtype=np.float64)
    YS = np.empty(a, dtype=np.float64)
    ZS = np.empty(a, dtype=np.float64)
    m = np.ascontiguousarray(m)
    xs = np.ascontiguousarray(xs)
    ys = np.ascontiguousarray(ys)
    zs = np.ascontiguousarray(zs)
    for i in range(a):
        XS[i] = m[0,0] * xs[i] + m[0,1] * ys[i] + m[0,2] * zs[i]
        YS[i] = m[1,0] * xs[i] + m[1,1] * ys[i] + m[1,2] * zs[i]
        ZS[i] = m[2,0] * xs[i] + m[2,1] * ys[i] + m[2,2] * zs[i]
    return XS, YS, ZS

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
        if event.key == 'up':
            self.next_plot()
        elif event.key == 'down':
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