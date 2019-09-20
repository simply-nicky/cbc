"""
File: utilfuncs.py (Python 2.X and 3.X)

Utility functions for convergent beam diffraction project.
Dependencies: scipy, numpy and numba.
"""
from __future__ import print_function

import os
import ctypes
import datetime
import errno
import logging
from math import sqrt, cos, sin, exp, pi
from multiprocessing import cpu_count
import numpy as np
import numba as nb
from numba.extending import get_cython_function_address
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy import LowLevelCallable, constants
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

CORES_NUM = cpu_count()

ADDR = get_cython_function_address("scipy.special.cython_special", "j0")
FUNC_TYPE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
J0_C = FUNC_TYPE(ADDR)

PAR_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
RES_PATH = 'results'
LOG_PATH = 'logs'

@nb.vectorize('float64(float64)')
def j0_vec(x):
    return J0_C(x)

@nb.njit
def j0(x):
    return j0_vec(x)

def jit_integrand(func):
    jit_func = nb.njit(func)
    @nb.cfunc(nb.float64(nb.intc, nb.types.CPointer(nb.float64)))
    def wrapper(n, args):
        return jit_func(args[0], args[1], args[2], args[3], args[4])
    return LowLevelCallable(wrapper.ctypes)

def quad_complex(func_re, func_im, a, b, **args):
    re = si.quad(func_re, a, b, **args)[0]
    im = si.quad(func_im, a, b, **args)[0]
    return re + 1j * im

@nb.njit(nb.complex128[:, :](nb.float64[:, :, :],
                             nb.float64[:, :],
                             nb.float64[:, :],
                             nb.float64[:, :],
                             nb.float64), fastmath=True)
def phase_inc(kins, xs, ys, zs, wavelength):
    a, b = xs.shape
    res = np.empty((a, b), dtype=np.complex128)
    kins = np.ascontiguousarray(kins)
    xs = np.ascontiguousarray(xs)
    ys = np.ascontiguousarray(ys)
    zs = np.ascontiguousarray(zs)
    for i in range(a):
        for j in range(b):
            _ph = kins[i, j, 0] * xs[i, j] + kins[i, j, 1] * ys[i, j] + kins[i, j, 2] * zs[i, j]
            res[i, j] = cos(2 * pi / wavelength * _ph) - sin(2 * pi / wavelength * _ph) * 1j
    return res

@nb.njit(nb.complex128[:](nb.float64[:, :],
                          nb.float64[:, :, :],
                          nb.float64[:, :],
                          nb.float64[:, :],
                          nb.float64[:, :], 
                          nb.float64[:, :],
                          nb.complex128[:, :],
                          nb.float64), fastmath=True)
def diff(kouts, kins, xs, ys, zs, asfcoeffs, us, wavelength):
    a = kouts.shape[0]
    b, c = xs.shape
    res = np.empty(a, dtype=np.complex128)
    kouts = np.ascontiguousarray(kouts)
    kins = np.ascontiguousarray(kins)
    us = np.ascontiguousarray(us)
    xs = np.ascontiguousarray(xs)
    ys = np.ascontiguousarray(ys)
    zs = np.ascontiguousarray(zs)
    for i in range(a):
        _res = 0.0
        for j in range(b):
            for k in range(c):
                _qs = ((kouts[i, 0] - kins[j, k, 0])**2 + (kouts[i, 1] - kins[j, k, 1])**2 + (kouts[i, 2] - kins[j, k, 2])**2) / 4e14 / wavelength**2
                _ph = kouts[i, 0] * xs[j, k] + kouts[i, 1] * ys[j, k] + kouts[i, 2] * zs[j, k]
                _res += (us[j, k] *
                         (asfcoeffs[k, 0] * exp(-asfcoeffs[k, 1] * _qs) + asfcoeffs[k, 2]) *
                         exp(-asfcoeffs[k, 3] * _qs) *
                         (cos(2 * pi / wavelength * _ph) + sin(2 * pi / wavelength * _ph) * 1j))
        res[i] = _res
    return res

def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise OSError(error.errno, error.strerror, error.filename)

def make_filename(path, filename, i=2):
    name, ext = os.path.splitext(filename)
    newname = name + "_{:d}".format(i) + ext
    if not os.path.isfile(os.path.join(path, filename)):
        return filename
    elif os.path.isfile(os.path.join(path, newname)):
        return make_filename(path, filename, i + 1)
    else:
        return newname

def get_logpath(filename=str(datetime.date.today()) + '.log'):
    make_dirs(os.path.join(PAR_PATH, LOG_PATH))
    logpath = os.path.join(PAR_PATH, LOG_PATH, filename)
    return logpath

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise
    rotation about the given axis by theta radians.

    axis - rotation axis
    theta - rotation angle
    """
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    _a = cos(theta / 2.0)
    _b, _c, _d = -axis * sin(theta / 2.0)
    return np.array([[_a * _a + _b * _b - _c * _c - _d * _d,
                      2 * (_b * _c + _a * _d),
                      2 * (_b * _d - _a * _c)],
                     [2 * (_b * _c - _a * _d),
                      _a * _a + _c * _c - _b * _b - _d * _d,
                      2 * (_c * _d + _a * _b)],
                     [2 * (_b * _d + _a * _c),
                      2 * (_c * _d - _a * _b),
                      _a * _a + _d * _d - _b * _b - _c * _c]])

def search_rec(path, ext='hdf5', file_list=None):
    """
    Search recursively in sub folders of given path for files with extension ext.

    path - a path to search
    ext - file extension to search

    Returns list of paths.
    """
    if not os.path.isdir(path):
        raise  ValueError("the path is invalid")
    if file_list is None:
        file_list = []
    for root, _, files in os.walk(path):
        for filename in files:
            if filename.endswith(ext):
                file_list.append(os.path.join(root, filename))
    return file_list

class AxesSeq(object):
    def __init__(self, datas):
        self.datas = datas
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.index = 0

    def on_keypress(self, event):
        if event.key == 'up' and self.index < len(self.datas) - 1:
            self.update_plot(self.index + 1)
        elif event.key == 'down' and self.index > 0:
            self.update_plot(self.index - 1)
        else:
            return
        self.fig.canvas.draw()

    def update_plot(self, index):
        self.index = index
        filename, (res, xs, ys) = self.datas[self.index]
        ints = np.abs(res)
        self.im.set_extent([xs.min(), xs.max(), ys.min(), ys.max()])
        self.im.set_clim(vmin=ints.min(), vmax=ints.max())
        self.im.set_data(ints)
        self.ax.set_title(filename)

    def show(self):
        filename, (res, xs, ys) = self.datas[self.index]
        ints = np.abs(res)
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        self.im = self.ax.imshow(ints,
                                 extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                                 interpolation='nearest',
                                 origin='lower')
        self.cbar = self.fig.colorbar(self.im, cax=cax, orientation='vertical')
        self.ax.set_title(filename)
        self.fig.canvas.draw()
        plt.show()

class DiffWorker(object):
    def __init__(self, kins, xs, ys, zs, us, asf_coeffs, wavelength, sigma):
        self.xs, self.ys, self.zs = xs, ys, zs
        self.us, self.kins, self.asf_coeffs, self.wavelength, self.sigma = us, kins, asf_coeffs, wavelength, sigma

    def __call__(self, kouts):
        res = diff(kouts, self.kins, self.xs, self.ys, self.zs, self.asf_coeffs, self.us, self.wavelength)
        return sqrt(self.sigma) * constants.value('classical electron radius') * 1e3 * res    