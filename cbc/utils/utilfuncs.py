"""
File: utilfuncs.py (Python 2.X and 3.X)

Utility functions for convergent beam diffraction project.
Dependencies: scipy, numpy and numba.
"""
from __future__ import print_function

import os, numpy as np, numba as nb, matplotlib.pyplot as plt, scipy.integrate as si, ctypes, datetime, errno, logging
from math import sqrt, cos, sin, exp, pi
from timeit import default_timer as timer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import LowLevelCallable, constants
from numba.extending import get_cython_function_address
from multiprocessing import cpu_count

cores_num = cpu_count()

addr = get_cython_function_address("scipy.special.cython_special", "j0")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
j0_c = functype(addr)

parpath = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
res_relpath = 'results'
log_relpath = 'logs'

@nb.vectorize('float64(float64)')
def j0_vec(x):
    return j0_c(x)

@nb.njit
def j0(x):
    return j0_vec(x)

def jit_integrand(func):
    jit_func = nb.njit(func)
    @nb.cfunc(nb.float64(nb.intc, nb.types.CPointer(nb.float64)))
    def wrapper(n, args):
        return jit_func(args[0], args[1], args[2])
    return LowLevelCallable(wrapper.ctypes)

def quad_complex(func_re, func_im, a, b, **args):
    re = si.quad(func_re, a, b, **args)[0]
    im = si.quad(func_im, a, b, **args)[0]
    return re + 1j * im

@nb.njit(nb.complex128[:,:](nb.float64[:,:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64), fastmath=True)
def phase_inc(kins, xs, ys, zs, wavelength):
    a, b = xs.shape
    res = np.empty((a, b), dtype=np.complex128)
    kins = np.ascontiguousarray(kins)
    xs = np.ascontiguousarray(xs)
    ys = np.ascontiguousarray(ys)
    zs = np.ascontiguousarray(zs)
    for i in range(a):
        for j in range(b):
            _ph = kins[i,j,0] * xs[i,j] + kins[i,j,1] * ys[i,j] + kins[i,j,2] * zs[i,j]
            res[i,j] = cos(2 * pi / wavelength * _ph) - sin(2 * pi / wavelength * _ph) * 1j
    return res

@nb.njit(nb.complex128[:](nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.complex128[:,:], nb.float64), fastmath=True)
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
                _qs = ((kouts[i,0] - kins[j,k,0])**2 + (kouts[i,1] - kins[j,k,1])**2 + (kouts[i,2] - kins[j,k,2])**2) / 2e7 / wavelength
                _ph = kouts[i,0] * xs[j,k] + kouts[i,1] * ys[j,k] + kouts[i,2] * zs[j,k]
                _asf = 0.0
                for l in range(5):
                    _asf += asfcoeffs[k,l] * exp(-asfcoeffs[k,l + 6] * _qs)
                _res += us[j,k] * (_asf + asfcoeffs[k, 5]) * exp(-asfcoeffs[k,-1] * _qs) * (cos(2 * pi / wavelength * _ph) + sin(2 * pi / wavelength * _ph) * 1j)
        res[i] = _res
    return res

@nb.njit(nb.types.UniTuple(nb.float64[:,:], 3)(nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:]), fastmath=True)
def rotate(m, xs, ys, zs):
    a, b = xs.shape
    XS = np.empty((a, b), dtype=np.float64)
    YS = np.empty((a, b), dtype=np.float64)
    ZS = np.empty((a, b), dtype=np.float64)
    m = np.ascontiguousarray(m)
    xs = np.ascontiguousarray(xs)
    ys = np.ascontiguousarray(ys)
    zs = np.ascontiguousarray(zs)
    for i in range(a):
        for j in range(b):
            XS[i,j] = m[0,0] * xs[i,j] + m[0,1] * ys[i,j] + m[0,2] * zs[i,j]
            YS[i,j] = m[1,0] * xs[i,j] + m[1,1] * ys[i,j] + m[1,2] * zs[i,j]
            ZS[i,j] = m[2,0] * xs[i,j] + m[2,1] * ys[i,j] + m[2,2] * zs[i,j]
    return XS, YS, ZS

def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST: raise

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
    make_dirs(os.path.join(parpath, log_relpath))
    logpath = os.path.join(parpath, log_relpath, filename)
    return logpath

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
        self.im = self.ax.imshow(ints, cmap='viridis', vmin=ints.min(), vmax=ints.max(),
                                extent = [xs.min(), xs.max(), ys.min(), ys.max()],
                                interpolation='nearest', origin='lower')
        self.cbar = self.fig.colorbar(self.im, cax=cax, orientation='vertical')
        self.ax.set_title(filename)
        self.fig.canvas.draw()
        plt.show()

try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

class worker_star(object):
    def __init__(self, worker):
        self.worker = worker
    
    def __call__(self, args):
        return self.worker(*args)

class DiffWorker(object):
    def __init__(self, kins, xs, ys, zs, us, asf_coeffs, wavelength, sigma):
        self.xs, self.ys, self.zs, self.us, self.kins, self.asf_coeffs, self.wavelength, self.sigma = xs, ys, zs, us, kins, asf_coeffs, wavelength, sigma

    def __call__(self, kouts):
        res = diff(kouts, self.kins, self.xs, self.ys, self.zs, self.asf_coeffs, self.us, self.wavelength)
        return sqrt(self.sigma) * constants.value('classical electron radius') * 1e3 * res    