"""
File: utilfuncs.py (Python 2.X and 3.X)

Utility functions for convergent beam diffraction project.
Dependencies: scipy, numpy and numba.
"""
import os
import datetime
import logging
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .c_funcs import diff_calc

try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

CORES_NUM = cpu_count()

PAR_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
RES_PATH = 'results'
LOG_PATH = 'logs'

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
    os.makedirs(os.path.join(PAR_PATH, LOG_PATH), exist_ok=True)
    logpath = os.path.join(PAR_PATH, LOG_PATH, filename)
    return logpath

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
        res = diff_calc(kouts, self.kins, self.xs, self.ys, self.zs, self.asf_coeffs, self.us, self.wavelength)
        return np.sqrt(self.sigma) * constants.value('classical electron radius') * 1e3 * res    