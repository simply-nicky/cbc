"""
File: wrapper.py (Python 2.X and 3.X) - name and Python versions compatibility are temporal.

Class wrapper module for convergent beam crystallography simulation.
Dependencies: numpy, matplotlib and h5py.

Made by Nikolay Ivanov, 2018-2019.
"""

from .lattice import Cell
from . import utils
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from math import sqrt
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np, os, concurrent.futures, h5py, datetime, matplotlib.pyplot as plt, logging

class Detector(object):
    """
    det_kouts function arguments class.

    detdist - distance between detector and sample
    Nx, Ny - numbers of pixels in x and y axes
    pixsize - pixel size
    """
    def __init__(self, detdist=54, Nx=512, Ny=512, pixsize=55e-3):
        self.detdist, self.Nx, self.Ny, self.pixsize = detdist, Nx, Ny, pixsize

    @property
    def arguments(self):
        return {'detector_distance': self.detdist, 'detector_size': (self.Nx, self.Ny), 'pixel_size': self.pixsize}

    @property
    def shape(self): return (self.Nx, self.Ny)

    @property
    def size(self): return self.Nx * self.Ny

    def det_coordinates(self):
        x_det = np.arange((-self.Nx + 1) / 2.0, (self.Nx + 1) / 2.0) * self.pixsize
        y_det = np.arange((-self.Ny + 1) / 2.0, (self.Ny + 1) / 2.0) * self.pixsize
        xs, ys = np.meshgrid(x_det, y_det)
        return xs.ravel(), ys.ravel()

    def kouts(self):
        xs, ys = self.det_coordinates()
        kxs = np.sin(np.arctan(np.sqrt(xs**2 + ys**2) / self.detdist)) * np.cos(np.arctan2(ys, xs))
        kys = np.sin(np.arctan(np.sqrt(xs**2 + ys**2) / self.detdist)) * np.sin(np.arctan2(ys, xs))
        return np.stack((kxs, kys, np.sqrt(1 - kxs**2 - kys**2)), axis=1)

    def kouts_yar(self):
        xs, ys = self.det_coordinates()
        kxs = np.arctan(np.sqrt(xs**2 + ys**2) / self.detdist) * np.cos(np.arctan2(ys, xs))
        kys = np.arctan(np.sqrt(xs**2 + ys**2) / self.detdist) * np.sin(np.arctan2(ys, xs))
        return np.stack((kxs, kys, np.sqrt(1 - kxs**2 - kys**2)), axis=1)

    def write(self, outfile):
        det_group = outfile.create_group('Detector')
        det_group.create_dataset('distance', data=self.detdist)
        det_group.create_dataset('Nx', data=self.Nx)
        det_group.create_dataset('Ny', data=self.Ny)
        det_group.create_dataset('pixel_size', data=self.pixsize)

class Setup(object):
    """
    diff_setup arguments class.

    timenow - starting time
    handler - hadler for logger
    level - logger level
    relpath - path to save results
    """
    def __init__(self, timenow=datetime.datetime.now(), handler = utils.NullHandler(), level=logging.INFO, relpath=utils.res_relpath):
        self.level, self.time, self.path = level, timenow, os.path.join(utils.parpath, relpath)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.handler = handler

class DiffSetup(object):
    """
    Diffraction setup class.
    Initializes logger, the path where to save results and starting time.
    """
    def __init__(self, setup=Setup()):
        self.time, self.path, self.logger = setup.time, setup.path, logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(setup.handler)
        self.logger.level = setup.level
        self.logger.info('Initializing diff_setup')
        self.logger.info('output path is %s' % self.path)

class Diff(DiffSetup):
    """
    Diffraction simulation setup class.

    beam - incoming beam class
    setup - Setup class object
    cell_args - CellArgs class object
    lat_args - LatArgs class object
    det_args - DetArgs class object
    """
    origin = np.zeros(3)

    def __init__(self, beam, lattice, setup=Setup(), detector=Detector()):
        DiffSetup.__init__(self, setup)
        self.sigma = detector.pixsize**2 / detector.detdist**2
        self.beam, self.lattice, self.detector = beam, lattice, detector

    def rotate_lat(self, axis, theta):
        """
        Rotate the sample around the axis by the angle theta

        axis = [nx, ny, nz] - rotation axis vector
        theta - rotation angle
        """
        m = utils.rotation_matrix(axis, theta)
        self.lattice.a = m.dot(self.lattice.a)
        self.lattice.b = m.dot(self.lattice.b)
        self.lattice.c = m.dot(self.lattice.c)
    
    def move_lat(self, pt):
        """
        Move the sample to the point pt.

        pt = [x, y, z] - array of point coordinates
        """
        self.origin = pt

    def coordinates(self):
        xs, ys, zs = self.lattice.coordinates()
        return xs + self.origin[0], ys + self.origin[1], zs + self.origin[2]

    def calculate(self):
        """
        Convergent gaussian beam diffraction based on Henry's equations.
        """
        self.logger.info("Setting up the convergent beam diffraction with following parameters:")
        for args in (self.lattice, self.detector):
            for (key, value) in args.arguments.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        return DiffCalc(self)

class DiffCalc(object):
    """
    Diffraction calculation class.

    setup - diff class object
    worker - worker function
    kxs, kys - arguments
    num - number of elements to calculate per one argument element
    """
    thread_size = 2**25
    k_size = 2**10

    def __init__(self, setup):
        self.setup = setup
        self.kouts = self.setup.detector.kouts_yar()
        self.xs, self.ys, self.zs = self.setup.coordinates()
        self.asf_coeffs = setup.lattice.cell.asf(setup.beam.wavelength)
        self.kins = setup.beam.wavevectors(self.xs, self.ys, self.zs)
        self.us = setup.beam.wave(self.xs, self.ys, self.zs)

    @property
    def size(self): return self.xs.size * self.setup.detector.size

    def _chunkify(self):
        thread_num = self.size // self.thread_size + 1
        self.k_thread_num = self.setup.detector.size // self.k_size + 1
        self.lat_thread_num = thread_num // self.k_thread_num + 1
        self.setup.logger.info('Chunking the data, wavevectors data thread number: %d, lattice data thread number: %d' % (self.k_thread_num, self.lat_thread_num))

    def serial(self):
        self._chunkify()
        self.setup.logger.info('Starting serial calculation')
        _res = []
        for xs, ys, zs, kins, us in zip(np.array_split(self.xs, self.lat_thread_num),
                                        np.array_split(self.ys, self.lat_thread_num),
                                        np.array_split(self.zs, self.lat_thread_num),
                                        np.array_split(self.kins, self.lat_thread_num),
                                        np.array_split(self.us, self.lat_thread_num)):
            _chunkres = []
            worker = utils.DiffWorker(kins, xs, ys, zs, us, self.asf_coeffs, self.setup.beam.wavelength, self.setup.sigma)
            for kouts in np.array_split(self.kouts, self.k_thread_num):
                _chunkres.extend(worker(kouts))
            _res.append(_chunkres)
        _res = np.array(_res).sum(axis=0).reshape(self.setup.detector.shape)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total' % _res.size)
        return DiffRes(self.setup, _res)

    def pool(self):
        self._chunkify()
        fut_list, res = [], []
        self.setup.logger.info('Starting concurrent calculation')
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for xs, ys, zs, kins, us in zip(np.array_split(self.xs, self.lat_thread_num),
                                            np.array_split(self.ys, self.lat_thread_num),
                                            np.array_split(self.zs, self.lat_thread_num),
                                            np.array_split(self.kins, self.lat_thread_num),
                                            np.array_split(self.us, self.lat_thread_num)):
                worker = utils.DiffWorker(kins, xs, ys, zs, us, self.asf_coeffs, self.setup.beam.wavelength, self.setup.sigma)
                futs = [executor.submit(worker, kouts) for kouts in np.array_split(self.kouts, self.k_thread_num)]
                fut_list.append(futs)
        for futs in fut_list:
            chunkres = np.concatenate([fut.result() for fut in futs])
            res.append(chunkres)
        res = np.sum(res, axis=0).reshape(self.setup.detector.shape)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total' % res.size)
        return DiffRes(self.setup, res)

class DiffRes(object):
    """
    Diffraction results class.

    setup - diff class object
    res - diffracted wave values for given kxs and kys
    kxs, kys - x and y coordinates of output wavevectors
    """
    def __init__(self, setup, res):
        self.setup, self.res = setup, res

    def plot(self, figsize=(10, 10), xlim=None, ylim=None):
        if xlim == None:
            xlim = (0, self.res.shape[0])
        if ylim == None:
            ylim = (0, self.res.shape[1])
        self.setup.logger.info('Plotting the results')
        ints = np.abs(self.res)[slice(*ylim), slice(*xlim)]
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = ax.imshow(ints, cmap='viridis', vmin=ints.min(), vmax=ints.max(),
                                interpolation='nearest', origin='lower')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()
        self.setup.logger.info('Plotting has ended')

    def savefig(self, figsize=(10, 10), xlim=None, ylim=None):
        if xlim == None:
            xlim = (0, self.res.shape[0])
        if ylim == None:
            ylim = (0, self.res.shape[1])
        self.setup.logger.info('Saving the results in eps image')
        _filename = utils.make_filename(self.setup.path, 'diff_' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.eps')
        self.setup.logger.info('Filename: %s' % _filename)
        ints = np.abs(self.res)[slice(*ylim), slice(*xlim)]
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = ax.imshow(ints, cmap='viridis', vmin=ints.min(), vmax=ints.max(),
                                interpolation='nearest', origin='lower')
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.savefig(_filename, format='eps')
        self.setup.logger.info('Image creation has ended')

    def logplot(self, figsize=(10, 10), xlim=None, ylim=None):
        if xlim == None:
            xlim = (0, self.res.shape[0])
        if ylim == None:
            ylim = (0, self.res.shape[1])
        self.setup.logger.info('Plotting the results in log scale')
        ints = np.abs(self.res)[slice(*ylim), slice(*xlim)]
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = ax.imshow(ints, cmap='viridis', norm=LogNorm(vmin=ints.min(), vmax=ints.max()),
                                interpolation='nearest', origin='lower')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()
        self.setup.logger.info('Plotting has ended')

    def write(self):
        self.setup.logger.info('Writing the results:')
        self.setup.logger.info('Folder: %s' % self.setup.path)
        utils.make_dirs(self.setup.path)
        _filename = utils.make_filename(self.setup.path, 'diff_' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.hdf5')
        self.setup.logger.info('Filename: %s' % _filename)
        _filediff = h5py.File(os.path.join(self.setup.path, _filename), 'w')
        _diff_setup = _filediff.create_group('experiment setup')
        _diff_setup.create_dataset('sample position', data=self.setup.origin)
        for args in (self.setup.beam, self.setup.lattice, self.setup.detector):
            args.write(_diff_setup)
        _diff_res = _filediff.create_group('data')
        _diff_res.create_dataset('diffracted lightwave values', data=self.res)
        _filediff.close()
        self.setup.logger.info('Writing is completed')