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
from scipy import constants
from math import sqrt
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np, os, concurrent.futures, h5py, datetime, matplotlib.pyplot as plt, logging

class Detector(object):
    """
    det_kouts function arguments class.

    det_dist - distance between detector and sample
    detNx, detNy - numbers of pixels in x and y axes
    pix_size - pixel size
    """
    def __init__(self, det_dist=54, detNx=512, detNy=512, pix_size=55e-3):
        self.det_dist, self.detNx, self.detNy, self.pix_size = det_dist, detNx, detNy, pix_size

    @property
    def arguments(self):
        return {'detector_distance': self.det_dist, 'detector_size': (self.detNx, self.detNy), 'pixel_size': self.pix_size}

    def kouts(self):
        x_det = np.arange((-self.detNx + 1) / 2.0, (self.detNx + 1) / 2.0) * self.pix_size
        y_det = np.arange((-self.detNy + 1) / 2.0, (self.detNy + 1) / 2.0) * self.pix_size
        return np.meshgrid(x_det / self.det_dist, y_det / self.det_dist)

    def write(self, outfile):
        det_group = outfile.create_group('detector')
        det_group.create_dataset('distance', data=self.det_dist)
        det_group.create_dataset('Nx', data=self.detNx)
        det_group.create_dataset('Ny', data=self.detNy)
        det_group.create_dataset('pixel_size', data=self.pix_size)

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
    def __init__(self, beam, lattice, setup=Setup(), detector=Detector()):
        DiffSetup.__init__(self, setup)
        self.sigma = detector.pix_size**2 / detector.det_dist**2
        self.beam, self.lattice, self.detector = beam, lattice, detector
        self.xs, self.ys, self.zs = self.lattice.coordinates()

    def rotate_lat(self, axis, theta):
        """
        Rotate the sample around the axis by the angle theta

        axis = [nx, ny, nz] - rotation axis vector
        theta - rotation angle
        """
        self.xs -= self.lattice.lat_orig[0]
        self.ys -= self.lattice.lat_orig[1]
        self.zs -= self.lattice.lat_orig[2]
        self.xs, self.ys, self.zs = utils.rotate(utils.rotation_matrix(axis, theta), self.xs, self.ys, self.zs)
        self.xs += self.lattice.lat_orig[0]
        self.ys += self.lattice.lat_orig[1]
        self.zs += self.lattice.lat_orig[2]
    
    def move_lat(self, pt):
        """
        Move the sample to the point pt.

        pt = [x, y, z] - array of point coordinates
        """
        assert len(pt) == 3, 'Point mest be size of three'
        self.xs += (pt[0] - self.lattice.lat_orig[0])
        self.ys += (pt[1] - self.lattice.lat_orig[1])
        self.zs += (pt[2] - self.lattice.lat_orig[2])
        self.lattice.lat_orig = pt

    def henry(self):
        """
        Convergent gaussian beam diffraction based on Henry's equations.
        """
        self.logger.info("Setup for diffraction based on Henry's equations with following parameters:")
        for args in (self.lattice, self.detector):
            for (key, value) in args.arguments.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        kxs, kys = self.detector.kouts()
        return DiffHenry(self, kxs, kys)

    def conv(self, knum=1000):
        """
        Convergent gaussian beam diffraction based on convolution equations.
        """
        self.logger.info("Setup for diffraction based on convolution equations with following parameters:")
        for args in (self.lattice, self.detector):
            for (key, value) in args.arguments.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        kxs, kys = self.detector.kout()
        # _asf_coeffs = self.lattice.cell.asf(self.beam.wavelength)
        # _kjs = np.repeat(self.beam.dist(knum)[:,np.newaxis], self.xs.shape[-1], axis=1)
        # _ufs = self.beam.fphase(_kjs[:,:,0], _kjs[:,:,1], self.lattice.lat_orig[-1])
        # _worker = partial(diff_conv, kjs=_kjs, ufs=_ufs, asf_coeffs=_asf_coeffs, sigma=self.sigma, wavelength=self.beam.wavelength)
        # _size = knum * self.xs.size * _kxs.size
        return DiffConv(self, kxs, kys, knum)

class DiffCalc(object):
    """
    Diffraction calculation class.

    setup - diff class object
    worker - worker function
    kxs, kys - arguments
    num - number of elements to calculate per one argument element
    """
    __metaclass__ = ABCMeta
    thread_size = 2**25
    k_size = 2**8

    @abstractproperty
    def setup(self): pass

    @abstractproperty
    def kxs(self): pass

    @abstractproperty
    def kys(self): pass

    @abstractproperty
    def size(self): pass

    @property
    def kouts(self): return np.stack((self.kxs.ravel(), self.kys.ravel(), 1.0 - (self.kxs.ravel()**2 + self.kys.ravel()**2) / 2.0), axis=1)

    def _chunkify(self):
        thread_num = self.size // self.thread_size + 1
        self.k_thread_num = self.kxs.size // self.k_size + 1
        self.lat_thread_num = thread_num // self.k_thread_num + 1
        self.setup.logger.info('Chunking the data, wavevectors data thread number: %d, lattice data thread number: %d' % (self.k_thread_num, self.lat_thread_num))

class DiffHenry(DiffCalc):
    setup, kxs, kys = None, None, None

    @property
    def size(self): self.setup.xs.size * self.kxs.size

    def __init__(self, setup, kxs, kys):
        self.setup, self.kxs, self.kys = setup, kxs, kys
        self.asf_coeffs = setup.lattice.asf(setup.beam.wavelength)
        self.kins = setup.beam.wavevectors(setup.xs, setup.ys, setup.zs)
        self.us = setup.beam.wave(setup.xs, setup.ys, setup.zs)
    
    def worker(self, kouts, kins, xs, ys, zs, us):
        _asfs = utils.asf_vals(kouts, kins, self.asf_coeffs, self.setup.beam.wavelength)
        _phs = utils.phase(kouts, xs, ys, zs, self.setup.beam.wavelength)
        return sqrt(self.setup.sigma) * constants.value('classical electron radius') * 1e3 * (_asfs * _phs * us).sum(axis=(-2,-1))

    def serial(self):
        self._chunkify()
        self.setup.logger.info('Starting serial calculation')
        _res = []
        for xs, ys, zs, kins, us in zip(np.array_split(self.setup.xs, self.lat_thread_num),
                                        np.array_split(self.setup.ys, self.lat_thread_num),
                                        np.array_split(self.setup.zs, self.lat_thread_num),
                                        np.array_split(self.kins, self.lat_thread_num),
                                        np.array_split(self.us, self.lat_thread_num)):
            _chunkres = []
            for kouts in np.array_split(self.kouts, self.k_thread_num):
                _chunkres.extend(self.worker(kouts=kouts, xs=xs, ys=ys, zs=zs, kins=kins, us=us))
            _res.append(_chunkres)
        _res = np.array(_res).sum(axis=0).reshape(self.kxs.shape)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total' % _res.size)
        return DiffRes(self.setup, _res, self.kxs, self.kys)

    def pool(self):
        self._chunkify()
        _res = []
        self.setup.logger.info('Starting concurrent calculation')
        for xs, ys, zs, kins, us in zip(np.array_split(self.setup.xs, self.lat_thread_num),
                                        np.array_split(self.setup.ys, self.lat_thread_num),
                                        np.array_split(self.setup.zs, self.lat_thread_num),
                                        np.array_split(self.kins, self.lat_thread_num),
                                        np.array_split(self.us, self.lat_thread_num)):
            worker = partial(self.worker, xs=xs, ys=ys, zs=zs, kins=kins, us=us)
            _chunkres = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for chunkval in executor.map(worker, np.array_split(self.kouts, self.k_thread_num)):
                    _chunkres.extend(chunkval)
            _res.append(_chunkres)
        _res = np.array(_res).sum(axis=0).reshape(self.kxs.shape)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total' % _res.size)
        return DiffRes(self.setup, _res, self.kxs, self.kys)

class DiffConv(DiffCalc):
    setup, kxs, kys = None, None, None

    @property
    def size(self): return self.knum * self.setup.xs.size * self.kxs.size

    def __init__(self, setup, kxs, kys, knum):
        self.setup, self.kxs, self.kys, self.knum = setup, kxs, kys, knum
        self.asf_coeffs = setup.lattice.asf(setup.beam.wavelength)
        self.kjs = np.repeat(setup.beam.dist(knum)[:,np.newaxis], setup.xs.shape[-1], axis=1)
        self.ufs = setup.beam.fphase(self.kjs[:,:,0], self.kjs[:,:,1], setup.lattice.lat_orig[-1])
    
    def worker(self, kouts, kjs, xs, ys, zs, ufs):
        _asfs = utils.asf_vals(kouts, kjs, self.asf_coeffs, self.setup.beam.wavelength)
        _phs = utils.phase_conv(kouts, kjs, xs, ys, zs, self.setup.beam.wavelength)
        return sqrt(self.setup.sigma) * constants.value('classical electron radius') * 1e3 * (ufs * _asfs * _phs).sum(axis=(-2,-1)) / kjs.shape[0]

    def serial(self):
        self._chunkify()
        self.setup.logger.info('Starting serial calculation')
        _res = []
        for xs, ys, zs, kjs, ufs in zip(np.array_split(self.setup.xs, self.lat_thread_num),
                                        np.array_split(self.setup.ys, self.lat_thread_num),
                                        np.array_split(self.setup.zs, self.lat_thread_num),
                                        np.array_split(self.kjs, self.lat_thread_num),
                                        np.array_split(self.ufs, self.lat_thread_num)):
            _chunkres = []
            for kouts in np.array_split(self.kouts, self.k_thread_num):
                _chunkres.extend(self.worker(kouts=kouts, xs=xs, ys=ys, zs=zs, kjs=kjs, ufs=ufs))
            _res.append(_chunkres)
        _res = np.array(_res).reshape(self.kxs.shape)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total' % _res.size)
        return DiffRes(self.setup, _res, self.kxs, self.kys)

    def pool(self):
        self._chunkify()
        _res = []
        self.setup.logger.info('Starting concurrent calculation')
        print(self.k_thread_num, self.lat_thread_num)
        for xs, ys, zs, kjs, ufs in zip(np.array_split(self.setup.xs, self.lat_thread_num),
                                        np.array_split(self.setup.ys, self.lat_thread_num),
                                        np.array_split(self.setup.zs, self.lat_thread_num),
                                        np.array_split(self.kjs, self.lat_thread_num),
                                        np.array_split(self.ufs, self.lat_thread_num)):
            worker = partial(self.worker, xs=xs, ys=ys, zs=zs, kjs=kjs, ufs=ufs)
            _chunkres = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for chunkval in executor.map(worker, np.array_split(self.kouts, self.k_thread_num)):
                    _chunkres.extend(chunkval)
            _res.append(_chunkres)
        _res = np.array(_res)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total' % _res.size)
        return DiffRes(self.setup, _res, self.kxs, self.kys)

class DiffRes(object):
    """
    Diffraction results class.

    setup - diff class object
    res - diffracted wave values for given kxs and kys
    kxs, kys - x and y coordinates of output wavevectors
    """
    def __init__(self, setup, res, kxs, kys):
        self.setup, self.res, self.kxs, self.kys = setup, res, kxs, kys

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
                                extent = [self.kxs.min(), self.kxs.max(), self.kys.min(), self.kys.max()],
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
                                extent = [self.kxs.min(), self.kxs.max(), self.kys.min(), self.kys.max()],
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
                                extent = [self.kxs.min(), self.kxs.max(), self.kys.min(), self.kys.max()],
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
        for args in (self.setup.beam, self.setup.lattice, self.setup.detector):
            args.write(_diff_setup)
        _diff_res = _filediff.create_group('results')
        _diff_res.create_dataset('x coordinate of output wavevectors', data=self.kxs)
        _diff_res.create_dataset('y coordinate of output wavevectors', data=self.kys)
        _diff_res.create_dataset('diffracted lightwave values', data=self.res)
        _filediff.close()
        self.setup.logger.info('Writing is completed')