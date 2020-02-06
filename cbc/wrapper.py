"""
File: wrapper.py (Python 2.X and 3.X) - name and Python versions compatibility are temporal.

Class wrapper module for convergent beam crystallography simulation.
Dependencies: numpy, matplotlib and h5py.

Made by Nikolay Ivanov, 2018-2019.
"""
import os
import concurrent.futures
import datetime
import logging
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import utils

class Detector(object):
    """
    Detector class.

    det_dist - distance between detector and sample
    det_nx, det_ny - numbers of pixels in x and y axes
    pix_size - pixel size
    """
    def __init__(self, det_dist=54, det_nx=512, det_ny=512, pix_size=55e-3):
        self.det_dist, self.det_nx, self.det_ny, self.pix_size = det_dist, det_nx, det_ny, pix_size

    @property
    def arguments(self):
        return {'detector_distance': self.det_dist,
                'detector_size': (self.det_nx, self.det_ny),
                'pixel_size': self.pix_size}

    @property
    def shape(self):
        return (self.det_nx, self.det_ny)

    @property
    def size(self):
        return self.det_nx * self.det_ny

    def det_coordinates(self):
        x_det = np.arange((-self.det_nx + 1) / 2.0, (self.det_nx + 1) / 2.0) * self.pix_size
        y_det = np.arange((-self.det_ny + 1) / 2.0, (self.det_ny + 1) / 2.0) * self.pix_size
        xs, ys = np.meshgrid(x_det, y_det)
        return xs.ravel(), ys.ravel()

    def write(self, outfile):
        det_group = outfile.create_group('Detector')
        det_group.create_dataset('distance', data=self.det_dist)
        det_group.create_dataset('det_nx', data=self.det_nx)
        det_group.create_dataset('det_ny', data=self.det_ny)
        det_group.create_dataset('pixel_size', data=self.pix_size)

class Setup(object):
    """
    Setup class. Initializes the output folder and the logger.

    timenow - starting time
    handler - hadler for logger
    level - logger level
    relpath - path to save results
    """
    def __init__(self, timenow, handler, level, relpath):
        self.time, self.path = timenow, os.path.join(utils.PAR_PATH, relpath)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(handler)
        self.logger.level = level
        self.logger.info('Initializing')
        self.logger.info('Output path is %s', self.path)

class DiffABC(Setup):
    """
    Diffraction simulation setup class.

    beam - incoming beam class
    setup - Setup class object
    cell_args - CellArgs class object
    lat_args - LatArgs class object
    det_args - DetArgs class object
    """
    __metaclass__ = ABCMeta
    latorigin = np.zeros(3)

    def __init__(self, beam, lattice, detector=Detector(), timenow=datetime.datetime.now(), handler=utils.NullHandler(), level=logging.INFO, relpath=utils.RES_PATH):
        super(DiffABC, self).__init__(timenow, handler, level, relpath)
        self.sigma = detector.pix_size**2 / detector.det_dist**2
        self.beam, self.lattice, self.detector = beam, lattice, detector
        self.detorigin = np.array([0, 0, self.detector.det_dist])

    @property
    def distance(self):
        return self.detorigin - self.latorigin

    @abstractmethod
    def kouts(self): pass

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
        Move the sample to the point pt = [x, y, z].
        """
        self.latorigin = pt

    def move_det(self, x, y):
        """
        Move the center of detector to the point [x, y].
        """
        self.detorigin[:2] = [x, y]

    def coordinates(self):
        """
        Return lattice atoms coordinates.
        """
        xs, ys, zs = self.lattice.coordinates()
        return xs + self.latorigin[0], ys + self.latorigin[1], zs + self.latorigin[2]

    def calculate(self):
        """
        Convergent gaussian beam diffraction based on Kinematical theory equations.
        """
        self.logger.info("Setting up the convergent beam diffraction with following parameters:")
        for args in (self.lattice, self.detector):
            for (key, value) in args.arguments.items():
                self.logger.info('%-9s=%+28s', key, value)
        return DiffCalc(self)

class DiffSA(DiffABC):
    def kouts(self):
        xs, ys = self.detector.det_coordinates()
        dx, dy, dz = xs + self.distance[0], ys + self.distance[1], self.distance[2]
        return np.stack((dx / dz, dy / dz, 1 - (dx**2 + dy**2) / dz**2 / 2), axis=1)

class Diff(DiffABC):
    def kouts(self):
        xs, ys = self.detector.det_coordinates()
        dx, dy, dz = xs + self.distance[0], ys + self.distance[1], self.distance[2]
        kxs = np.sin(np.arctan(np.sqrt(dx**2 + dy**2) / dz)) * np.cos(np.arctan2(dy, dx))
        kys = np.sin(np.arctan(np.sqrt(dx**2 + dy**2) / dz)) * np.sin(np.arctan2(dy, dx))
        return np.stack((kxs, kys, np.sqrt(1 - kxs**2 - kys**2)), axis=1)

class DiffYar(DiffABC):
    def kouts(self):
        xs, ys = self.detector.det_coordinates()
        dx, dy, dz = xs + self.distance[0], ys + self.distance[1], self.distance[2]
        kxs = np.arctan(np.sqrt(dx**2 + dy**2) / dz) * np.cos(np.arctan2(dy, dx))
        kys = np.arctan(np.sqrt(dx**2 + dy**2) / dz) * np.sin(np.arctan2(dy, dx))
        return np.stack((kxs, kys, 1 - (kxs**2 + kys**2) / 2), axis=1)

class DiffCalc(object):
    """
    Diffraction calculation class.

    setup - diffraction simulation setup class object
    """
    thread_size = 2**25
    k_size = 2**10

    def __init__(self, setup):
        self.setup = setup
        self.kouts = self.setup.kouts()
        self.xs, self.ys, self.zs = self.setup.coordinates()
        self.asf_coeffs = setup.lattice.cell.asf(setup.beam.wavelength)
        self.kins = setup.beam.wavevectors(self.xs, self.ys, self.zs)
        self.us = setup.beam.wave(self.xs, self.ys, self.zs)
        thread_num = self.size // self.thread_size + 1
        self.k_thread_num = self.setup.detector.size // self.k_size + 1
        self.lat_thread_num = thread_num // self.k_thread_num + 1
        self.setup.logger.info('Chunking the data, wavevectors data thread number: %d, lattice data thread number: %d',
                               self.k_thread_num,
                               self.lat_thread_num)

    @property
    def size(self):
        return self.xs.size * self.setup.detector.size

    def serial(self):
        self.setup.logger.info('Starting serial calculation')
        res = []
        for xs, ys, zs, kins, us in zip(np.array_split(self.xs, self.lat_thread_num),
                                        np.array_split(self.ys, self.lat_thread_num),
                                        np.array_split(self.zs, self.lat_thread_num),
                                        np.array_split(self.kins, self.lat_thread_num),
                                        np.array_split(self.us, self.lat_thread_num)):
            _chunkres = []
            worker = utils.DiffWorker(kins, xs, ys, zs, us, self.asf_coeffs, self.setup.beam.wavelength, self.setup.sigma)
            for kouts in np.array_split(self.kouts, self.k_thread_num):
                _chunkres.extend(worker(kouts))
            res.append(_chunkres)
        res = np.array(res).sum(axis=0).reshape(self.setup.detector.shape)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total', res.size)
        return DiffRes(self.setup, res, self.kouts.reshape((self.setup.detector.shape + (3,))))

    def pool(self):
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
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total', res.size)
        return DiffRes(self.setup, res, self.kouts.reshape((self.setup.detector.shape + (3,))))

class DiffRes(object):
    """
    Diffraction results class.

    setup - diffraction simulation setup class object
    res - diffracted wave values for given output wavevectors
    kouts - output wavevectors
    """
    def __init__(self, setup, res, kouts):
        self.setup, self.res, self.kouts = setup, res, kouts

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
        im = ax.imshow(ints, interpolation='nearest', origin='lower')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()
        self.setup.logger.info('Plotting has ended')

    def savefig(self, figsize=(10, 10), xlim=None, ylim=None):
        if xlim == None:
            xlim = (0, self.res.shape[0])
        if ylim == None:
            ylim = (0, self.res.shape[1])
        self.setup.logger.info('Saving the results in eps image')
        _filename = 'diff_' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.eps'
        _file_path = utils.make_filename(self.setup.path, _filename)
        self.setup.logger.info('Filename: %s', _filename)
        ints = np.abs(self.res)[slice(*ylim), slice(*xlim)]
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = ax.imshow(ints, interpolation='nearest', origin='lower')
        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.savefig(_file_path, format='eps')
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
        im = ax.imshow(ints,
                       norm=LogNorm(vmin=ints.min(), vmax=ints.max()),
                       interpolation='nearest',
                       origin='lower')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()
        self.setup.logger.info('Plotting has ended')

    def write(self):
        self.setup.logger.info('Writing the results:')
        self.setup.logger.info('Folder: %s' % self.setup.path)
        utils.make_dirs(self.setup.path)
        _filename = 'diff_' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.hdf5'
        _file_path = utils.make_filename(self.setup.path, _filename)
        self.setup.logger.info('Filename: %s', _filename)
        _filediff = h5py.File(os.path.join(self.setup.path, _file_path), 'w')
        _diff_setup = _filediff.create_group('experiment setup')
        _diff_setup.create_dataset('sample position', data=self.setup.latorigin)
        _diff_setup.create_dataset('detector position', data=self.setup.detorigin)
        for args in (self.setup.beam, self.setup.lattice, self.setup.detector):
            args.write(_diff_setup)
        _diff_res = _filediff.create_group('data')
        _diff_res.create_dataset('data', data=self.res)
        _diff_res.create_dataset('wavevectors', data=self.kouts)
        _filediff.close()
        self.setup.logger.info('Writing is completed')
