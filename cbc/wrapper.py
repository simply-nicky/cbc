"""
File: wrapper.py (Python 2.X and 3.X) - name and Python versions compatibility are temporal.

Class wrapper module for convergent beam crystallography simulation.
Dependencies: numpy, matplotlib abd h5py.

Made by Nikolay Ivanov, 2018-2019.
"""

from .functions import asf_coeffs, gaussian, gaussian_f, gaussian_kins, gaussian_dist, bessel, bessel_kins, uniform_dist, lattice, det_kouts, diff_henry, diff_conv, diff_nocoh
from . import utils
import numpy as np, os, concurrent.futures, h5py, datetime, logging, errno
from functools import partial
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from abc import ABCMeta, abstractmethod

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

class LatArgs(object): 
    """
    lattice function arguments class.
    
    Nx, Ny, Nz - numbers of unit cells in a sample
    a, b, c - unit cell edge lengths
    lat_orig - lattice origin point
    """
    lat_orig = np.zeros(3)

    def __init__(self, a=2e-5, b=2.5e-5, c=3e-5, Nx=20, Ny=20, Nz=20):
        self.a, self.b, self.c = a, b, c
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz

class DetArgs(object):
    """
    det_kouts function arguments class.

    det_dist - distance between detector and sample
    detNx, detNy - numbers of pixels in x and y axes
    pix_size - pixel size
    """
    def __init__(self, det_dist=54, detNx=512, detNy=512, pix_size=55e-3):
        self.det_dist, self.detNx, self.detNy, self.pix_size = det_dist, detNx, detNy, pix_size

class SetupArgs(object):
    """
    diff_setup arguments class.

    timenow - starting time
    handler - hadler for logger
    level - logger level
    relpath - path to save results
    """
    def __init__(self, timenow=datetime.datetime.now(), handler = NullHandler(), level=logging.INFO, relpath='results'):
        parpath = os.path.join(os.path.dirname(__file__), os.path.pardir)
        self.level, self.time, self.path = level, timenow, os.path.join(parpath, relpath)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.handler = handler

class DiffSetup(object):
    """
    Diffraction setup class.
    Initializes logger, the path where to save results and starting time.
    """
    def __init__(self, setup_args=SetupArgs()):
        self.time, self.path, self.logger = setup_args.time, setup_args.path, logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(setup_args.handler)
        self.logger.level = setup_args.level
        self.logger.info('Initializing diff_setup')
        self.logger.info('output path is %s' % self.path)

class Beam(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def wave(self):
        pass
        
    @abstractmethod
    def wavevectors(self):
        pass

    @abstractmethod
    def dist(self):
        pass

class GausBeam(Beam):
    def __init__(self, waist, wavelength):
        self.waist, self.wavelength = waist, wavelength
        self.us, self.ds, self.ks = gaussian, gaussian_dist, gaussian_kins

    def wave(self, xs, ys, zs):
        return self.us(xs, ys, zs, self.waist, self.wavelength)

    def wavevectors(self, xs, ys, zs):
        return self.ks(xs, ys, zs, self.waist, self.wavelength)

    def dist(self, N):
        return self.ds(N, self.waist, self.wavelength)

class BesselBeam(Beam):
    def __init__(self, waist, wavelength):
        self.waist, self.wavelength = waist, wavelength
        self.us, self.ds, self.ks = bessel, uniform_dist, bessel_kins

    def wave(self, xs, ys, zs):
        return self.us(xs, ys, zs, self.waist, self.wavelength)

    def wavevectors(self, xs, ys, zs):
        return self.ks(xs, ys, zs, self.waist, self.wavelength)

    def dist(self, N):
        return self.ds(N, self.waist, self.wavelength)

class Diff(DiffSetup):
    """
    Diffraction simulation setup class.

    self_args, lat_args, det_args and asf_args - class objects
    waist - beam waist radius
    wavelength - light wavelength
    elem - sample material chemical element
    """
    def __init__(self, beam, setup_args=SetupArgs(), lat_args=LatArgs(), det_args=DetArgs(), elem='Au'):
        DiffSetup.__init__(self, setup_args)
        self.sigma, self.elem = det_args.pix_size**2 / det_args.det_dist**2, elem
        self.beam, self.lat_args, self.det_args = beam, lat_args, det_args
        self.xs, self.ys, self.zs = lattice(**self.lat_args.__dict__)

    def rotate_lat(self, axis, theta):
        """
        Rotate the sample around the axis by the angle theta
        """
        self.xs -= self.lat_args.lat_orig[0]
        self.ys -= self.lat_args.lat_orig[1]
        self.zs -= self.lat_args.lat_orig[2]
        self.xs, self.ys, self.zs = utils.rotate(utils.rotation_matrix(axis, theta), self.xs, self.ys, self.zs)
        self.xs += self.lat_args.lat_orig[0]
        self.ys += self.lat_args.lat_orig[1]
        self.zs += self.lat_args.lat_orig[2]
    
    def move_lat(self, z):
        """
        Move the sample up- or downstream by distance z.
        """
        self.lat_args.lat_orig[2] = z
        self.zs += self.lat_args.lat_orig[2]

    def henry(self):
        """
        Convergent gaussian beam diffraction based on Henry's equations.
        """
        self.logger.info("Setup for diffraction based on Henry's equations with following parameters:")
        for args in (self.lat_args, self.det_args):
            for (key, value) in args.__dict__.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        _kxs, _kys = det_kouts(**self.det_args.__dict__)
        _asf_coeffs = asf_coeffs(self.elem, self.beam.wavelength)
        _kins = self.beam.wavevectors(self.xs, self.ys, self.zs)
        _us = self.beam.wave(self.xs, self.ys, self.zs) * utils.phase_inc(_kins, self.xs, self.ys, self.zs, self.beam.wavelength)
        _worker = partial(diff_henry, xs=self.xs, ys=self.ys, zs=self.zs, kins=_kins, us=_us, asf_coeffs=_asf_coeffs, sigma=self.sigma, wavelength=self.beam.wavelength)
        _num = self.xs.size
        return DiffCalc(self, _worker, _kxs, _kys, _num)

    def conv(self, knum=1000):
        """
        Convergent gaussian beam diffraction based on convolution equations.
        """
        self.logger.info("Setup for diffraction based on convolution equations with following parameters:")
        for args in (self.lat_args, self.det_args):
            for (key, value) in args.__dict__.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        _kxs, _kys = det_kouts(**self.det_args.__dict__)
        _asf_coeffs = asf_coeffs(self.elem, self.beam.wavelength)
        _kjs = self.beam.dist(knum)
        _worker = partial(diff_conv, xs=self.xs, ys=self.ys, zs=self.zs, kjs=_kjs, asf_coeffs=_asf_coeffs, sigma=self.sigma, wavelength=self.beam.wavelength)
        _num = knum
        return DiffCalc(self, _worker, _kxs, _kys, _num)

    def nocoh(self, knum=1000):
        """
        Convergent gaussian beam diffraction based on convolution noncoherent equations.
        """
        self.logger.info("Setup for diffraction based on convolution equations with following parameters:")
        for args in (self.lat_args, self.det_args):
            for (key, value) in args.__dict__.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        _kxs, _kys = det_kouts(**self.det_args.__dict__)
        _asf_coeffs = asf_coeffs(self.elem, self.beam.wavelength)
        _kjs = self.beam.dist(knum)
        _worker = partial(diff_nocoh, xs=self.xs, ys=self.ys, zs=self.zs, kjs=_kjs, asf_coeffs=_asf_coeffs, sigma=self.sigma, wavelength=self.beam.wavelength)
        _num = knum
        return DiffCalc(self, _worker, _kxs, _kys, _num)

class DiffCalc(object):
    """
    Diffraction calculation class.

    setup - diff class object
    worker - worker function
    kxs, kys - arguments
    num - number of elements to calculate per one argument element
    """
    thread_size = 20000000          # ~1-2 Gb peak RAM usage

    def __init__(self, setup, worker, kxs, kys, num):
        self.setup, self.worker, self.kxs, self.kys, self.num = setup, worker, kxs, kys, num
    
    def serial(self):
        _chunk_size = self.thread_size // self.num
        _thread_num = self.kxs.size // _chunk_size + 1
        self.setup.logger.info('Starting serial calculation')
        _res = []
        for diff in map(worker_star(self.worker), zip(np.array_split(self.kxs.ravel(), _thread_num), np.array_split(self.kys.ravel(), _thread_num))):
                _res.extend(diff)
        _res = np.array(_res).reshape(self.kxs.shape)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total' % _res.size)
        return DiffRes(self.setup, _res, self.kxs, self.kys)

    def pool(self):
        _chunk_size = self.thread_size // self.num
        _thread_num = max(cpu_count(), self.kxs.size // _chunk_size)
        _res = []
        self.setup.logger.info('Starting concurrent calculation, %d threads, %d chunk size' % (_thread_num, _chunk_size))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for diff in executor.map(worker_star(self.worker), zip(np.array_split(self.kxs.ravel(), _thread_num), np.array_split(self.kys.ravel(), _thread_num))):
                _res.extend(diff)
        self.setup.logger.info('The calculation has ended, %d diffraction pattern values total' % len(_res))
        _res = np.array(_res).reshape(self.kxs.shape)
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

    def plot(self, figsize=(12, 10)):
        self.setup.logger.info('Plotting the results')
        ints = np.abs(self.res)
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = ax.imshow(ints, cmap='viridis', vmin=ints.min(), vmax=ints.max(),
                                extent = [self.kxs.min(), self.kxs.max(), self.kys.min(), self.kys.max()],
                                interpolation='nearest', origin='lower')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.show()
        self.setup.logger.info('Plotting has ended')

    def logplot(self, figsize=(12, 10)):
        self.setup.logger.info('Plotting the results in log scale')
        ints = np.abs(self.res)
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
        try:
            os.makedirs(self.setup.path)
        except OSError as e:
            if e.errno != errno.EEXIST: raise
        _filename = utils.make_filename(self.setup.path, 'diff_' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.hdf5')
        self.setup.logger.info('Filename: %s' % _filename)
        _filediff = h5py.File(os.path.join(self.setup.path, _filename), 'w')
        _diff_args = _filediff.create_group('arguments')
        _diff_args.create_dataset('sample\'s material', data=self.setup.elem)
        _diff_args.create_dataset('incoming beam', data=self.setup.beam.__class__)
        _diff_args.create_dataset('wavelength', data=self.setup.beam.wavelength)
        _diff_args.create_dataset('beam waist radius', data=self.setup.beam.waist)
        for args in (self.setup.lat_args, self.setup.det_args):
            for (key, value) in args.__dict__.items():
                _diff_args.create_dataset(key, data=value)
        _diff_res = _filediff.create_group('results')
        _diff_res.create_dataset('x coordinate of output wavevectors', data=self.kxs)
        _diff_res.create_dataset('y coordinate of output wavevectors', data=self.kys)
        _diff_res.create_dataset('diffracted lightwave values', data=self.res)
        _filediff.close()
        self.setup.logger.info('Writing is completed')