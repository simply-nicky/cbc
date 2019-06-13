"""
File: wrapper.py (Python 2.X and 3.X) - name and Python versions compatibility are temporal.

Class wrapper module for convergent beam crystallography simulation.
Dependencies: numpy, matplotlib and h5py.

Made by Nikolay Ivanov, 2018-2019.
"""

from .functions import asf_coeffs, rbeam, cbeam, lensbeam_kins, gaussian, gaussian_f, gaussian_kins, gaussian_dist, bessel, bessel_kins, uniform_dist, lattice, det_kouts, diff_henry, diff_conv, diff_nocoh
from . import utils
import numpy as np, os, concurrent.futures, h5py, datetime, logging
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
    Lattice function arguments class.
    
    Nx, Ny, Nz - numbers of unit cells in a sample
    a, b, c - unit cell edge lengths
    lat_orig - lattice origin point
    """
    lat_orig = np.zeros(3)

    def __init__(self, a=2e-5, b=2.5e-5, c=3e-5, Nx=20, Ny=20, Nz=20):
        self.a, self.b, self.c = a, b, c
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz

class CellArgs(object):
    """
    Unit cell function arguments class.
    
    XS, YS, ZS - atom coordinates within the unit cell
    bs - an array of B-factors
    elems - an array of the abbreviations of chemical elements
    """
    def __init__(self, XS=np.zeros(1), YS=np.zeros(1), ZS=np.zeros(1), bs=np.zeros(1), elems=['Au']):
        self.XS, self.YS, self.ZS, self.bs, self.elems = XS, YS, ZS, bs, elems

    @classmethod
    def importpdb(cls, filename):
        return cls(*utils.pdb.importpdb(filename))

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
    def __init__(self, timenow=datetime.datetime.now(), handler = NullHandler(), level=logging.INFO, relpath=utils.res_relpath):
        self.level, self.time, self.path = level, timenow, os.path.join(utils.parpath, relpath)
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

class GausBeam(Beam):
    """
    Gaussian beam class.

    waist - beam waist radius
    wavelength - light wavelength
    """
    def __init__(self, waist, wavelength):
        self.waist, self.wavelength = waist, wavelength
        self.us, self.ds, self.ks, self.uf = gaussian, gaussian_dist, gaussian_kins, gaussian_f

    def wave(self, xs, ys, zs):
        return self.us(xs, ys, zs, self.waist, self.wavelength) * utils.phase_inc(self.wavevectors(xs, ys, zs), xs, ys, zs, self.wavelength)

    def wavevectors(self, xs, ys, zs):
        return self.ks(xs, ys, zs, self.waist, self.wavelength)

    def dist(self, N):
        return self.ds(N, self.waist, self.wavelength)

    def fphase(self, kxs, kys, z):
        return self.uf(kxs, kys, z, self.waist, self.wavelength)

class BesselBeam(Beam):
    """
    Bessel beam class.

    waist - beam waist radius
    wavelength - light wavelength
    """
    def __init__(self, waist, wavelength):
        self.waist, self.wavelength = waist, wavelength
        self.us, self.ds, self.ks = bessel, uniform_dist, bessel_kins

    def wave(self, xs, ys, zs):
        return self.us(xs, ys, zs, self.waist, self.wavelength) * utils.phase_inc(self.wavevectors(xs, ys, zs), xs, ys, zs, self.wavelength)

    def wavevectors(self, xs, ys, zs):
        return self.ks(xs, ys, zs, self.waist, self.wavelength)

    def dist(self, N):
        return self.ds(N, self.waist, self.wavelength)

class RectBeam(Beam):
    """
    Rectangular aperture lens beam class.

    f - focal length
    ap - half aperture size
    wavelength - light wavelength
    """
    def __init__(self, f, ap, wavelength):
        self.f, self.ap, self.wavelength = f, ap, wavelength
        self.us,  self.ks = rbeam, lensbeam_kins

    def wave(self, xs, ys, zs):
        worker = partial(self.us, f=self.f, ap=self.ap, wavelength=self.wavelength)
        us = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for u in executor.map(worker_star(worker), zip(np.array_split(xs, cpu_count()), np.array_split(ys, cpu_count()), np.array_split(zs, cpu_count()))):
                us.extend(u)
        return np.array(us)

    def wavevectors(self, xs, ys, zs):
        return self.ks(xs, ys, zs, self.f, self.wavelength)

class CircBeam(Beam):
    """
    Circular aperture lens beam class.

    f - focal length
    ap - half aperture size
    wavelength - light wavelength
    """
    def __init__(self, f, ap, wavelength):
        self.f, self.ap, self.wavelength = f, ap, wavelength
        self.us, self.ks = cbeam, lensbeam_kins

    def wave(self, xs, ys, zs):
        worker = partial(self.us, f=self.f, ap=self.ap, wavelength=self.wavelength)
        us = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for u in executor.map(worker_star(worker), zip(np.array_split(xs, cpu_count()), np.array_split(ys, cpu_count()), np.array_split(zs, cpu_count()))):
                us.extend(u)
        return np.array(us)

    def wavevectors(self, xs, ys, zs):
        return self.ks(xs, ys, zs, self.f, self.wavelength)

class Diff(DiffSetup):
    """
    Diffraction simulation setup class.

    beam - incoming beam class
    setup_args - SetupArgs class object
    cell_args - CellArgs class object
    lat_args - LatArgs class object
    det_args - DetArgs class object
    """
    def __init__(self, beam, setup_args=SetupArgs(), cell_args=CellArgs(), lat_args=LatArgs(), det_args=DetArgs()):
        DiffSetup.__init__(self, setup_args)
        self.sigma = det_args.pix_size**2 / det_args.det_dist**2
        self.beam, self.cell_args, self.lat_args, self.det_args = beam, cell_args, lat_args, det_args
        self.xs, self.ys, self.zs = lattice(a=self.lat_args.a, b=self.lat_args.b, c=self.lat_args.c, Nx=self.lat_args.Nx, Ny=self.lat_args.Ny, Nz=self.lat_args.Nz,
                                            XS=self.cell_args.XS, YS=self.cell_args.YS, ZS=self.cell_args.ZS, lat_orig=self.lat_args.lat_orig)

    def rotate_lat(self, axis, theta):
        """
        Rotate the sample around the axis by the angle theta

        axis = [nx, ny, nz] - rotation axis vector
        theta - rotation angle
        """
        self.xs -= self.lat_args.lat_orig[0]
        self.ys -= self.lat_args.lat_orig[1]
        self.zs -= self.lat_args.lat_orig[2]
        self.xs, self.ys, self.zs = utils.rotate(utils.rotation_matrix(axis, theta), self.xs, self.ys, self.zs)
        self.xs += self.lat_args.lat_orig[0]
        self.ys += self.lat_args.lat_orig[1]
        self.zs += self.lat_args.lat_orig[2]
    
    def move_lat(self, pt):
        """
        Move the sample to the point pt.

        pt = [x, y, z] - array of point coordinates
        """
        assert len(pt) == 3, 'Point mest be size of three'
        self.xs += (pt[0] - self.lat_args.lat_orig[0])
        self.ys += (pt[1] - self.lat_args.lat_orig[1])
        self.zs += (pt[2] - self.lat_args.lat_orig[2])
        self.lat_args.lat_orig = pt

    def henry(self):
        """
        Convergent gaussian beam diffraction based on Henry's equations.
        """
        self.logger.info("Setup for diffraction based on Henry's equations with following parameters:")
        for args in (self.lat_args, self.det_args):
            for (key, value) in args.__dict__.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        _kxs, _kys = det_kouts(**self.det_args.__dict__)
        _asf_coeffs = asf_coeffs(self.cell_args.elems, self.cell_args.bs, self.beam.wavelength)
        _kins = self.beam.wavevectors(self.xs, self.ys, self.zs)
        _us = self.beam.wave(self.xs, self.ys, self.zs)
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
        _asf_coeffs = asf_coeffs(self.cell_args.elems, self.cell_args.bs, self.beam.wavelength)
        _kjs = np.repeat(self.beam.dist(knum)[:,np.newaxis], self.xs.shape[-1], axis=1)
        _ufs = self.beam.fphase(_kjs[:,:,0], _kjs[:,:,1], self.lat_args.lat_orig[-1])
        _worker = partial(diff_conv, xs=self.xs, ys=self.ys, zs=self.zs, kjs=_kjs, ufs=_ufs, asf_coeffs=_asf_coeffs, sigma=self.sigma, wavelength=self.beam.wavelength)
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
        _asf_coeffs = asf_coeffs(self.cell_args.elems, self.cell_args.bs, self.beam.wavelength)
        _kjs = np.repeat(self.beam.dist(knum)[:,np.newaxis], self.xs.shape[-1], axis=1)
        _ufs = self.beam.fphase(_kjs[:,:,0], _kjs[:,:,1], self.lat_args.lat_orig[-1])
        _worker = partial(diff_nocoh, xs=self.xs, ys=self.ys, zs=self.zs, kjs=_kjs, ufs=_ufs, asf_coeffs=_asf_coeffs, sigma=self.sigma, wavelength=self.beam.wavelength)
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
    thread_size = 20000000          # ~1-2 Gb peak RAM usage per thread

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
        for args in (self.setup.cell_args, self.setup.beam, self.setup.lat_args, self.setup.det_args):
            _args_group = _diff_setup.create_group(args.__class__.__name__)
            for (key, value) in args.__dict__.items():
                if key == 'elems':
                    _args_group.create_dataset(key, data=np.array(value, 'S2'), dtype=h5py.special_dtype(vlen=str))
                elif key == 'us' or key == 'ks' or key == 'ds' or key == 'uf':
                    pass
                else:
                    _args_group.create_dataset(key, data=value)
        _diff_res = _filediff.create_group('results')
        _diff_res.create_dataset('x coordinate of output wavevectors', data=self.kxs)
        _diff_res.create_dataset('y coordinate of output wavevectors', data=self.kys)
        _diff_res.create_dataset('diffracted lightwave values', data=self.res)
        _filediff.close()
        self.setup.logger.info('Writing is completed')