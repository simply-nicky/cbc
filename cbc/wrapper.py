"""
File: wrapper.py (Python 2.X and 3.X) - name and Python versions compatibility are temporal.

Class wrapper module for convergent beam crystallography simulation.
Dependencies: numpy, matplotlib abd h5py.

Made by Nikolay Ivanov, 2018-2019.
"""

from .functions import rotation_matrix, asf_coeffs, asf_correct, asf_val, gaussian, gaussian_f, lattice, make_grid, kins, kins_grid, kouts, kout_grid, diff_grid, diff_work, diff_plane
from . import utils
import numpy as np, os, concurrent.futures, h5py, datetime, logging, errno
from functools import partial
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

try:
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

class ASF(object):
    def __init__(self, asf_fit='cbc/asf/Au/asf_q_fit.txt', asf_hw='cbc/asf/Au/asf_hw.txt', wavelength=1.5e-7):
        self.coeffs = asf_coeffs(asf_fit)
        asf_correct(self.coeffs, asf_hw, wavelength)

    def __call__(self, s):
        return asf_val(s, self.coeffs)

class lat_args(object): 
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

class kout_args(object):
    """
    kouts function arguments class.

    det_dist - distance between detector and sample
    detNx, detNy - numbers of pixels in x and y axes
    pix_size - pixel size
    """
    def __init__(self, det_dist=54, detNx=512, detNy=512, pix_size=55e-3):
        self.det_dist, self.detNx, self.detNy, self.pix_size = det_dist, detNx, detNy, pix_size

class asf_args(object):
    """
    asf function arguments class.

    asf_hw - the filename with atomic scattering factor values for different photon energies
    asf_fit - the filename with analytical fit coefficients
    """
    def __init__(self, asf_hw='cbc/asf/Au/asf_hw.txt', asf_fit='cbc/asf/Au/asf_q_fit.txt'):
        self.asf_hw, self.asf_fit = asf_hw, asf_fit

class setup_args(object):
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

class diff_setup(object):
    """
    Diffraction setup class.
    Initializes logger, the path where to save results and starting time.
    """
    def __init__(self, setup_args=setup_args()):
        self.time, self.path, self.logger = setup_args.time, setup_args.path, logging.getLogger(self.__class__.__name__)
        self.logger.addHandler(setup_args.handler)
        self.logger.level = setup_args.level
        self.logger.info('Initializing diff_setup')
        self.logger.info('output path is %s' % self.path)

class diff(diff_setup):
    """
    Diffraction simulation arguments class.

    self_args, lat_args, kout_args and asf_args - class objects
    waist - beam waist radius
    wavelength - light wavelength
    """
    def __init__(self, setup_args=setup_args(), lat_args=lat_args(), kout_args=kout_args(), asf_args=asf_args(), waist=2e-5, wavelength=1.5e-7):
        super(diff, self).__init__(setup_args)
        self.waist, self.wavelength, self.sigma = waist, wavelength, kout_args.pix_size**2 / kout_args.det_dist**2
        self.lat_args, self.kout_args, self.asf_args = lat_args, kout_args, asf_args
        self.lat_pts = lattice(**self.lat_args.__dict__)

    def rotate_lat(self, axis, theta):
        self.lat_pts -= self.lat_args.lat_orig
        self.lat_pts = np.tensordot(self.lat_pts, rotation_matrix(axis, theta), axes=(1,1))
        self.lat_pts += self.lat_args.lat_orig
    
    def move_lat(self, z=None):
        if z is None:
            z = max(self.lat_args.Nx * self.lat_args.a, self.lat_args.Ny * self.lat_args.b, self.lat_args.Nz * self.lat_args.c) / self.wavelength * np.pi * self.waist
        self.lat_args.lat_orig = [0, 0, z]
        self.lat_pts += [0, 0, z]

    def diff(self):
        """
        Serially calculate diffraction results in a grid.
        """
        self.logger.info('Starting serial calculation of diffraction pattern with following parameters:')
        for args in (self.lat_args, self.kout_args):
            for (key, value) in args.__dict__.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        _kxs, _kys = kout_grid(**self.kout_args.__dict__)
        _asf_coeffs = ASF(wavelength=self.wavelength, **self.asf_args.__dict__).coeffs
        _diffs = diff_grid(_kxs, _kys, self.lat_pts, asf_coeffs=_asf_coeffs, waist=self.waist, sigma=self.sigma, wavelength=self.wavelength)
        self.logger.info('The calculation has ended, %d diffraction pattern values total' % _diffs.size)
        return diff_res(_kxs, _kys, _diffs, self.time, self.path, self.logger, self.lat_args, self.kout_args, self.asf_args, self.waist, self.wavelength)

    def diff_pool(self, chunk_size=2**8):
        """
        Concurrently calculate diffraction pattern.
        """
        self.logger.info('Starting concurrent calculation of diffraction pattern with following parameters:')
        for args in (self.lat_args, self.kout_args):
            for (key, value) in args.__dict__.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        _kouts = kouts(**self.kout_args.__dict__)
        _us = gaussian(self.lat_pts, self.waist, self.wavelength)
        _asf_coeffs = ASF(wavelength=self.wavelength, **self.asf_args.__dict__).coeffs
        _kins = kins(self.lat_pts, self.waist, self.wavelength)
        _worker = partial(diff_work, lat_pts=self.lat_pts, kins=_kins, us=_us, asf_coeffs=_asf_coeffs, sigma=self.sigma, wavelength=self.wavelength)
        _n = max(cpu_count(), len(_kouts) / chunk_size)
        _diff_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for diff in executor.map(_worker, np.array_split(_kouts, _n)):
                _diff_list.extend(diff)
        self.logger.info('The calculation has ended, %d diffraction pattern values total' % len(_diff_list))
        return diff_res(*make_grid(_kouts, _diff_list), time=self.time, path=self.path, logger=self.logger, lat_args=self.lat_args, kout_args=self.kout_args, asf_args=self.asf_args, waist=self.waist, wavelength=self.wavelength)
    
    def diff_noinfr(self, chunk_size=2**6, knum=20):
        """
        Concurrently calculate diffraction patttern with no interference.
        """
        self.logger.info('Starting concurrent calculation of diffraction pattern with no interference with following parameters:')
        for args in (self.lat_args, self.kout_args):
            for (key, value) in args.__dict__.items():
                self.logger.info('%-9s=%+28s' % (key, value))
        _kouts = kouts(**self.kout_args.__dict__)
        _kins, _kdx = kins_grid(2 * self.wavelength / np.pi / self.waist, knum)
        _us = np.abs(gaussian_f(_kins, self.lat_args.lat_orig[-1], self.waist, self.wavelength))
        _asf_coeffs = ASF(wavelength=self.wavelength, **self.asf_args.__dict__).coeffs
        _worker = partial(diff_plane, kins=_kins, lat_pts=self.lat_pts, us=_us, asf_coeffs=_asf_coeffs, sigma=self.sigma, wavelength=self.wavelength)
        _n = max(cpu_count(), len(_kouts) / chunk_size)
        _diff_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for diff in executor.map(_worker, np.array_split(_kouts, _n)):
                _diff_list.extend(_kdx**2 * diff)
        self.logger.info('The calculation has ended, %d diffraction pattern values total' % len(_diff_list))
        return diff_res(*make_grid(_kouts, _diff_list), time=self.time, path=self.path, logger=self.logger, lat_args=self.lat_args, kout_args=self.kout_args, asf_args=self.asf_args, waist=self.waist, wavelength=self.wavelength)

class diff_res(diff):
    """
    Diffraction results class.

    kxs, kys - x and y coordinates of output wavevectors
    diffs - diffracted wave values for given kxs and kys
    time, path, logger, lat_args, kout_args, asf_args, waist, wavelength - attributes inherited from diff class
    """
    def __init__(self, kxs, kys, diffs, time, path, logger, lat_args, kout_args, asf_args, waist, wavelength):
        self.kxs, self.kys, self.diffs = kxs, kys, diffs
        self.time, self.path, self.logger, self.lat_args, self.kout_args, self.asf_args, self.waist, self.wavelength = time, path, logger, lat_args, kout_args, asf_args, waist, wavelength

    def plot(self):
        self.logger.info('Plotting the results')
        plt.contourf(self.kxs, self.kys, np.abs(self.diffs))
        plt.show()

    def write(self):
        self.logger.info('Writing the results:')
        self.logger.info('Folder: %s' % self.path)
        try:
            os.makedirs(self.path)
        except OSError as e:
            if e.errno != errno.EEXIST: raise
        _filename = utils.make_filename(self.path, 'diff_' + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S') + '.hdf5')
        self.logger.info('Filename: %s' % _filename)
        _filediff = h5py.File(os.path.join(self.path, _filename), 'w')
        _diff_args = _filediff.create_group('arguments')
        _diff_args.create_dataset('wavelength', data=self.wavelength)
        _diff_args.create_dataset('beam waist radius', data=self.waist)
        for args in (self.lat_args, self.asf_args, self.kout_args):
            for (key, value) in args.__dict__.items():
                _diff_args.create_dataset(key, data=value)
        _diff_res = _filediff.create_group('results')
        _diff_res.create_dataset('x coordinate of output wavevectors', data=self.kxs)
        _diff_res.create_dataset('y coordinate of output wavevectors', data=self.kys)
        _diff_res.create_dataset('diffracted lightwave values', data=self.diffs)
        _filediff.close()
        self.logger.info('Writing is completed')