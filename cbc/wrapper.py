"""
File: wrapper.py (Python 3.X) - name and Python versions compatibility are temporal.

Class wrapper module for convergent beam crystallography simulation.
Dependencies: numpy, matplotlib abd h5py.

Made by Nikolay Ivanov, 2018-2019.
"""

from .functions import asf_advanced, gaussian, lattice, make_grid, kin, kouts, kout_grid, diff_grid, diff_work
import numpy as np
import concurrent.futures
from functools import partial
import matplotlib.pyplot as plt
import h5py
import datetime

class lat_args(object): 
    """
    lattice function arguments class.
    
    Nx, Ny, Nz - numbers of unit cells in a sample
    a, b, c - unit cell edge lengths
    lat_orig - lattice origin point
    """
    def __init__(self, a=2e-5, b=2.5e-5, c=3e-5, Nx=20, Ny=20, Nz=20, lat_orig=[0,0,0]):
        self.a, self.b, self.c = a, b, c
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.lat_orig = lat_orig

class kout_args(object):
    """
    kouts function arguments class

    det_dist - distance between detector and sample
    detNx, detNy - numbers of pixels in x and y axes
    pix_size - pixel size
    """
    def __init__(self, det_dist=54, detNx=512, detNy=512, pix_size=55e-3):
        self.det_dist, self.detNx, self.detNy, self.pix_size = det_dist, detNx, detNy, pix_size

class asf_args(object):
    """
    asf_advanced function arguments class

    asf_hw - the filename with atomic scattering factor values for different photon energies
    asf_fit - the filename with analytical fit coefficients
    """
    def __init__(self, asf_hw='cbc/asf_hw_Au.txt', asf_fit='cbc/asf_q_fit_Au_2.txt'):
        self.asf_hw, self.asf_fit = asf_hw, asf_fit

class diff(object):
    """
    Diffraction simulation arguments class.

    lat_args, kout_args and asf_args - class objects
    waist - beam waist radius
    wavelength - light wavelength
    """
    def __init__(self, lat_args=lat_args(), kout_args=kout_args(), asf_args=asf_args(), waist=2e-5, wavelength=1.5e-7):
        lat_args.lat_orig = [0, 0, lat_args.Nx * lat_args.c / wavelength * np.pi * waist]
        self.waist, self.wavelength, self.sigma = waist, wavelength, kout_args.pix_size**2 / kout_args.det_dist**2
        self.lat_args, self.kout_args, self.asf_args = lat_args, kout_args, asf_args
    
    def diff_grid(self):
        """
        Calculate diffraction results.
        """
        self.lat_pts = lattice(**self.lat_args.__dict__)
        _kxs, _kys = kout_grid(**self.kout_args.__dict__)
        _asf = asf_advanced(**self.asf_args.__dict__, wavelength=self.wavelength)
        _diffs = diff_grid(_kxs, _kys, self.lat_pts, asf=_asf, waist=self.waist, sigma=self.sigma, wavelength=self.wavelength)
        return diff_res(_kxs, _kys, _diffs, self.lat_args, self.kout_args, self.asf_args, self.waist, self.wavelength)

    def diff_pool(self):
        """
        Calculate diffraction results concurrently.
        """
        self.lat_pts = lattice(**self.lat_args.__dict__)
        _kouts = kouts(**self.kout_args.__dict__)
        _us = np.array([gaussian(*pt, waist=self.waist, wavelength=self.wavelength) for pt in self.lat_pts])
        _kins = np.array([kin(*pt, waist=self.waist, wavelength=self.wavelength) for pt in self.lat_pts])
        _worker = partial(diff_work, lat_pts=self.lat_pts, kins=_kins, us=_us, **self.asf_args.__dict__, sigma=self.sigma, wavelength=self.wavelength)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            _diff_list = [diff for diff in executor.map(_worker, _kouts)]
        return diff_res(*make_grid(_kouts, _diff_list), lat_args=self.lat_args, kout_args=self.kout_args, asf_args=self.asf_args, waist=self.waist, wavelength=self.wavelength)

class diff_res(diff):
    """
    Diffraction results class.

    kxs, kys - x and y coordinates of output wavevectors
    diffs - diffracted wave values for given kxs and kys
    """
    def __init__(self, kxs, kys, diffs, lat_args, kout_args, asf_args, waist, wavelength):
        super(diff_res, self).__init__(lat_args, kout_args, asf_args, waist, wavelength)
        self.kxs, self.kys, self.diffs = kxs, kys, diffs

    def plot(self):
        plt.contourf(self.kxs, self.kys, np.abs(self.diffs))
        plt.show()

    def write(self):
        _filediff = h5py.File('diff_' + datetime.datetime.now().strftime('%d-%m-%Y_%H_%M') + '.hdf5', 'w')
        _diff_args = _filediff.create_group('arguments')
        _diff_args.create_dataset('wavelength', data=self.wavelength)
        _diff_args.create_dataset('beam waist radius', data=self.waist)
        for (key, value) in dict(**self.lat_args.__dict__, **self.asf_args.__dict__, **self.kout_args.__dict__).items():
            _diff_args.create_dataset(key, data=value)
        _diff_res = _filediff.create_group('results')
        _diff_res.create_dataset('x coordinate of output wavevectors', data=self.kxs)
        _diff_res.create_dataset('y coordinate of output wavevectors', data=self.kys)
        _diff_res.create_dataset('diffracted lightwave values', data=self.diffs)