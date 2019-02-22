"""
File: functions.py (Python 2.X and 3.X) - name and Python versions compatibility are temporal.

Module with all functions for Convergent gaussian beam crystallography simulation. Every distance is in mm units.
Dependencies: scipy and numpy.

Made by Nikolay Ivanov, 2018-2019.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal, constants
from math import sqrt, cos, sin, exp
from functools import partial
from . import utils
from timeit import default_timer as timer

def asf_coeffs(elem = 'Au', wavelength=1.5e-7):
    """
    Return Wasmeier and Kirfel atomic scattering factor fit coefficients. for a given chemical element elem.
    Coefficients are put in a list as follows: [a1,  a2,  a3,  a4,  a5,  c,  b1,  b2,  b3,  b4,  b5].
    c coefficient is corrected based on Henke asf value for a given wavelength.

    elem - the abbreviation of a chemical element
    wavelength - light wavelength
    """
    en = constants.c * constants.h / constants.e / wavelength * 1e3     #photon energy in eV
    _asf_coeffs = utils.asf.waskif[elem]
    ens, f1s = utils.asf.henke[elem][0:2]
    _asf_coeffs[5] = interp1d(ens, f1s, kind='cubic')(en) - _asf_coeffs[:5].sum()
    return _asf_coeffs

def asf_vals(ss, asf_coeffs):
    """
    Return a numpy ndarray of atomic scattering factor values for given sin(theta) / lambda value.

    ss - sin(theta) / lambda [Angstrom^-1] numpy ndarray
    asf_coeffs - atomic scattering factor fit coefficients
    """
    acoeffs, bcoeffs = asf_coeffs[:5], asf_coeffs[6:]
    return (utils.asf_sum(ss.ravel(), acoeffs, bcoeffs) + asf_coeffs[5]).reshape(ss.shape)

def gaussian(xs, ys, zs, waist=1e-4, wavelength=1.5e-7):
    """
    Return a np.array of gaussian beam amplitudes for given array of points.

    xs, ys, zs - point coordinates
    waist - beam waist radius
    wavelength - light wavelength
    """
    k = 2 * np.pi / wavelength
    return np.pi**-1 * waist**-2 * np.exp(-1j * k * zs) / (1 - 2j * zs / k / waist**2) * np.exp(-(xs**2 + ys**2) / waist**2 / (1 - 2j * zs / k / waist**2))

def gaussian_f(kxs, kys, z, waist=1e-4, wavelength=1.5e-7):
    """
    Return a np.array of gaussian Fourier transform beam amplitudes for given arrays of spatial frequencies kins and propagation coordinate z.

    kxs, kys - spatial frequency coordinates
    z - propogation coordinate
    waist - beam waist radius
    wavelength - light wavelength
    """
    k = 2 * np.pi / wavelength
    return (2 * np.pi)**-2 * np.exp(-1j * k * z) * np.exp(-(kxs**2 + kys**2) * k**2 * waist**2 / 4) * np.exp(-0.5j * k * (kxs**2 + kys**2)  * z)

def gaussian_kins(xs, ys, zs, waist=1e-4, wavelength=1.5e-7):
    """
    Return incoming wavevector of gaussian beam for given coordinate (x, y, z).

    xs, ys, zs - point coordinates
    waist - beam waist radius
    wavelength - light wavelength

    Return a np.array of three incoming wavevector coordinates.
    """
    zr = np.pi * waist**2 / wavelength
    Rs = zs + zr**2 / zs
    return np.dstack((xs / Rs, ys / Rs, 1 - (xs**2 + ys**2) / 2.0 / Rs**2))[0]

def gaussian_dist(N, z, waist, wavelength):
    """
    Return random incoming wavevector based on gaussian beam distribution.

    N - number of wavevectors
    z - propogation coordinate
    waist - beam waist radius    
    wavelength - light wavelength
    """
    zr = np.pi * waist**2 / wavelength
    wz = waist * sqrt(1 + z**2 / zr**2)
    thdiv = wavelength / np.pi / wz
    return np.random.multivariate_normal([0, 0], [[thdiv**2 / 2, 0], [0, thdiv**2 / 2]], N)

def kout_parax(kxs, kys):
    """
    Return wavevector in paraxial approximation.

    kxs, kys - spatial frequency coordinates
    """
    kouts = np.empty(kxs.shape + (3,))
    kouts[:,0] = kxs; kouts[:,1] = kys; kouts[:,2] = 1.0 - (kxs**2 + kys**2) / 2.0
    return kouts

def det_kouts(det_dist=54, detNx=512, detNy=512, pix_size=55e-3):
    """
    Return output wave vectors array for given detector at given distance from the sample.

    det_dist - distance between detector and sample
    detNx, detNy - numbers of pixels in x and y axes
    pix_size - pixel size

    Return a tuple x and y coordinates of output wavevectors (kx, ky).
    """
    x_det = np.arange((-detNx + 1) / 2.0, (detNx + 1) / 2.0) * pix_size
    y_det = np.arange((-detNy + 1) / 2.0, (detNy + 1) / 2.0) * pix_size
    return np.meshgrid(x_det / det_dist, y_det / det_dist)

def lattice(a, b, c, Nx, Ny, Nz, lat_orig=[0, 0, 0]):
    """
    Return atom coordinates of a cristalline sample.

    Nx, Ny, Nz - numbers of unit cells in a sample
    a, b, c - unit cell edge lengths
    lat_orig - lattice origin point

    Return a tuple of atom position coordinates (xs, ys, zs).
    """
    assert len(lat_orig) == 3, 'origin argument is invalid, it must have 3 values'
    xval = a * np.arange((-Nx + 1) / 2.0, (Nx + 1) / 2.0) + lat_orig[0]
    yval = b * np.arange((-Ny + 1) / 2.0, (Ny + 1) / 2.0) + lat_orig[1]
    zval = c * np.arange((-Nz + 1) / 2.0, (Nz + 1) / 2.0) + lat_orig[2]
    xs, ys, zs = np.meshgrid(xval, yval, zval)
    return xs.ravel(), ys.ravel(), zs.ravel()

def diff_henry(kxs, kys, xs, ys, zs, kins, us, asf_coeffs, waist, sigma, wavelength=1.5e-7):
    """
    Return diffraction pattern intensity for given array of output wavevectors base on Henry's equations.

    kxs, kys - x and y coordinates of output wavevectors
    xs, ys, zs - coordinates of sample lattice atoms
    kins - gaussian beam incoming wavevectors
    us - gaussian beam wave amplitudes
    asf_coeffs - atomic scattering factor fit coefficients
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return np.array of diffracted wave values with the same shape as kxs and kys.
    """
    _kouts = kout_parax(kxs, kys)
    _qabs = utils.q_abs(_kouts, kins) / 2.0 / wavelength / 1e7
    _asfs = asf_vals(_qabs, asf_coeffs)
    _phs = utils.phase(_kouts, xs, ys, zs, wavelength)
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * (_asfs * _phs * us).sum(axis=-1)

def diff_conv(kxs, kys, xs, ys, zs, kis, kjs, us, asf_coeffs, waist, sigma, wavelength):
    """
    Return diffraction pattern intensity for given array of output wavevectors base on convolution equations.

    kxs, kys - x and y coordinates of output wavevectors
    xs, ys, zs - coordinates of sample lattice atoms
    kins - gaussian beam incoming wavevectors
    kjs - convolution incoming wavevectors based on gaussian beam distribution
    us - gaussian beam wave amplitudes
    asf_coeffs - atomic scattering factor fit coefficients
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return np.array of diffracted wave values with the same shape as kxs and kys.
    """
    _kouts = kout_parax(kxs, kys)
    _qabs = utils.q_abs_conv(_kouts, kis, kjs)
    _asfs = asf_vals(_qabs, asf_coeffs)
    _phs = utils.phase_conv(_kouts, kjs, xs, ys, zs, wavelength)
    _phis = np.exp(-2j * np.pi / wavelength * np.einsum('ij,ji->i', kis, np.vstack((xs, ys, zs))))
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * ((_asfs * _phs).sum(axis=-1) * _phis).sum(axis=-1) / kjs.shape[0]

if __name__ == "__main__":
    pass