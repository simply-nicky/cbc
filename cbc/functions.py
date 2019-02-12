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
    
def asf_val(s, asf_coeffs):
    """
    Return atomic scattering factor value for given sin(theta) / lambda value.

    s - sin(theta) / lambda [Angstrom^-1]
    asf_coeffs - atomic scattering factor fit coefficients
    """
    val = 0
    for acoeff, bcoeff in zip(asf_coeffs[:5], asf_coeffs[6:]):
        val += acoeff * exp(-s**2 * bcoeff)
    return val + asf_coeffs[5]

def asf_vals(ss, asf_coeffs):
    """
    Return a numpy ndarray of atomic scattering factor values for given sin(theta) / lambda value.

    ss - sin(theta) / lambda [Angstrom^-1] numpy ndarray
    asf_coeffs - atomic scattering factor fit coefficients
    """
    acoeffs, bcoeffs = asf_coeffs[:5], asf_coeffs[6:]
    return (acoeffs * np.exp(-np.multiply.outer(ss**2, bcoeffs))).sum(axis=-1) + asf_coeffs[5]     # -ss[:, :, np.newaxis]**2 * bcoeffs[np.newaxis, :]

def gaussian(pts, waist=1e-4, wavelength=1.5e-7):
    """
    Return a np.array of gaussian beam amplitudes for given array of points.

    pts - array of points
    waist - beam waist radius
    wavelength - light wavelength
    """
    k = 2 * np.pi / wavelength
    xs, ys, zs = pts.T
    return np.pi**-1 * waist**-2 * np.exp(1j * k * zs) / (1 + 2j * zs / k / waist**2) * np.exp(-(xs**2 + ys**2) / waist**2 / (1 + 2j * zs / k / waist**2))

def gaussian_f(kins, z=0, waist=1e-4, wavelength=1.5e-7):
    """
    Return a np.array of gaussian Fourier transform beam amplitudes for given arrays of spatial frequencies kins and propagation coordinate z.

    kins - array of spatial frequencies
    z - propogation coordinate
    waist - beam waist radius
    wavelength - light wavelength
    """
    k = 2 * np.pi / wavelength
    kxs, kys = kins.T
    return (2 * np.pi)**-2 * np.exp(1j * k * z) * np.exp(-(kxs**2 + kys**2) * k**2 * (waist**2 / 4 + 1j * z / 2 / k))

def normal(mu, sigma, N):
    rs = np.random.normal(mu, sigma, N)
    phis = 2 * np.pi * np.random.rand(N)
    return np.dstack((rs * np.cos(phis), rs * np.sin(phis)))[0]

def uniform(N, a=0, b=1):
    rs = (b - a) * np.random.rand(N) + a
    phis = 2 * np.pi * np.random.rand(N)
    return np.dstack((rs * np.cos(phis), rs * np.sin(phis)))[0]

def kins(pts, waist=1e-4, wavelength=1.5e-7):
    """
    Return incoming wavevector of gaussian beam for given coordinate (x, y, z).

    waist - beam waist radius
    wavelength - light wavelength

    Return a np.array of three incoming wavevector coordinates.
    """
    k = 2 * np.pi / wavelength
    zr = k * waist**2 / 2
    xs, ys, zs = pts.T
    Rs = zs + zr**2 / zs
    return np.dstack((xs, ys, Rs))[0] / np.sqrt(xs**2 + ys**2 + Rs**2)[:, np.newaxis]

def kins_grid(rad=1, num=10):
    _kvals, _kdx = np.linspace(-rad, rad, num=num, endpoint=True, retstep=True)
    _kins = np.array([[kx, ky] for kx in _kvals for ky in _kvals])
    return (_kins[_kins[:,0]**2 + _kins[:,1]**2 < rad**2], _kdx)

def kouts(det_dist=54, detNx=512, detNy=512, pix_size=55e-3):
    """
    Return output wave vectors array for given detector at given distance from the sample.

    det_dist - distance between detector and sample
    detNx, detNy - numbers of pixels in x and y axes
    pix_size - pixel size

    Return a np.array of x and y coordinates of output wavevectors (kx, ky).
    """
    x_det = np.arange((-detNx + 1) / 2.0, (detNx + 1) / 2.0) * pix_size
    y_det = np.arange((-detNy + 1) / 2.0, (detNy + 1) / 2.0) * pix_size
    return np.array([[kx, ky] for kx in x_det / det_dist for ky in y_det / det_dist])

def kout_grid(det_dist=54, detNx=512, detNy=512, pix_size=55e-3):
    """
    Return output wave vectors array for given detector at given distance from the sample.

    det_dist - distance between detector and sample
    detNx, detNy - numbers of pixels in x and y axes
    pix_size - pixel size

    Return two (detNx, detNy) np.arrays of kx and ky output wavevector coordinates.
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

    Return a np.array of all atom positions in a sample.
    """
    assert len(lat_orig) == 3, 'origin argument is invalid, it must have 3 values'
    xval = a * np.arange((-Nx + 1) / 2.0, (Nx + 1) / 2.0)
    yval = b * np.arange((-Ny + 1) / 2.0, (Ny + 1) / 2.0)
    zval = c * np.arange((-Nz + 1) / 2.0, (Nz + 1) / 2.0)
    return np.add([[x, y, z] for x in xval for y in yval for z in zval], lat_orig)

def window(Nx, Ny, Nz):
    wx, wy, wz = map(signal.bohman, (Nx, Ny, Nz))
    return np.array([x * y * z for x in wx for y in wy for z in wz])

def diff_grid(kxs, kys, lat_pts, asf_coeffs, waist, sigma, wavelength=1.5e-7):
    """
    Return diffraction pattern intensity for given array of output wavevectors.

    kxs - x coordinates of output wavevectors
    kys - y coordinates of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    asf_coeffs - atomic scattering factor fit coefficients
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return np.array of diffracted wave values with the same shape as kxs and kys.
    """
    assert kxs.shape == kys.shape, 'kx and ky must have the same shape'
    _us = gaussian(lat_pts, waist, wavelength)
    _kins = kins(lat_pts, waist, wavelength)
    _kouts = np.concatenate((kxs[:, :, np.newaxis], kys[:, :, np.newaxis], np.sqrt(1 - kxs**2 - kys**2)[:, :, np.newaxis]), axis=2)
    _qs = np.add(_kouts[:, :, np.newaxis], -1 * _kins) / 2.0 / wavelength / 1e7
    _asfs = asf_vals(np.sqrt((_qs**2).sum(axis=-1)), asf_coeffs)
    _exps = np.exp(2 * np.pi / wavelength * np.dot(_kouts, lat_pts.T) * 1j)
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * (_asfs * _exps * _us).sum(axis=-1)   

def make_grid(kouts, fvals=None):
    """
    Return grid of coordinates (like in numpy.meshgrid) and corresponding function values grid based on list of points pts and list of function values funcvals.
    pts should be sorted the same way as nditer iterates through the grid array!

    pts - list of points
    funcvals - function values

    Return grid array for every axis and a grid of function values.
    """
    kxvals, kyvals = np.unique(kouts.T[0]), np.unique(kouts.T[1])
    kxs, kys = np.meshgrid(kxvals, kyvals)
    fgrid = np.array(fvals).reshape(kxs.shape, order='F')
    return (kxs, kys, fgrid)

def diff_gen(kouts, lat_pts, asf_coeffs, waist, sigma, wavelength=1.5e-7):
    """
    Yield diffraction pattern for given array of output wavevectors.

    kouts - tuple of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    asf_coeffs - atomic scattering factor fit coefficients
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Generator function of diffracted lightwave velues for given kouts.
    """
    _us = gaussian(lat_pts, waist, wavelength)
    _kins = kins(lat_pts, waist, wavelength)
    for kout in kouts:
        kx, ky = kout
        kout_ext = np.array([kx, ky, sqrt(1 - kx**2 - ky**2)])
        qs = np.add(-1 * _kins, kout_ext) / 2.0 / wavelength / 1e7
        asfs = np.array([asf_val(sqrt(absv), asf_coeffs) for absv in (qs * qs).sum(axis=1)])
        exps = np.exp(2 * np.pi / wavelength * np.dot(lat_pts, kout_ext) * 1j)
        yield sqrt(sigma) * constants.value('classical electron radius') * 1e3 * np.sum(asfs * _us * exps)

def diff_work(kouts, lat_pts, asf_coeffs, us, kins, sigma, wavelength=1.5e-7):
    """
    Return diffraction pattern for given array of output wavevectors.

    kouts - tuple of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    asf_coeffs - atomic scattering factor fit coefficients
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return list of diffracted wave values for given kouts.
    """
    kxs, kys = kouts.T
    kouts = np.dstack((kxs, kys, np.sqrt(1 - kxs**2 - kys**2)))[0]
    qs = np.add(kouts[:, np.newaxis], -1 * kins) / 2.0 / wavelength / 1e7
    asfs = asf_vals(np.sqrt((qs*qs).sum(axis=-1)), asf_coeffs)
    exps = np.exp(2 * np.pi / wavelength * np.dot(kouts, lat_pts.T) * 1j)
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * (asfs * exps * us).sum(axis=-1)

def diff_plane(kouts, lat_pts, window, us, asf_coeffs, kins, sigma, wavelength=1.5e-7):
    """
    Return diffraction pattern for given array of output wavevectors.

    kouts - tuple of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    asf_coeffs - atomic scattering factor fit coefficients
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return list of diffracted wave values for given kouts.
    """
    koutxs, koutys = kouts.T
    kinxs, kinys = kins.T
    _kouts = np.dstack((koutxs, koutys, np.sqrt(1 - koutxs**2 - koutys**2)))[0]
    _kins = np.dstack((kinxs, kinys, np.sqrt(1 - kinxs**2 - kinys**2)))[0]
    _qs = np.add(_kouts[:, np.newaxis], -1 * _kins) / 2.0 / wavelength / 1e7
    _asfs = asf_vals(np.sqrt((_qs**2).sum(axis=-1)), asf_coeffs)
    _phins = np.exp(-2 * np.pi / wavelength * utils.outerdot(_kins, lat_pts) * 1j) * window
    _phouts = np.exp(2 * np.pi / wavelength * utils.outerdot(_kouts, lat_pts) * 1j) 
    _exps = np.abs(utils.couterdot(_phouts, _phins))
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * (_asfs * _exps * us).sum(axis=-1)

if __name__ == "__main__":
    pass