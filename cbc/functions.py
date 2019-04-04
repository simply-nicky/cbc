"""
File: functions.py (Python 2.X and 3.X) - name and Python versions compatibility are temporal.

Module with all functions for Convergent gaussian beam crystallography simulation. Every distance is in mm units.
Dependencies: scipy and numpy.

Made by Nikolay Ivanov, 2018-2019.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import signal, constants, special
from math import sqrt, cos, sin, exp, pi
from functools import partial
from . import utils
from timeit import default_timer as timer

def asf_coeffs(elems = ['Au'], bs = np.zeros(1), wavelength=1.5e-7):
    """
    Return Wasmeier and Kirfel atomic scattering factor fit coefficients. for a given chemical element elem.
    Coefficients are put in a list as follows: [a1,  a2,  a3,  a4,  a5,  c,  b1,  b2,  b3,  b4,  b5].
    c coefficient is corrected based on Henke asf value for a given wavelength.

    elem - the abbreviation of a chemical element
    wavelength - light wavelength
    """
    en = constants.c * constants.h / constants.e / wavelength * 1e3     #photon energy in eV
    _asf_list = []
    for elem, b in zip(elems, bs):
        _asf_coeffs = utils.asf.waskif[elem]
        ens, f1s = utils.asf.henke[elem][0:2]
        _asf_coeffs[5] = interp1d(ens, f1s, kind='cubic')(en) - _asf_coeffs[:5].sum()
        np.append(_asf_coeffs, b)
        _asf_list.append(_asf_coeffs)
    return np.array(_asf_list)

@utils.jit_integrand
def rbeam_integrand_re(xx, x, z, f, wavelength):
    k = 2 * np.pi / wavelength
    return cos(k * xx**2 / 2 * (1 / f - 1 / z) + k / z * x * xx)

@utils.jit_integrand
def rbeam_integrand_im(xx, x, z, f, wavelength):
    k = 2 * np.pi / wavelength
    return -sin(k * xx**2 / 2 * (1 / f - 1 / z) + k / z * x * xx)

def rbeam(xs, ys, zs, f, ap, wavelength):
    k = 2 * np.pi / wavelength
    coeffs = -1j * np.exp(1j * k * (zs + f)) / wavelength / (zs + f) * np.exp(1j * k / 2.0 / (zs + f) * (xs**2 + ys**2))
    xvals = np.array([utils.quad_complex(rbeam_integrand_re, rbeam_integrand_im, -ap, ap, args=(x, z + f, f, wavelength), limit=int(2.0 * ap / sqrt(2.0 * wavelength * abs(z)))) for x, z in zip(xs.ravel(), zs.ravel())]).reshape(xs.shape)
    yvals = np.array([utils.quad_complex(rbeam_integrand_re, rbeam_integrand_im, -ap, ap, args=(y, z + f, f, wavelength), limit=int(2.0 * ap / sqrt(2.0 * wavelength * abs(z)))) for y, z in zip(ys.ravel(), zs.ravel())]).reshape(xs.shape)
    return coeffs * xvals * yvals

def lensbeam_kins(xs, ys, zs, f, wavelength):
    Rs = np.sqrt(xs**2 + ys**2 + zs**2)
    return np.stack((xs / Rs, ys / Rs, 1 - (xs**2 + ys**2) / 2.0 / Rs**2), axis=-1)

@utils.jit_integrand
def circ_re(rr, r, z, f, wavelength):
    k = 2 * np.pi / wavelength
    return cos(k * rr**2 / 2 * (1 / f - 1 / z)) * utils.j0(k * r * rr / z) * 2 * pi * rr

@utils.jit_integrand
def circ_im(rr, r, z, f, wavelength):
    k = 2 * np.pi / wavelength
    return -sin(k * rr**2 / 2 * (1 / f - 1 / z)) * utils.j0(k * r * rr / z) * 2 * pi * rr

def cbeam(xs, ys, zs, f, ap, wavelength):
    k = 2 * np.pi / wavelength
    coeffs = -1j * np.exp(1j * k * (zs + f)) / wavelength / (zs + f) * np.exp(1j * k * (xs**2 + ys**2) / 2.0 / (zs + f))
    rvals = np.array([utils.quad_complex(circ_re, circ_im, 0, ap, args=(sqrt(x**2 + y**2), z + f, f, wavelength), limit=int(2.0 * ap / sqrt(2.0 * wavelength * abs(z)))) for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel())]).reshape(xs.shape)
    return coeffs * rvals

def gaussian(xs, ys, zs, waist=1e-4, wavelength=1.5e-7):
    """
    Return a np.array of gaussian beam amplitudes for given array of points.

    xs, ys, zs - point coordinates
    waist - beam waist radius
    wavelength - light wavelength
    """
    zr = np.pi * waist**2 / wavelength
    wz = waist * np.sqrt(1 + zs**2 / zr**2)
    return np.pi**-1 * waist**-1 * wz**-1 * np.exp(-(xs**2 + ys**2) / wz**2)

def gaussian_f(kxs, kys, z, waist=1e-4, wavelength=1.5e-7):
    """
    Return a np.array of gaussian Fourier transform beam amplitudes for given arrays of spatial frequencies kins and propagation coordinate z.

    kxs, kys - spatial frequency coordinates
    z - propogation coordinate
    waist - beam waist radius
    wavelength - light wavelength
    """
    k = 2 * np.pi / wavelength
    return np.exp(-1j * k * z * (1.0 - (kxs**2 + kys**2) / 2.0))

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
    return np.stack((xs / Rs, ys / Rs, 1 - (xs**2 + ys**2) / 2.0 / Rs**2), axis=-1)

def gaussian_dist(N, waist, wavelength):
    """
    Return random incoming wavevector based on gaussian beam distribution.

    N - number of wavevectors
    z - propogation coordinate
    waist - beam waist radius    
    wavelength - light wavelength
    """
    thdiv = wavelength / np.pi / waist
    kxs, kys = np.random.multivariate_normal([0, 0], [[thdiv**2 / 2, 0], [0, thdiv**2 / 2]], N).T
    return kout_parax(kxs, kys)

def bessel(xs, ys, zs, waist, wavelength):
    k = 2 * np.pi / wavelength
    thdiv = wavelength / np.pi / waist
    return special.jv(1, k * thdiv * np.sqrt(xs**2 + ys**2)) / thdiv / np.pi / np.sqrt(xs**2 + ys**2)

def bessel_kins(xs, ys, zs, waist=1e-4, wavelength=1.5e-7):
    thdiv = wavelength / np.pi / waist
    return np.tile([0.0, 0.0, 1.0 - thdiv**2 / 2], xs.shape + (1,))

def uniform_dist(N, waist, wavelength):
    thdiv = wavelength / np.pi / waist
    ths = thdiv * np.sqrt(np.random.random(N))
    phis = 2 * np.pi * np.random.random(N)
    return np.stack((ths * np.cos(phis), ths * np.sin(phis), 1 - ths**2 / 2), axis=-1)

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

def lattice(a, b, c, Nx, Ny, Nz, XS=np.zeros(1), YS=np.zeros(1), ZS=np.zeros(1), lat_orig=[0, 0, 0]):
    """
    Return atom coordinates of a crystalline sample.

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
    return np.add.outer(xs.ravel(), XS), np.add.outer(ys.ravel(), YS), np.add.outer(zs.ravel(), ZS)

def diff_henry(kxs, kys, xs, ys, zs, kins, us, asf_coeffs, sigma, wavelength=1.5e-7):
    """
    Return diffraction pattern intensity for given array of output wavevectors base on Henry's equations.

    kxs, kys - x and y coordinates of output wavevectors
    xs, ys, zs - coordinates of sample lattice atoms
    kins - gaussian beam incoming wavevectors
    us - gaussian beam wave amplitudes
    asf_coeffs - atomic scattering factor fit coefficients
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return np.array of diffracted wave values with the same shape as kxs and kys.
    """
    _kouts = kout_parax(kxs, kys)
    _qabs = utils.q_abs(_kouts, kins, wavelength)
    _asfs = utils.asf_sum(_qabs, asf_coeffs)
    _phs = utils.phase(_kouts, xs, ys, zs, wavelength)
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * (_asfs * _phs * us).sum(axis=(-2,-1))

def diff_conv(kxs, kys, xs, ys, zs, kjs, ufs, asf_coeffs, sigma, wavelength):
    """
    Return diffraction pattern intensity for given array of output wavevectors base on convolution equations.

    kxs, kys - x and y coordinates of output wavevectors
    xs, ys, zs - coordinates of sample lattice atoms
    kjs - convolution incoming wavevectors based on gaussian beam distribution
    asf_coeffs - atomic scattering factor fit coefficients
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return np.array of diffracted wave values with the same shape as kxs and kys.
    """
    _kouts = kout_parax(kxs, kys)
    _qabs = utils.q_abs(_kouts, kjs, wavelength)
    _asfs = utils.asf_sum(_qabs, asf_coeffs)
    _phs = utils.phase_conv(_kouts, kjs, xs, ys, zs, wavelength)
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * (ufs * _asfs * _phs).sum(axis=(-2,-1)) / kjs.shape[0]

def diff_nocoh(kxs, kys, xs, ys, zs, kjs, ufs, asf_coeffs, sigma, wavelength):
    """
    Return diffraction pattern intensity for given array of output wavevectors base on convolution noncoherent equations.

    kxs, kys - x and y coordinates of output wavevectors
    xs, ys, zs - coordinates of sample lattice atoms
    kjs - convolution incoming wavevectors based on gaussian beam distribution
    asf_coeffs - atomic scattering factor fit coefficients
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return np.array of diffracted wave values with the same shape as kxs and kys.
    """
    _kouts = kout_parax(kxs, kys)
    _qabs = utils.q_abs(_kouts, kjs, wavelength)
    _asfs = utils.asf_sum(_qabs, asf_coeffs)
    _phs = utils.phase_conv(_kouts, kjs, xs, ys, zs, wavelength)
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * np.abs(ufs * _asfs * _phs).sum(axis=(-2,-1)) / kjs.shape[0]

if __name__ == "__main__":
    pass