"""
File: functions.py (Python 2.X and 3.X) - name and Python versions compatibility are temporal.

Module with all functions for Convergent gaussian beam crystallography simulation. Every distance is in mm units.
Dependencies: scipy and numpy.

Made by Nikolay Ivanov, 2018-2019.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import constants
from math import sqrt, cos, sin
from functools import partial

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    axis - rotation axis
    theta - rotation angle
    """
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def asf_parser(filename):
    """
    Input a txt file, that contains atomic scattering factor f [el/atoms] depending on scattering wavevector or photon energy (see https://it.iucr.org/Cb/ch6o1v0001/#table6o1o1o1 for more information).
    The file must contain two columns, the first is the argument and the second is f in el/atoms.

    filename - the filename with atomic scattering factor values, the file must be located in the same folder as this program
    """
    x = []
    y = []
    for line in open(str(filename)):
        parts = line.split()
        try:
            x.append(float(parts[0]))
            y.append(float(parts[1]))
        except:
            continue
    return interp1d(x, y, kind='cubic')

def asf_fit_parser(filename):
    """
    Input a txt file, than contains atomic scattering factor analytical fit coefficients (see https://it.iucr.org/Cb/ch6o1v0001/#table6o1o1o1 for more information).
    The file must contain a given list of coefficients: [a1, b1,.. an, bn,.. , c].

    filename - the filename with atomic scattering factor fit coefficients, the file must be located in the same folder as this program
    """
    coefs = []
    for line in open(str(filename)):
        parts = line.split()
        try:
            for part in parts:
                coefs.append(float(part))
        except:
            continue
    assert len(coefs) % 2 == 1, 'the fit coefficients file is invalid, there must be odd number of coefficients'
    bcoefs, acoefs = np.array(coefs[1:-1:2]), np.array(coefs[0:-1:2])
    return lambda s: sum(acoefs * np.exp(-s**2 * bcoefs)) + coefs[-1]

def asf(asf_hw, asf_fit, wavelength=1.5e-7):
    """
    Input two txt files, the first one contains atomic scattering factor for different photon energy and the second one contains analytical fit coefficients (see https://it.iucr.org/Cb/ch6o1v0001/#table6o1o1o1 for more information).
    The first file must contain two columns, the first is the argument and the second is f in el/atoms.
    The second file must contain a given list of coefficients: [a1, b1,.. an, bn,.. , c].
    The files must be located in the same folder as this program.

    asf_hw - the filename with atomic scattering factor values for different photon energies
    asf_fit - the filename with analytical fit coefficients
    wavelength - light wavelength
    """
    en = constants.c * constants.h / constants.e / wavelength * 1e3     #photon energy in eV
    asf_q = asf_fit_parser(asf_fit)
    asf_0 = asf_parser(asf_hw)(en) - asf_q(0)
    return lambda s: asf_0 + asf_q(s)

def gaussian(x, y, z, waist=1e-4, wavelength=1.5e-7):
    """
    Return Gaussian beam amplitude for given coordinate (x, y, z).

    x, y, z - coordinates
    waist - beam waist radius
    wavelength - light wavelength
    """
    k = 2 * np.pi / wavelength
    zr = k * waist**2 / 2
    wz = waist * sqrt(1.0 + (z / zr)**2)
    R =  + zr**2 / z
    return sqrt(2 / np.pi) * wz**-1 * np.exp(-(x * x + y * y) / wz**2) * np.exp(-k * z * 1j + np.arctan(z / zr) * 1j) * np.exp(-k * (x * x + y * y) * 1j / 2 / R)

def kin(x, y, z, waist=1e-4, wavelength=1.5e-7):
    """
    Return incoming wavevector of gaussian beam for given coordinate (x, y, z).

    waist - beam waist radius
    wavelength - light wavelength

    Return a list of three incoming wavevector coordinates.
    """
    k = 2 * np.pi / wavelength
    zr = k * waist**2 / 2
    R = z + zr**2 / z
    return np.array([x, y, R]) / sqrt(x * x + y * y + R * R)

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
    return [[kx, ky] for kx in x_det / det_dist for ky in y_det / det_dist]

def kout_ext(kx, ky):
    return [kx, ky, sqrt(1 - kx**2 - ky**2)]

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
    assert len(lat_orig) ==3, 'origin argument is invalid, it must have 3 values'
    xval = a * np.arange((-Nx + 1) / 2.0, (Nx + 1) / 2.0)
    yval = b * np.arange((-Ny + 1) / 2.0, (Ny + 1) / 2.0)
    zval = c * np.arange((-Nz + 1) / 2.0, (Nz + 1) / 2.0)
    return np.add([[x, y, z] for x in xval for y in yval for z in zval], lat_orig)

def diff_list(kouts, lat_pts, asf, waist, sigma, wavelength=1.5e-7):
    """
    Return diffraction pattern for given array of output wavevectors.

    kouts - tuple of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    asf - atomic scattering factor for atoms in the sample
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return list of diffracted wave values for given kouts.
    """
    diffs = []
    us = np.array([gaussian(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    kins = np.array([kin(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    for kout in kouts:
        qs = np.add(-1 * kins, kout_ext(*kout)) / 2.0 / wavelength / 1e7
        asfs = np.array([asf(sqrt(absv)) for absv in (qs * qs).sum(axis=1)])
        exps = np.exp(2 * np.pi / wavelength * np.dot(lat_pts, kout_ext(*kout)) * 1j)
        diffs.append(sqrt(sigma) * constants.value('classical electron radius') * 1e3 * np.sum(asfs * us * exps))
    return diffs

def diff_grid(kxs, kys, lat_pts, asf, waist, sigma, wavelength=1.5e-7):
    """
    Return diffraction pattern intensity for given array of output wavevectors.

    kxs - x coordinates of output wavevectors
    kys - y coordinates of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    asf - atomic scattering factor for atoms in the sample
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Return np.array of diffracted wave values with the same shape as kxs and kys.
    """
    assert kxs.shape == kys.shape, 'kx and ky must have the same shape'
    us = np.array([gaussian(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    kins = np.array([kin(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    it = np.nditer([kxs, kys, None], op_flags = [['readonly'], ['readonly'], ['writeonly', 'allocate']], op_dtypes = ['float64', 'float64', 'complex128'])
    for kx, ky, diff in it:
        kout = [kx, ky, sqrt(1 - kx**2 - ky**2)]
        qs = np.add(-1 * kins, kout) / 2.0 / wavelength / 1e7
        asfs = np.array([asf(sqrt(absv)) for absv in (qs * qs).sum(axis=1)])
        exps = np.exp(2 * np.pi / wavelength * np.dot(lat_pts, kout) * 1j)
        diff[...] = sqrt(sigma) * constants.value('classical electron radius') * 1e3 * np.sum(asfs * us * exps)
    return it.operands[-1]

def make_grid(kouts, funcvals=None):
    """
    Return grid of coordinates (like in numpy.meshgrid) and corresponding function values grid based on list of points pts and list of function values funcvals.
    pts should be sorted the same way as nditer iterates through the grid array!

    pts - list of points
    funcvals - function values

    Return grid array for every axis and a grid of function values.
    """
    coords = map(np.unique, np.array(kouts).T)
    grid = np.meshgrid(*coords)
    funcgrid = np.zeros(grid[0].shape, dtype='complex128')
    it = np.nditer(funcgrid, flags = ['f_index'], op_flags = ['writeonly'], op_dtypes = ['complex128'])
    for f in it:
        f[...] = funcvals[it.index]
    grid.append(funcgrid)
    return tuple(grid)

def diff_gen(kouts, lat_pts, asf, waist, sigma, wavelength=1.5e-7):
    """
    Yield diffraction pattern for given array of output wavevectors.

    kouts - tuple of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    asf - atomic scattering factor for atoms in the sample
    waist - beam waist radius
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength

    Generator function of diffracted lightwave velues for given kouts.
    """
    us = np.array([gaussian(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    kins = np.array([kin(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    for kout in kouts:
        qs = np.add(-1 * kins, kout) / 2.0 / wavelength / 1e7
        asfs = np.array([asf(sqrt(absv)) for absv in (qs * qs).sum(axis=1)])
        exps = np.exp(2 * np.pi / wavelength * np.dot(lat_pts, kout_ext(*kout)) * 1j)
        yield sqrt(sigma) * constants.value('classical electron radius') * 1e3 * np.sum(asfs * us * exps)

def diff_work(kout, lat_pts, kins, us, asf_hw, asf_fit, sigma, wavelength):
    """
    Worker function for difraction pattern for multiprocessing.

    kouts - tuple of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    kins - list of incoming wavevectors
    us - list of gaussian beam wave values
    asf_hw, asf_fit - atomic scattering files (see asf function)
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength
    """
    _asf = asf(asf_hw, asf_fit, wavelength)
    qs = np.add(-kins, kout_ext(*kout)) / 2.0 / wavelength / 1e7
    asfs = np.array([_asf(sqrt(absv)) for absv in (qs * qs).sum(axis=1)])
    exps = np.exp(2 * np.pi / wavelength * np.dot(lat_pts, kout_ext(*kout)) * 1j)
    return sqrt(sigma) * constants.value('classical electron radius') * 1e3 * np.sum(asfs * us * exps)

def selftest(filename, filename_fit):
    """
    Plot atomic scattering factor as well it's analytical fit.
    """
    import matplotlib.pyplot as plt
    asf = asf_parser(filename)
    asf_fit = asf_fit_parser(filename_fit)
    x = np.linspace(0, 6, num=101, endpoint=True)
    plt.plot(x, asf(x), 'r-', x, asf_fit(x), 'b-')
    plt.show()

if __name__ == "__main__":
    selftest('asf/asf_q_Au.txt', 'asf/asf_q_fit_Au.txt')