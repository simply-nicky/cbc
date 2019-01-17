"""
File: main.py (Python 3.X) - name and Python versions compatibility are temporal.

Main file for Convergent gaussian beam diffraction crystallography simulation. Every distance is in mm units.
Dependencies: scipy, numpy and matplotlib packages.

Made by Nikolay Ivanov, 2018-2019.
"""

import numpy as np

def asf_parser(filename):
    """
    Input a txt file, that contains atomic scattering factor f [el/atoms] depending on scattering wavevector or photon energy (see https://it.iucr.org/Cb/ch6o1v0001/#table6o1o1o1 for more information).
    The file must contain two columns, the first is the argument and the second is f in el/atoms.

    filename - the filename with atomic scattering factor values, the file must be located in the same folder as this program
    """
    from scipy.interpolate import interp1d
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

    filename - the filename with atomic scattering factor fit coefficients, the file must be located in the same folder as this program.
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
    return lambda s: sum((a * np.exp(-b * s**2) for (a, b) in zip(coefs[0:-1:2], coefs[1:-1:2]))) + coefs[-1]

def asf_advanced(asf_hw, asf_fit, wavelength=1.5e-7):
    """
    Input two txt files, the first one contains atomic scattering factor for different photon energy and the second one contains analytical fit coefficients (see https://it.iucr.org/Cb/ch6o1v0001/#table6o1o1o1 for more information).
    The first file must contain two columns, the first is the argument and the second is f in el/atoms.
    The second file must contain a given list of coefficients: [a1, b1,.. an, bn,.. , c].
    The files must be located in the same folder as this program.

    asf_hw - the filename with atomic scattering factor values for different photon energies
    asf_fit - the filename with analytical fit coefficients.
    wavelength - light wavelength
    """
    from scipy import constants
    asf_f = asf_parser(asf_hw)
    asf_fit_f = asf_fit_parser(asf_fit)
    en = constants.c * constants.h / constants.e / wavelength
    return lambda s: asf_f(en) + asf_fit_f(s) - asf_fit_f(0)

def gaussian(x, y, z, waist=1e-4, wavelength=1.5e-7):
    """
    Return Gaussian beam amplitude for given coordinate (x, y, z).

    x, y, z - coordinates.
    waist - waist radius.
    wavelength - light wavelength.
    """
    k = 2 * np.pi / wavelength
    zr = k * waist**2 / 2
    wz = waist * np.sqrt(1.0 + (z / zr)**2)
    R = z + zr**2 / z
    return np.sqrt(2 / np.pi) * wz**-1 * np.exp(-(x * x + y * y) / wz**2) * np.exp(-k * z * 1j + np.arctan(z / zr) * 1j) * np.exp(-k * (x * x + y * y) * 1j / 2 / R)

def kin(x, y, z, waist=1e-4, wavelength=1.5e-7):
    """
    Return incoming wavevector of gaussian beam for given coordinate (x, y, z).

    waist - beam waist radius.
    wavelength - light wavelength.
    """
    k = 2 * np.pi / wavelength
    zr = k * waist**2 / 2
    R = z + zr**2 / z
    return [x, y, R] / np.sqrt(x * x + y * y + R * R)

def kouts(det_dist, Nx = 512, Ny = 512, pix_size = 55e-3):
    """
    Return output wave vectors array for given detector at given distance from the sample.

    det_dist - distance between detector and sample.
    Nx, Ny - numbers of pixels in x and y axes.
    pix_size - pixel size.

    Return a np.array of x and y coordinates of output wavevectors (kx, ky).
    """
    x_det = np.arange((-Nx + 1) / 2, (Nx + 1) / 2) * pix_size
    y_det = np.arange((-Ny + 1) / 2, (Ny + 1) / 2) * pix_size
    return [[kx, ky] for kx in x_det / det_dist for ky in y_det / det_dist]

def kout_ext(kx, ky):
    return [kx, ky, np.sqrt(1 - kx**2 - ky**2)]

def kout_grid(det_dist, Nx = 512, Ny = 512, pix_size = 55e-3):
    """
    Return output wave vectors array for given detector at given distance from the sample.

    det_dist - distance between detector and sample.
    Nx, Ny - numbers of pixels in x and y axes.
    pix_size - pixel size.

    Return two (Nx, Ny) np.arrays of kx and ky output wavevector coordinates.
    """
    x_det = np.arange((-Nx + 1) / 2, (Nx + 1) / 2) * pix_size
    y_det = np.arange((-Ny + 1) / 2, (Ny + 1) / 2) * pix_size
    return np.meshgrid(x_det / det_dist, y_det / det_dist)  

def lattice(a, b, c, Nx, Ny, Nz, origin=[0, 0, 0]):
    """
    Return atom coordinates of a cristalline sample.

    Nx, Ny, Nz - numbers of unit cells in a sample.
    a, b, c - unit cell edge lengths.

    Return a np.array of all atom positions in a sample.
    """
    assert len(origin) ==3, 'origin argument is invalid, it must have 3 values'
    return [[x + origin[0], y + origin[1], z + origin[2]] for x in np.arange((-Nx + 1) / 2, (Nx + 1) / 2) * a for y in np.arange((-Ny + 1) / 2, (Ny + 1) / 2) * b for z in np.arange((-Nz + 1) / 2, (Nz + 1) / 2) * c]

def diff(kouts, lat_pts, asf, waist, sigma, wavelength=1.5e-7):
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
    from scipy import constants
    diffs = []
    us = np.array([gaussian(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    kins = np.array([kin(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    for kout in kouts:
        asfs = np.array([asf(np.linalg.norm(kout_ext(*kout) - kin) / 2 / wavelength) for kin in kins]) 
        exps = np.array([np.exp(2 * np.pi / wavelength * np.dot(kout_ext(*kout), pt) * 1j) for pt in lat_pts])
        vec = asfs * us * exps
        diffs.append(np.sqrt(sigma) * constants.value('classical electron radius') * 1e3 * vec.sum())
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

    Return np.array of diffracted wave values with the same shape as kxs, kys and kzs.
    """
    assert kxs.shape == kys.shape, 'kx and ky must have the same shape'
    from scipy import constants
    us = np.array([gaussian(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    kins = np.array([kin(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    it = np.nditer([kxs, kys, None], op_flags = [['readonly'], ['readonly'], ['writeonly', 'allocate']], op_dtypes = ['float64', 'float64', 'complex128'])
    for kx, ky, diff in it:
        kout = [kx, ky, np.sqrt(1 - kx**2 - ky**2)]
        asfs = np.array([asf(np.linalg.norm(kout - kin) / 2 / wavelength) for kin in kins])
        exps = np.array([np.exp(2 * np.pi / wavelength * np.dot(kout, pt) * 1j) for pt in lat_pts])
        vec = asfs * us * exps
        diff[...] = np.sqrt(sigma) * constants.value('classical electron radius') * 1e3 * vec.sum()
    return it.operands[-1]

def make_grid(kouts, funcvals=None):
    """
    Return grid of coordinates (like in numpy.meshgrid) and corresponding function values grid based on list of points pts and list of function values funcvals.
    pts should be sorted the same way as nditer iterates through the grid array!

    pts - list of points
    funcvals - function values

    Return grid array for every axis and function values grid.
    """
    coords = map(np.unique, np.array(kouts).T)
    grid = np.meshgrid(*coords)
    funcgrid = np.zeros(grid[0].shape, dtype='complex128')
    it = np.nditer(funcgrid, flags = ['f_index'], op_flags = ['writeonly'], op_dtypes = ['complex128'])
    for f in it:
        f[...] = funcvals[it.index]
    return grid, funcgrid

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
    from scipy import constants
    us = np.array([gaussian(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    kins = np.array([kin(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    for kout in kouts:
        asfs = np.array([asf(np.linalg.norm(kout_ext(*kout) - kin) / 2 / wavelength) for kin in kins]) 
        exps = np.array([np.exp(2 * np.pi / wavelength * np.dot(kout_ext(*kout), pt) * 1j) for pt in lat_pts])
        vec = asfs * us * exps
        yield np.sqrt(sigma) * constants.value('classical electron radius') * 1e3 * vec.sum()

def diff_work(kout, lat_pts, kins, us, asf_hw, asf_fit, sigma, wavelength):
    """
    Worker function for difraction pattern for multiprocessing.

    kouts - tuple of output wavevectors
    lat_pts - coordinates of sample lattice atoms
    kins - list of incoming wavevectors
    us - list of gaussian beam wave values
    asf_hw, asf_fit - atomic scattering files (see asf_advanced function)
    sigma - the solid angle of a detector pixel
    wavelength - light wavelength
    """
    from scipy import constants
    asf = asf_advanced(asf_hw, asf_fit, wavelength)
    asfs = np.array([asf(np.linalg.norm(kout_ext(*kout) - kin) / 2 / wavelength) for kin in kins]) 
    exps = np.array([np.exp(2 * np.pi / wavelength * np.dot(kout_ext(*kout), pt) * 1j) for pt in lat_pts])
    vec = asfs * us * exps
    return np.sqrt(sigma) * constants.value('classical electron radius') * 1e3 * vec.sum()

def selftest(filename, filename_fit, filename_fit2):
    """
    Plot atomic scattering factor as well it's analytical fit.
    """
    import matplotlib.pyplot as plt
    asf = asf_parser(filename)
    asf_fit = asf_fit_parser(filename_fit)
    asf_fit2 = asf_fit_parser(filename_fit2)
    x = np.linspace(0, 6, num=101, endpoint=True)
    plt.plot(x, asf(x), 'r-', x, asf_fit(x), 'b-', x, asf_fit2(x), 'g-')
    plt.show()

if __name__ == "__main__":
    selftest('asf_q_Au.txt', 'asf_q_fit_Au.txt', 'asf_q_fit_Au_2.txt')