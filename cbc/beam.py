from . import utils
from scipy import special
from math import cos, sin, pi, sqrt
from abc import ABCMeta, abstractmethod, abstractproperty
import concurrent.futures, numpy as np

class ABCBeam(object):
    __metaclass__ = ABCMeta

    @property
    def k(self): return 2 * pi / self.wavelength

    @abstractproperty
    def wavelength(self): pass

    @abstractmethod
    def wave(self, xs, ys, zs): pass

    @abstractmethod
    def wavevectors(self, xs, ys, zs): pass

    @abstractmethod
    def _write_args(self, outfile): pass

    def write(self, outfile):
        beam_group = outfile.create_group(self.__class__.__name__)
        beam_group.create_dataset('wavelength', data=self.wavelength)
        self._write_args(beam_group)

class Beam(ABCBeam):
    __metaclass__ = ABCMeta

    @property
    def zr(self): return pi * self.waist**2 / self.wavelength

    @property
    def thdiv(self): return self.wavelength / pi / self.waist

    @abstractproperty
    def waist(self): pass

    @abstractmethod
    def amplitude(self, xs, ys, zs): pass

    @abstractmethod
    def dist(self, N): pass

    def _write_args(self, outfile):
        outfile.create_dataset('waist', data=self.waist)

    def wave(self, xs, ys, zs):
        return self.amplitude(xs, ys, zs) * utils.phase_inc(self.wavevectors(xs, ys, zs), xs, ys, zs, self.wavelength)

class LensBeam(ABCBeam):
    __metaclass__ = ABCMeta
        
    @abstractproperty
    def focus(self): pass

    @abstractproperty
    def aperture(self): pass

    @abstractmethod
    def worker(self, xs, ys, zs): pass

    def _write_args(self, outfile):
        outfile.create_dataset('focus', data=self.focus)
        outfile.create_dataset('aperture', data=self.aperture)

    def wave(self, xs, ys, zs):
        us = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for u in executor.map(utils.worker_star(self.worker), zip(np.array_split(xs, utils.cores_num), np.array_split(ys, utils.cores_num), np.array_split(zs, utils.cores_num))):
                us.extend(u)
        return np.array(us)

    def wavevectors(self, xs, ys, zs):
        Rs = np.sqrt(xs**2 + ys**2 + zs**2)
        return np.stack((xs / Rs, ys / Rs, 1 - (xs**2 + ys**2) / 2.0 / Rs**2), axis=-1)

class GausBeam(Beam):
    """
    Gaussian beam class.

    waist - beam waist radius
    wavelength - light wavelength
    """
    wavelength, waist = None, None

    def __init__(self, waist, wavelength):
        self.waist, self.wavelength = waist, wavelength

    def amplitude(self, xs, ys, zs):
        wz = self.waist * np.sqrt(1 + zs**2 / self.zr**2)
        return pi**-1 * self.waist**-1 * wz**-1 * np.exp(-(xs**2 + ys**2) / wz**2)

    def wavevectors(self, xs, ys, zs):
        Rs = zs + self.zr**2 / zs
        return np.stack((xs / Rs, ys / Rs, 1 - (xs**2 + ys**2) / 2.0 / Rs**2), axis=-1)

    def dist(self, N):
        kxs, kys = np.random.multivariate_normal([0, 0], [[self.thdiv**2 / 2, 0], [0, self.thdiv**2 / 2]], N).T
        return np.stack((kxs, kys, 1.0 - (kxs**2 + kys**2) / 2.0), axis=1)

    def fphase(self, kxs, kys, z):
        return np.exp(-1j * self.k * z * (1.0 - (kxs**2 + kys**2) / 2.0))

class BesselBeam(Beam):
    """
    Bessel beam class.

    waist - beam waist radius
    wavelength - light wavelength
    """
    wavelength, waist = None, None

    def __init__(self, waist, wavelength):
        self.waist, self.wavelength = waist, wavelength

    def amplitude(self, xs, ys, zs):
        return special.jv(1, self.k * self.thdiv * np.sqrt(xs**2 + ys**2)) / self.thdiv / pi / np.sqrt(xs**2 + ys**2)

    def wavevectors(self, xs, ys, zs):
        return np.tile([0.0, 0.0, 1.0 - self.thdiv**2 / 2], xs.shape + (1,))

    def dist(self, N):
        ths = self.thdiv * np.sqrt(np.random.random(N))
        phis = 2 * pi * np.random.random(N)
        return np.stack((ths * np.cos(phis), ths * np.sin(phis), 1 - ths**2 / 2), axis=-1)

class RectBeam(LensBeam):
    """
    Rectangular aperture lens beam class.

    f - focal length
    ap - half aperture size
    wavelength - light wavelength
    """
    wavelength, focus, aperture = None, None, None

    def rbeam_integrand_re(self, xx, x, z):
        @utils.jit_integrand
        def integrand(xx, x, z):
            return cos(self.k * xx**2 / 2 * (1 / self.focus - 1 / z) + self.k / z * x * xx)
        return integrand

    def rbeam_integrand_im(self, xx, x, z):
        @utils.jit_integrand
        def integrand(xx, x, z):
            return -sin(self.k * xx**2 / 2 * (1 / self.focus - 1 / z) + self.k / z * x * xx)
        return integrand

    def __init__(self, focus, aperture, wavelength):
        self.focus, self.aperture, self.wavelength = focus, aperture, wavelength
        
    def worker(self, xs, ys, zs): 
        coeffs = -1j * np.exp(1j * self.k * (zs + self.focus)) / self.wavelength / (zs + self.focus) * np.exp(1j * self.k / 2.0 / (zs + self.focus) * (xs**2 + ys**2))
        xvals = np.array([utils.quad_complex(self.rbeam_integrand_re, self.rbeam_integrand_im, -self.aperture, self.aperture, args=(x, z + self.focus), limit=int(2.0 * self.aperture / sqrt(2.0 * self.wavelength * abs(z)))) for x, z in zip(xs.ravel(), zs.ravel())]).reshape(xs.shape)
        yvals = np.array([utils.quad_complex(self.rbeam_integrand_re, self.rbeam_integrand_im, -self.aperture, self.aperture, args=(y, z + self.focus), limit=int(2.0 * self.aperture / sqrt(2.0 * self.wavelength * abs(z)))) for y, z in zip(ys.ravel(), zs.ravel())]).reshape(xs.shape)
        return coeffs * xvals * yvals

class CircBeam(LensBeam):
    """
    Circular aperture lens beam class.

    f - focal length
    ap - half aperture size
    wavelength - light wavelength
    """
    wavelength, focus, aperture = None, None, None

    def circ_re(self, rr, r, z):
        @utils.jit_integrand
        def integrand(rr, r, z):
            return cos(self.k * rr**2 / 2 * (1 / self.focus - 1 / z)) * utils.j0(self.k * r * rr / z) * 2 * pi * rr
        return integrand

    def circ_im(self, rr, r, z):
        @utils.jit_integrand
        def integrand(rr, r, z):
            return -sin(self.k * rr**2 / 2 * (1 / self.focus - 1 / z)) * utils.j0(self.k * r * rr / z) * 2 * pi * rr
        return integrand

    def __init__(self, focus, aperture, wavelength):
        self.focus, self.aperture, self.wavelength = focus, aperture, wavelength
        
    def worker(self, xs, ys, zs): 
        coeffs = -1j * np.exp(1j * self.k * (zs + self.focus)) / self.wavelength / (zs + self.focus) * np.exp(1j * self.k * (xs**2 + ys**2) / 2.0 / (zs + self.focus))
        rvals = np.array([utils.quad_complex(self.circ_re, self.circ_im, 0, self.aperture, args=(sqrt(x**2 + y**2), z + self.focus), limit=int(2.0 * self.aperture / sqrt(2.0 * self.wavelength * abs(z)))) for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel())]).reshape(xs.shape)
        return coeffs * rvals
