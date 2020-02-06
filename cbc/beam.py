"""
beam.py - incoming beam classes module
"""
import concurrent.futures
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import jv
from scipy.integrate import quad
from . import utils

class ABCBeam(object):
    __metaclass__ = ABCMeta

    def __init__(self, wavelength):
        self.wavelength, self.k = wavelength, 2 * np.pi / wavelength

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

    def __init__(self, waist, wavelength):
        super(Beam, self).__init__(wavelength)
        self.waist, self.zr = waist, np.pi * self.waist**2 / self.wavelength
        self.thdiv = self.wavelength / np.pi / self.waist

    @abstractmethod
    def amplitude(self, xs, ys, zs): pass

    @abstractmethod
    def dist(self, N): pass

    def _write_args(self, outfile):
        outfile.create_dataset('waist', data=self.waist)

    def wave(self, xs, ys, zs):
        return self.amplitude(xs, ys, zs) * utils.phase_inc(self.wavevectors(xs, ys, zs), xs, ys, zs, self.wavelength)

class GausBeam(Beam):
    """
    Gaussian beam class.

    waist - beam waist radius
    wavelength - light wavelength
    """
    def amplitude(self, xs, ys, zs):
        wz = self.waist * np.sqrt(1 + zs**2 / self.zr**2)
        return np.pi**-1 * self.waist**-1 * wz**-1 * np.exp(-(xs**2 + ys**2) / wz**2)

    def wavevectors(self, xs, ys, zs):
        rs = zs + self.zr**2 / zs
        return np.stack((xs / rs, ys / rs, 1 - (xs**2 + ys**2) / 2.0 / rs**2), axis=-1)

    def dist(self, N):
        k_xs, k_ys = np.random.multivariate_normal([0, 0], [[self.thdiv**2 / 2, 0],
                                                            [0, self.thdiv**2 / 2]], N).T
        return np.stack((k_xs, k_ys, 1.0 - (k_xs**2 + k_ys**2) / 2.0), axis=1)

    def fphase(self, k_xs, k_ys, z):
        return np.exp(-1j * self.k * z * (1.0 - (k_xs**2 + k_ys**2) / 2.0))

class BesselBeam(Beam):
    """
    Bessel beam class.

    waist - beam waist radius
    wavelength - light wavelength
    """
    def amplitude(self, xs, ys, zs):
        rs = np.sqrt(xs**2 + ys**2)
        return jv(1, self.k * self.thdiv * rs) / self.thdiv / np.pi / rs

    def wavevectors(self, xs, ys, zs):
        return np.tile([0.0, 0.0, 1.0 - self.thdiv**2 / 2], xs.shape + (1,))

    def dist(self, N):
        ths = self.thdiv * np.sqrt(np.random.rand(N))
        phis = 2 * np.pi * np.random.rand(N)
        return np.stack((ths * np.cos(phis), ths * np.sin(phis), 1 - ths**2 / 2), axis=-1)

class LensBeam(ABCBeam):
    __metaclass__ = ABCMeta

    def __init__(self, focus, aperture, wavelength):
        super(LensBeam, self).__init__(wavelength)
        self.focus, self.aperture = focus, aperture

    @abstractmethod
    def worker(self, xs, ys, zs): pass

    def _write_args(self, outfile):
        outfile.create_dataset('focus', data=self.focus)
        outfile.create_dataset('aperture', data=self.aperture)

    def wave(self, xs, ys, zs):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futs = []
            for xchunk, ychunk, zchunk in zip(np.array_split(xs, utils.CORES_NUM),
                                              np.array_split(ys, utils.CORES_NUM),
                                              np.array_split(zs, utils.CORES_NUM)):
                futs.append(executor.submit(self.worker, xchunk, ychunk, zchunk))
        return np.concatenate([fut.result() for fut in futs])

    def wavevectors(self, xs, ys, zs):
        rs = np.sqrt(xs**2 + ys**2 + zs**2)
        return np.stack((xs / rs, ys / rs, 1 - (xs**2 + ys**2) / 2.0 / rs**2), axis=-1)

class RectLens(LensBeam):
    """
    Rectangular aperture lens beam class.

    f - focal length
    ap - half aperture size
    wavelength - light wavelength
    """
    def worker(self, xs, ys, zs):
        coeffs = -1j * np.exp(1j * self.k * (zs + self.focus)) / self.wavelength / (zs + self.focus) * np.exp(1j * self.k / 2.0 / (zs + self.focus) * (xs**2 + ys**2))
        x_vals, y_vals = [], []
        for x_val, y_val, z_val in zip(xs.ravel(), ys.ravel(), zs.ravel()):
            limit = int(2.0 * self.aperture / np.sqrt(2.0 * self.wavelength * abs(z_val)))
            x_int_re = utils.llc_rectlens_re(x_val, z_val + self.focus, self.k, self.focus)
            x_int_im = utils.llc_rectlens_im(x_val, z_val + self.focus, self.k, self.focus)
            y_int_re = utils.llc_rectlens_re(y_val, z_val + self.focus, self.k, self.focus)
            y_int_im = utils.llc_rectlens_im(y_val, z_val + self.focus, self.k, self.focus)
            x_quad_re = quad(x_int_re, -self.aperture, self.aperture, limit=limit)[0]
            x_quad_im = quad(x_int_im, -self.aperture, self.aperture, limit=limit)[0]
            y_quad_re = quad(y_int_re, -self.aperture, self.aperture, limit=limit)[0]
            y_quad_im = quad(y_int_im, -self.aperture, self.aperture, limit=limit)[0]
            x_vals.append(x_quad_re + x_quad_im * 1j)
            y_vals.append(y_quad_re + y_quad_im * 1j)
        return coeffs * np.array(x_vals).reshape(xs.shape) * np.array(y_vals).reshape(ys.shape)

class CircLens(LensBeam):
    """
    Circular aperture lens beam class.

    f - focal length
    ap - half aperture size
    wavelength - light wavelength
    """
    def worker(self, xs, ys, zs):
        coeffs = -1j * np.exp(1j * self.k * (zs + self.focus)) / self.wavelength / (zs + self.focus) * np.exp(1j * self.k * (xs**2 + ys**2) / 2.0 / (zs + self.focus))
        r_vals = []
        for x_val, y_val, z_val in zip(xs.ravel(), ys.ravel(), zs.ravel()):
            limit = int(2.0 * self.aperture / np.sqrt(2.0 * self.wavelength * abs(z_val)))
            r_val = np.sqrt(x_val**2 + y_val**2)
            int_re = utils.llc_circlens_re(r_val, z_val + self.focus, self.k, self.focus)
            int_im = utils.llc_circlens_im(r_val, z_val + self.focus, self.k, self.focus)
            quad_re = quad(int_re, 0, self.aperture, limit=limit)[0]
            quad_im = quad(int_im, 0, self.aperture, limit=limit)[0]
            r_vals.append(quad_re + quad_im * 1j)
        return coeffs * np.array(r_vals).reshape(xs.shape)
