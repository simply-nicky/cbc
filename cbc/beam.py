from .functions import rbeam, cbeam, lensbeam_kins, gaussian, gaussian_f, gaussian_kins, gaussian_dist, bessel, bessel_kins, uniform_dist
from . import utils
from abc import ABCMeta, abstractmethod, abstractproperty
import concurrent.futures, numpy as np

class ABCBeam(object):
    __metaclass__ = ABCMeta

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
        return lensbeam_kins(xs, ys, zs, self.focus, self.wavelength)

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
        return gaussian(xs, ys, zs, self.waist, self.wavelength)

    def wavevectors(self, xs, ys, zs):
        return gaussian_kins(xs, ys, zs, self.waist, self.wavelength)

    def dist(self, N):
        return gaussian_dist(N, self.waist, self.wavelength)

    def fphase(self, kxs, kys, z):
        return gaussian_f(kxs, kys, z, self.waist, self.wavelength)

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
        return bessel(xs, ys, zs, self.waist, self.wavelength)

    def wavevectors(self, xs, ys, zs):
        return bessel_kins(xs, ys, zs, self.waist, self.wavelength)

    def dist(self, N):
        return uniform_dist(N, self.waist, self.wavelength)

class RectBeam(LensBeam):
    """
    Rectangular aperture lens beam class.

    f - focal length
    ap - half aperture size
    wavelength - light wavelength
    """
    wavelength, focus, aperture = None, None, None

    def __init__(self, focus, aperture, wavelength):
        self.focus, self.aperture, self.wavelength = focus, aperture, wavelength
        
    def worker(self, xs, ys, zs): 
        return rbeam(xs, ys, zs, self.focus, self.aperture, self.wavelength)

class CircBeam(LensBeam):
    """
    Circular aperture lens beam class.

    f - focal length
    ap - half aperture size
    wavelength - light wavelength
    """
    wavelength, focus, aperture = None, None, None

    def __init__(self, focus, aperture, wavelength):
        self.focus, self.aperture, self.wavelength = focus, aperture, wavelength
        
    def worker(self, xs, ys, zs): 
        return cbeam(xs, ys, zs, self.focus, self.aperture, self.wavelength)
