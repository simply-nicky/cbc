from . import utils
from scipy import constants
from scipy.interpolate import interp1d
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np, h5py
    
class Cell(object):
    """
    Unit cell function arguments class.
    
    XS, YS, ZS - atom coordinates within the unit cell
    bs - an array of B-factors
    elems - an array of the abbreviations of chemical elements
    """
    def __init__(self, XS=np.zeros(1), YS=np.zeros(1), ZS=np.zeros(1), bs=np.zeros(1), elems=['Au']):
        self.XS, self.YS, self.ZS, self.bs, self.elems = XS, YS, ZS, bs, elems

    @classmethod
    def importpdb(cls, filename):
        return cls(*utils.pdb.importpdb(filename))

    def asf(self, wavelength):
        en = constants.c * constants.h / constants.e / wavelength * 1e3     #photon energy in eV
        _asf_list = []
        for elem, b in zip(self.elems, self.bs):
            _asf_coeffs = np.append(utils.asf.waskif[elem], b)
            ens, f1s = utils.asf.henke[elem][0:2]
            _asf_coeffs[5] = interp1d(ens, f1s, kind='cubic')(en) - _asf_coeffs[:5].sum()
            _asf_list.append(_asf_coeffs)
        return np.array(_asf_list)

    def write(self, outfile):
        cell_group = outfile.create_group(self.__class__.__name__)
        cell_group.create_dataset('elems', data=np.array(self.elems, 'S2'), dtype=h5py.special_dtype(vlen=str))
        cell_group.create_dataset('B-factor', data=self.bs)
        coord_group = cell_group.create_group('cell_coord')
        coord_group.create_dataset('x-coordinate', data=self.XS)
        coord_group.create_dataset('y-coordinate', data=self.YS)
        coord_group.create_dataset('z-coordinate', data=self.ZS)

class Lattice(object): 
    __metaclass__ = ABCMeta
    lat_orig = np.zeros(3)

    @abstractproperty
    def a(self): pass
    
    @abstractproperty
    def b(self): pass
    
    @abstractproperty
    def c(self): pass

    @abstractproperty
    def cell(self): pass

    @abstractproperty
    def arguments(self): pass

    @abstractmethod
    def coordinates(self): pass
    
    @abstractmethod
    def _write_size(self, outfile): pass

    def _write_vectors(self, outfile):
        vec_group = outfile.create_group('lattice_vectors')
        vec_group.create_dataset('a', data=self.a)
        vec_group.create_dataset('b', data=self.b)
        vec_group.create_dataset('c', data=self.c)
    
    def write(self, outfile):
        lat_group = outfile.create_group(self.__class__.__name__)
        self.cell.write(outfile)
        self._write_vectors(lat_group)
        self._write_size(lat_group)

class CubicLattice(Lattice):
    """
    Lattice function arguments class.
    
    Nx, Ny, Nz - numbers of unit cells in a sample
    a, b, c - unit cell edge lengths
    lat_orig - lattice origin point
    """
    a, b, c, cell = None, None, None, None

    def __init__(self, cell=Cell(), a=np.array([7.9e-6, 0, 0]), b=np.array([0, 7.9e-6, 0]), c=np.array([0, 0, 3.8e-6]), Nx=20, Ny=20, Nz=20):
        self.cell = cell
        self.a, self.b, self.c = a, b, c
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz

    @property
    def arguments(self):
        return {'lattice_vectors': (self.a, self.b, self.c), 'lattice_size': (self.Nx, self.Ny, self.Nz), 'lattice_origin': self.lat_orig}

    def coordinates(self):
        nxval = np.arange((-self.Nx + 1) / 2., (self.Nx + 1) / 2.)
        nyval = np.arange((-self.Ny + 1) / 2., (self.Ny + 1) / 2.)
        nzval = np.arange((-self.Nz + 1) / 2., (self.Nz + 1) / 2.)
        nx, ny, nz = np.meshgrid(nxval, nyval, nzval)
        pts = np.multiply.outer(self.a, nx) + np.multiply.outer(self.b, ny) + np.multiply.outer(self.c, nz)
        return np.add.outer(pts[0].ravel(), self.cell.XS), np.add.outer(pts[1].ravel(), self.cell.YS), np.add.outer(pts[2].ravel(), self.cell.ZS)

    def _write_size(self, outfile):
        size_group = outfile.create_group('lattice_size')
        size_group.create_dataset('Nx', data=self.Nx)
        size_group.create_dataset('Ny', data=self.Ny)
        size_group.create_dataset('Nz', data=self.Nz)

class BallLattice(Lattice):
    """
    Lattice function arguments class.
    
    Nx, Ny, Nz - numbers of unit cells in a sample
    a, b, c - unit cell edge lengths
    lat_orig - lattice origin point
    """
    a, b, c, cell = None, None, None, None

    def __init__(self, cell=Cell(), a=np.array([7.9e-6, 0, 0]), b=np.array([0, 7.9e-6, 0]), c=np.array([0, 0, 3.8e-6]), r=1e-4):
        self.cell = cell
        self.a, self.b, self.c, self.r = a, b, c, r

    @property
    def arguments(self):
        return {'lattice_vectors': (self.a, self.b, self.c), 'lattice_radius': self.r, 'lattice_origin': self.lat_orig}

    def coordinates(self):
        Nx, Ny, Nz = self.r / np.sqrt(self.a.dot(self.a)), self.r / np.sqrt(self.b.dot(self.b)), self.r / np.sqrt(self.c.dot(self.c))
        nxval, nyval, nzval = np.arange(-Nx, Nx), np.arange(-Ny, Ny), np.arange(-Nz, Nz)
        nx, ny, nz = np.meshgrid(nxval, nyval, nzval)
        pts = np.multiply.outer(self.a, nx) + np.multiply.outer(self.b, ny) + np.multiply.outer(self.c, nz)
        mask = (np.sqrt(pts[0]**2 + pts[1]**2 + pts[2]**2) < self.r)
        return np.add.outer(pts[0][mask].ravel(), self.cell.XS), np.add.outer(pts[1][mask].ravel(), self.cell.YS), np.add.outer(pts[2][mask].ravel(), self.cell.ZS)

    def _write_size(self, outfile):
        outfile.create_dataset('radius', data=self.r)
