"""
lattice.py - lattice and cell geametries classes module
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from scipy import constants
from scipy.interpolate import interp1d
import numpy as np
import h5py
from . import utils

def rec_basis(a_vec, b_vec, c_vec):
    """
    Return reciprocal basis vectors

    a_vec, b_vec, c_vec - basis vectors
    """
    a_rec = np.cross(b_vec, c_vec) / (np.cross(b_vec, c_vec).dot(a_vec))
    b_rec = np.cross(c_vec, a_vec) / (np.cross(c_vec, a_vec).dot(b_vec))
    c_rec = np.cross(a_vec, b_vec) / (np.cross(a_vec, b_vec).dot(c_vec))
    return a_rec, b_rec, c_rec

class Cell(object):
    """
    Unit cell class.

    atom_xs, atom_ys, atom_zs - atom coordinates within the unit cell
    b_vals - an array of B-factors
    elems - an array of the abbreviations of chemical elements
    """
    def __init__(self, atom_xs=np.zeros(1), atom_ys=np.zeros(1), atom_zs=np.zeros(1), b_vals=np.zeros(1), elems=np.array(['Au'], dtype=object)):
        self.atom_xs, self.atom_ys, self.atom_zs = atom_xs, atom_ys, atom_zs
        self.b_vals, self.elems = b_vals, elems

    @classmethod
    def importpdb(cls, filename):
        return cls(*utils.pdb.importpdb(filename))

    def asf(self, wavelength):
        energy = constants.c * constants.h / constants.e / wavelength * 1e3     #photon energy in eV
        _asf_list = []
        for elem, b_val in zip(self.elems, self.b_vals):
            energies, f1s = utils.asf.henke[elem][0:2]
            _asf_coeffs = np.array([utils.asf.waskif[elem][:5].sum(),
                                    utils.asf.waskif[elem][6:8].mean(),
                                    interp1d(energies, f1s, kind='cubic')(energy) - utils.asf.waskif[elem][:5].sum(),
                                    b_val])
            _asf_list.append(_asf_coeffs)
        return np.array(_asf_list)

    def write(self, outfile):
        cell_group = outfile.create_group(self.__class__.__name__)
        cell_group.create_dataset('elems', data=np.array(self.elems, 'S2'), dtype=h5py.special_dtype(vlen=str))
        cell_group.create_dataset('B-factor', data=self.b_vals)
        coord_group = cell_group.create_group('cell_coord')
        coord_group.create_dataset('x-coordinate', data=self.atom_xs)
        coord_group.create_dataset('y-coordinate', data=self.atom_ys)
        coord_group.create_dataset('z-coordinate', data=self.atom_zs)

class ABCLattice(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def basis_a(self): pass

    @abstractproperty
    def basis_b(self): pass

    @abstractproperty
    def basis_c(self): pass

    @abstractproperty
    def lat_na(self): pass

    @abstractproperty
    def lat_nb(self): pass

    @abstractproperty
    def lat_nc(self): pass

    def _pts(self):
        na_rng = np.arange(-self.lat_na, self.lat_na + 1)
        nb_rng = np.arange(-self.lat_nb, self.lat_nb + 1)
        nc_rng = np.arange(-self.lat_nc, self.lat_nc + 1)
        na_vals, nb_vals, nc_vals = np.meshgrid(na_rng, nb_rng, nc_rng)
        return np.multiply.outer(self.basis_a, na_vals) + np.multiply.outer(self.basis_b, nb_vals) + np.multiply.outer(self.basis_c, nc_vals)

class Lattice(ABCLattice):
    @abstractproperty
    def cell(self): pass

    @abstractproperty
    def arguments(self): pass

    @abstractmethod
    def _write_size(self, outfile): pass

    def _write_vectors(self, outfile):
        vec_group = outfile.create_group('lattice_vectors')
        vec_group.create_dataset('a', data=self.basis_a)
        vec_group.create_dataset('b', data=self.basis_b)
        vec_group.create_dataset('c', data=self.basis_c)

    def write(self, outfile):
        lat_group = outfile.create_group(self.__class__.__name__)
        self.cell.write(outfile)
        self._write_vectors(lat_group)
        self._write_size(lat_group)

class CubicLattice(Lattice):
    """
    Cube shaped lattice class.

    basis_a, basis_b, basis_c - unit cell vectors [mm]
    lat_na, lat_nb, lat_nc - numbers of unit cells in a sample
    cell - Cell class instance
    """
    lat_na, lat_nb, lat_nc = None, None, None
    basis_a, basis_b, basis_c, cell = None, None, None, None

    def __init__(self, basis_a, basis_b, basis_c, lat_na=20, lat_nb=20, lat_nc=20, cell=Cell()):
        self.cell = cell
        self.basis_a, self.basis_b, self.basis_c = basis_a, basis_b, basis_c
        self.lat_na, self.lat_nb, self.lat_nc = lat_na // 2, lat_nb // 2, lat_nc // 2

    @property
    def arguments(self):
        return {'lattice_vectors': (self.basis_a, self.basis_b, self.basis_c),
                'lattice_size': (self.lat_na, self.lat_nb, self.lat_nc)}

    def coordinates(self):
        pts = self._pts()
        return (np.add.outer(pts[0].ravel(), self.cell.atom_xs),
                np.add.outer(pts[1].ravel(), self.cell.atom_ys),
                np.add.outer(pts[2].ravel(), self.cell.atom_zs))

    def _write_size(self, outfile):
        size_group = outfile.create_group('lattice_size')
        size_group.create_dataset('Na', data=self.lat_na)
        size_group.create_dataset('Nb', data=self.lat_nb)
        size_group.create_dataset('Nc', data=self.lat_nc)

class BallLattice(Lattice):
    """
    Ball shaped lattice class.
    
    basis_a, basis_b, basis_c - unit cell vectors [mm]
    lat_r - radius of lattice
    cell - Cell class instance
    """
    basis_a, basis_b, basis_c, cell = None, None, None, None
    lat_na, lat_nb, lat_nc = None, None, None

    def __init__(self, basis_a, basis_b, basis_c, lat_r, cell=Cell()):
        self.cell = cell
        self.basis_a, self.basis_b, self.basis_c, self.lat_r = basis_a, basis_b, basis_c, lat_r
        self.lat_na = self.lat_r // np.sqrt(self.basis_a.dot(self.basis_a))
        self.lat_nb = self.lat_r // np.sqrt(self.basis_b.dot(self.basis_b))
        self.lat_nc = self.lat_r // np.sqrt(self.basis_c.dot(self.basis_c))

    @property
    def arguments(self):
        return {'lattice_vectors': (self.basis_a, self.basis_b, self.basis_c),
                'lattice_radius': self.lat_r}

    def coordinates(self):
        pts = self._pts()
        mask = (np.sqrt(pts[0]**2 + pts[1]**2 + pts[2]**2) < self.lat_r)
        return (np.add.outer(pts[0][mask].ravel(), self.cell.atom_xs),
                np.add.outer(pts[1][mask].ravel(), self.cell.atom_ys),
                np.add.outer(pts[2][mask].ravel(), self.cell.atom_zs))

    def _write_size(self, outfile):
        outfile.create_dataset('radius', data=self.lat_r)

class RecLattice(ABCLattice):
    """
    Reciprocal lattice class

    basis_a, basis_b, basis_c - basis vectors of the reciprocal lattice [mm^-1]
    wavelength - light carrier  wavelength [mm]
    q_max - maximum lattice vector in dimensionless units
    """
    basis_a, basis_b, basis_c = None, None, None
    lat_na, lat_nb, lat_nc = None, None, None

    def __init__(self, basis_a, basis_b, basis_c, q_max, wavelength):
        self.basis_a, self.basis_b, self.basis_c = basis_a * wavelength, basis_b * wavelength, basis_c * wavelength
        self.q_max = q_max
        self.lat_na = self.q_max // np.sqrt(self.basis_a.dot(self.basis_a))
        self.lat_nb = self.q_max // np.sqrt(self.basis_b.dot(self.basis_b))
        self.lat_nc = self.q_max // np.sqrt(self.basis_c.dot(self.basis_c))

    def vectors(self):
        pts = self._pts()
        mask = (np.sqrt(pts[0]**2 + pts[1]**2 + pts[2]**2) < self.q_max) & (np.sqrt(pts[0]**2 + pts[1]**2 + pts[2]**2) != 0)
        return (pts[0][mask].ravel(),
                pts[1][mask].ravel(),
                pts[2][mask].ravel(),
                np.sqrt(pts[0]**2 + pts[1]**2 + pts[2]**2)[mask].ravel())
