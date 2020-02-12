"""
model.py - convergent beam diffraction forward model
"""
from math import sqrt
from abc import ABCMeta, abstractmethod
import numpy as np
from . import utils

class ABCLattice(metaclass=ABCMeta):
    """
    Abstract reciprocal lattice in convergent beam diffraction class

    rec_basis - reciprocal lattice basis vectors
    """
    rec_vec, hkl_idxs = None, None

    def __init__(self, rec_basis):
        self.rec_basis, self.inv_basis = rec_basis, np.linalg.inv(rec_basis)
        self.basis_sizes = np.sqrt((rec_basis**2).sum(axis=1))
        self._init_rec_vec()
        self.rec_abs, self.rec_th, self.rec_phi, self.source = utils.init_source(self.rec_vec)

    @abstractmethod
    def _init_rec_vec(self):
        pass

    def kout(self):
        """
        Return output wave vectors of source points
        """
        return self.source + self.rec_vec

    def det_pts(self, exp_set):
        """
        Return detector diffraction orders points for a detector at given distance

        exp_set - FrameSetup class object
        """
        return exp_set.det_pts(self.kout())

class BallLattice(ABCLattice):
    """
    Ball shaped reciprocal lattice class

    rec_basis - reciprocal lattice basis vectors
    q_max - maximax scattering vector norm value
    """
    def __init__(self, rec_basis, q_max):
        self.q_max = q_max
        if self.q_max > 2:
            raise ValueError('q_max must be less than 2')
        super(BallLattice, self).__init__(rec_basis)

    def _init_rec_vec(self):
        """
        Initialize reciprocal lattice vectors
        """
        lat_size = np.rint(self.q_max / self.basis_sizes)
        h_idxs = np.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = np.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = np.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = np.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl_idxs = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)
        rec_vec = hkl_idxs.dot(self.rec_basis)
        rec_abs = np.sqrt((rec_vec * rec_vec).sum(axis=1))
        idxs = np.where((rec_abs != 0) & (rec_abs < self.q_max))
        self.rec_vec, self.hkl_idxs = rec_vec[idxs], hkl_idxs[idxs]

class RecLattice(ABCLattice):
    """
    Reciprocal lattice class

    rec_basis - reciprocal lattice basis vectors
    q_max - maximax scattering vector norm value
    """
    def __init__(self, rec_basis, hkl_idxs):
        self.hkl_idxs = hkl_idxs
        super(RecLattice, self).__init__(rec_basis)

    def _init_rec_vec(self):
        self.rec_vec = self.hkl_idxs.dot(self.rec_basis)

class ABCModel():
    """
    Abstract convergent beam diffraction pattern generator class

    rec_basis - reciprocal lattice basis vectors
    num_ap - convergent beam numerical aperture
    q_max - maximax scattering vector norm value
    """
    mask = None
    masked_attrs = ['rec_vec', 'hkl_idxs', 'rec_abs', 'rec_phi', 'rec_th', 'source']

    def __init__(self, rec_lat, num_ap):
        self.rec_lat, self.num_ap = rec_lat, num_ap
        self._init_mask()

    @abstractmethod
    def _init_mask(self):
        pass

    @abstractmethod
    def source_lines(self):
        pass

    def __getattr__(self, attr):
        if attr in ABCModel.masked_attrs:
            return self.rec_lat.__getattribute__(attr)[self.mask]

    def kout(self):
        """
        Return output wave vectors of source points
        """
        return self.source + self.rec_vec

    def det_pts(self, exp_set):
        """
        Return detector diffraction orders points for a detector at given distance

        exp_set - FrameSetup class object
        """
        return exp_set.det_pts(self.kout())

    def kout_lines(self):
        """
        Return output wave vectors of diffraction streaks
        """
        return self.source_lines() + self.rec_vec[:, None]

    def det_lines(self, exp_set):
        """
        Return detector diffraction orders points for a detector at given distance

        exp_set - FrameSetup class object
        """
        return exp_set.det_pts(self.kout_lines())

class CircModel(ABCModel):
    """
    Circular aperture convergent beam diffraction pattern generator class

    rec_basis - reciprocal lattice basis vectors
    num_ap - convergent beam numerical aperture
    q_max - maximax scattering vector norm value
    """
    def _init_mask(self):
        self.mask = (np.abs(np.sin(self.rec_lat.rec_th - np.arccos(self.rec_lat.rec_abs / 2))) < self.num_ap)

    def source_lines(self):
        """
        Return source lines of a circular aperture lens
        """
        term1 = self.rec_abs**2 + 2 * self.rec_vec[..., 2] * sqrt(1 - self.num_ap**2)
        term2 = 2 * np.sqrt(self.rec_vec[..., 0]**2 + self.rec_vec[..., 1]**2) * self.num_ap
        source_pt1 = np.stack((-self.num_ap * np.cos(self.rec_phi + np.arccos(term1 / term2)),
                               -self.num_ap * np.sin(self.rec_phi + np.arccos(term1 / term2)),
                               np.repeat(sqrt(1 - self.num_ap**2), self.rec_vec.shape[0])), axis=-1)
        source_pt2 = np.stack((-self.num_ap * np.cos(self.rec_phi - np.arccos(term1 / term2)),
                               -self.num_ap * np.sin(self.rec_phi - np.arccos(term1 / term2)),
                               np.repeat(sqrt(1 - self.num_ap**2), self.rec_vec.shape[0])), axis=-1)
        return np.stack((source_pt1, source_pt2), axis=1)

class RectModel(ABCModel):
    """
    Rectangular aperture convergent beam diffraction pattern generator class

    rec_basis - reciprocal lattice basis vectors
    num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axis
    q_max - maximax scattering vector norm value
    """
    def _init_mask(self):
        self._source_lines, self.mask = utils.model_source_lines(self.rec_lat.source,
                                                                 self.rec_lat.rec_vec,
                                                                 self.num_ap[0],
                                                                 self.num_ap[1])

    def source_lines(self):
        """
        Return source lines of a rectangular aperture lens
        """
        return self._source_lines
    