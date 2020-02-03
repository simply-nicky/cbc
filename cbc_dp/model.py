"""
model.py - convergent beam diffraction forward model
"""
from math import sqrt
from abc import ABCMeta, abstractmethod
import numpy as np
from . import utils

class RecLattice(metaclass=ABCMeta):
    """
    Abstract reciprocal lattice in convergent beam diffraction class

    rec_basis - reciprocal lattice basis vectors
    """
    rec_vec, hkl_idxs, source = None, None, None

    def __init__(self, rec_basis):
        self.rec_basis, self.inv_basis = rec_basis, np.linalg.inv(rec_basis)
        self.basis_sizes = np.sqrt((rec_basis**2).sum(axis=1))
        self.init_rec_vectors()
        self.init_source()

    @abstractmethod
    def init_rec_vectors(self):
        pass

    @abstractmethod
    def init_source(self):
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

class BallLattice(RecLattice):
    """
    Ball shaped reciprocal lattice class

    rec_basis - reciprocal lattice basis vectors
    q_max - maximax scattering vector norm value
    """
    rec_vec, hkl_idxs = None, None

    def __init__(self, rec_basis, q_max):
        self.q_max = q_max
        if self.q_max > 2:
            raise ValueError('q_max must be less than 2')
        super(BallLattice, self).__init__(rec_basis)

    def init_source(self):
        """
        Initialize source line origin points
        """
        self.rec_abs, self.rec_th, self.rec_phi, self.source = utils.source_ball(self.rec_vec)

    def init_rec_vectors(self):
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

class ABCModel(BallLattice):
    """
    Abstract convergent beam diffraction pattern generator class

    rec_basis - reciprocal lattice basis vectors
    num_ap - convergent beam numerical aperture
    q_max - maximax scattering vector norm value
    """
    @abstractmethod
    def source_lines(self):
        pass

    def apply_mask(self, mask):
        """
        Select out reciprocal vectors based on a mask
        """
        self.rec_vec = self.rec_vec[mask]
        self.hkl_idxs = self.hkl_idxs[mask]
        self.rec_abs = self.rec_abs[mask]
        self.rec_th = self.rec_th[mask]
        self.rec_phi = self.rec_phi[mask]
        self.source = self.source[mask]

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
    def __init__(self, rec_basis, num_ap, q_max):
        self.num_ap = num_ap
        super(CircModel, self).__init__(rec_basis, q_max)
        self.apply_mask(np.abs(np.sin(self.rec_th - np.arccos(self.rec_abs / 2))) < self.num_ap)

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

    def __init__(self, rec_basis, num_ap, q_max):
        self.num_ap = num_ap
        super(RectModel, self).__init__(rec_basis, q_max)
        self.apply_mask(np.abs(np.sin(self.rec_th - np.arccos(self.rec_abs / 2))) < np.sqrt(self.num_ap[0]**2 + self.num_ap[1]**2))
        self.init_source_lines()

    def init_source_lines(self):
        """
        Derive source lines of a rectangular aperture lens
        """
        self._source_lines, _sol_mask = utils.model_source_lines(self.source,
                                                                 self.rec_vec,
                                                                 self.num_ap[0],
                                                                 self.num_ap[1])
        self.apply_mask(_sol_mask)

    def source_lines(self):
        """
        Return source lines of a rectangular aperture lens
        """
        return self._source_lines