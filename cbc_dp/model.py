"""
model.py - convergent beam diffraction forward model
"""
from math import sqrt
from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import ma
from . import utils
from .feat_detect import RecVectors

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

    @property
    def kout(self):
        """
        Return output wave vectors of source points
        """
        return self.source + self.rec_vec

    def det_pts(self, exp_set):
        """
        Return detector diffraction orders points for a detector at given distance

        exp_set - ExperrimentSettings class object
        """
        k_out = self.kout
        det_x = exp_set.det_pos[2] * np.tan(np.arccos(k_out[..., 2])) * np.cos(np.arctan2(k_out[..., 1], k_out[..., 0]))
        det_y = exp_set.det_pos[2] * np.tan(np.arccos(k_out[..., 2])) * np.sin(np.arctan2(k_out[..., 1], k_out[..., 0]))
        return (np.stack((det_x, det_y), axis=-1) + exp_set.det_pos[:2]) / exp_set.pix_size

class IndexLattice(RecLattice):
    """
    Voting reciprocal lattice points class used in indexing algorithm

    rec_basis - reciprocal lattice basis vectors
    exp_vec - experimental scattering vectors
    num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axis
    """
    sol_m = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]]])

    def __init__(self, rec_basis, num_ap, exp_vec):
        self.num_ap, self.exp_vec = num_ap, exp_vec
        super(IndexLattice, self).__init__(rec_basis)
        self.init_source_lines()

    def init_source(self):
        """
        Initialize source line origin points
        """
        self.rec_abs, self.rec_th, self.rec_phi, self.source = utils.source_index(self.rec_vec)

    def hkl_box(self):
        """
        Return hkl indices voting box based on numerical aperture of the lens
        """
        h_vals = np.arange(0, self.box_size[0])
        k_vals = np.arange(0, self.box_size[1])
        l_vals = np.arange(0, self.box_size[2])
        h_grid, k_grid, l_grid = np.meshgrid(h_vals, k_vals, l_vals)
        return np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=-1)

    def init_rec_vectors(self):
        """
        Initialize voting lattice scattering vectors
        """
        na_box = np.array([self.num_ap[0], self.num_ap[1], self.num_ap.max()**2 / 2])
        self.box_size = 2 * np.ceil(np.abs(na_box.dot(self.inv_basis)))
        hkl_origin = np.floor(self.exp_vec.dot(self.inv_basis)) - np.ones(3) * (self.box_size - 2) // 2
        self.hkl_idxs = hkl_origin[:, None] + self.hkl_box()[None]
        self.rec_vec = self.hkl_idxs.dot(self.rec_basis)

    def init_source_lines(self):
        """
        Mask lattice scattering vectors based on rectangular aperture lens confinement
        """
        self.source_lines = utils.source_lines(source=self.source,
                                               rec_vec=self.rec_vec,
                                               num_ap_x=self.num_ap[0],
                                               num_ap_y=self.num_ap[1])

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

        exp_set - ExperrimentSettings class object
        """
        k_out = self.kout_lines()
        det_x = exp_set.det_pos[2] * np.tan(np.arccos(k_out[..., 2])) * np.cos(np.arctan2(k_out[..., 1], k_out[..., 0]))
        det_y = exp_set.det_pos[2] * np.tan(np.arccos(k_out[..., 2])) * np.sin(np.arctan2(k_out[..., 1], k_out[..., 0]))
        return (np.stack((det_x, det_y), axis=-1) + exp_set.det_pos[:2]) / exp_set.pix_size

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
    sol_m = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]]])

    def __init__(self, rec_basis, num_ap, q_max):
        self.num_ap = num_ap
        super(RectModel, self).__init__(rec_basis, q_max)
        self.apply_mask(np.abs(np.sin(self.rec_th - np.arccos(self.rec_abs / 2))) < np.sqrt(self.num_ap[0]**2 + self.num_ap[1]**2))
        self.bounds = np.array([[[self.num_ap[0], 0],
                                 [-self.num_ap[0], 0],
                                 [0, self.num_ap[1]],
                                 [0, -self.num_ap[1]]]])
        self.init_source_lines()

    def init_source_lines(self):
        """
        Derive source lines of a rectangular aperture lens
        """
        coeff1 = (self.source * self.rec_vec).sum(axis=-1)[..., None] - (self.bounds * self.rec_vec[..., None, :2]).sum(axis=-1)
        coeff2 = np.stack((self.rec_vec[..., 1], self.rec_vec[..., 1], self.rec_vec[..., 0], self.rec_vec[..., 0]), axis=-1)
        alpha = coeff2**2 + self.rec_vec[..., None, 2]**2
        betta = coeff2 * coeff1
        gamma = coeff1**2 - self.rec_vec[..., None, 2]**2 * (1 - self.bounds.sum(axis=2)**2)
        delta = betta**2 - alpha * gamma

        solution = np.concatenate((self.bounds + self.sol_m * ((betta + ma.sqrt(delta)) / alpha)[..., None],
                                   self.bounds + self.sol_m * ((betta - ma.sqrt(delta)) / alpha)[..., None]), axis=1)
        solution = np.stack((solution[..., 0],
                             solution[..., 1],
                             np.sqrt(1 - solution[..., 0]**2 - solution[..., 1]**2)), axis=-1)
        ort_mask = np.abs(((self.source[:, None] - solution) * self.rec_vec[:, None]).sum(axis=-1)) < 1e-6
        sol_mask = (np.abs(solution[..., 0]) <= self.num_ap[0]) & (np.abs(solution[..., 1]) <= self.num_ap[1]) & ort_mask

        self.apply_mask(sol_mask.any(axis=1))
        self._source_lines = solution[sol_mask].reshape((-1, 2, 3))

    def source_lines(self):
        """
        Return source lines of a rectangular aperture lens
        """
        return self._source_lines

def qindex_point(rec_basis):
    """
    Return QIndexTF class point based on a reciprocal basis array
    """
    basis_sizes = np.sqrt((rec_basis**2).sum(axis=1))
    or_mat = rec_basis / basis_sizes[:, None]
    return np.concatenate((basis_sizes, np.arccos(or_mat).ravel()))

class FullTF():
    """
    Indexing solution target function class based on scattering vectors error

    lines - detected diffraction streaks positions at the detector [pixels]
    num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axis

    Point is a flattened array of basis vectors lengths and orientation matrix
    """
    mat_shape = (3, 3)
    pix_size = 75 * 1e-3

    def __init__(self, lines, num_ap):
        self.lines, self.num_ap = lines, num_ap
        self.na_size = np.sqrt((num_ap**2).sum())

    @property
    def kin(self):
        return np.tile(np.array([0, 0, 1]), (self.lines.shape[0], 2, 1))

    def exp_vec(self, point):
        """
        Return outcoming wavevectors array of the examined frame
        """
        kout = utils.kout(self.lines, point, self.pix_size)
        return RecVectors(kout=kout, kin=self.kin)
   
    def rec_basis(self, point):
        """
        Return rectangular lattice basis vectors based on the point
        """
        return np.cos(point[6:]).reshape(self.mat_shape) * point[3:6][:, None]
    
    def idxs(self, point):
        """
        Return indices of the best lattice vectors based on the point
        """
        exp_vec = self.exp_vec(point)
        index_lat = IndexLattice(rec_basis=self.rec_basis(point),
                                 exp_vec=exp_vec.scat_vec.mean(axis=1),
                                 num_ap=self.num_ap)
        return utils.fitness_idxs(index_lat.rec_vec, exp_vec.kout)
    
    def __call__(self, point):
        exp_vec = self.exp_vec(point)
        index_lat = IndexLattice(rec_basis=self.rec_basis(point),
                                 exp_vec=exp_vec.scat_vec.mean(axis=1),
                                 num_ap=self.num_ap)
        return utils.fitness(index_lat.rec_vec, exp_vec.kout)

class RotTF(FullTF):
    """
    Indexing solution target function class based on scattering vectors error
    with unchanged relative vector positions implied

    lines - detected diffraction streaks positions at the detector [pixels]
    num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axis
    rec_basis - initial reciprocal lattice basis vectors

    Point is a flattened array of basis vectors lengths and euler angles [phi1, Phi, phi2]
    Euler angles with Bunge convention are used
    """
    def __init__(self, data, num_ap, rec_basis):
        super(RotTF, self).__init__(data, num_ap)
        self.ref_basis = rec_basis

    def rec_basis(self, point):
        """
        Return orthogonal orientation matrix based on euler angles
        """
        return self.ref_basis.dot(utils.euler_matrix(point[3], point[4], point[5]).T)
    