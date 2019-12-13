"""
model.py - convergent beam diffraction forward model
"""
from math import sqrt
from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import ma
from . import utils

class RecLattice(metaclass=ABCMeta):
    """
    Abstract reciprocal lattice in convergent beam diffraction class

    rec_basis - reciprocal lattice basis vectors
    """
    rec_vec, hkl_idxs = None, None

    def __init__(self, rec_basis):
        self.rec_basis, self.inv_basis = rec_basis, np.linalg.inv(rec_basis)
        self.basis_sizes = np.sqrt((rec_basis**2).sum(axis=1))
        self.init_rec_vectors()
        self.init_source()

    @abstractmethod
    def init_rec_vectors(self):
        pass

    def init_source(self):
        """
        Initialize source line origin points
        """
        self.rec_abs = ma.sqrt((self.rec_vec**2).sum(axis=-1))
        self.rec_th = ma.arccos(-self.rec_vec[..., 2] / self.rec_abs)
        self.rec_phi = np.arctan2(self.rec_vec[..., 1], self.rec_vec[..., 0])
        self.source = ma.stack((-np.sin(self.rec_th - ma.arccos(self.rec_abs / 2)) * np.cos(self.rec_phi),
                                -np.sin(self.rec_th - ma.arccos(self.rec_abs / 2)) * np.sin(self.rec_phi),
                                np.cos(self.rec_th - ma.arccos(self.rec_abs / 2))), axis=-1)

    def kout_pts(self):
        """
        Return output wave vectors of source points
        """
        return self.source + self.rec_vec

    def det_pts(self, exp_set):
        """
        Return detector diffraction orders points for a detector at given distance

        exp_set - ExperrimentSettings class object
        """
        k_out = self.kout_pts()
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
        self.bounds = np.array([[[self.num_ap[0], 0],
                                 [-self.num_ap[0], 0],
                                 [0, self.num_ap[1]],
                                 [0, -self.num_ap[1]]]])
        super(IndexLattice, self).__init__(rec_basis)
        self.init_source_lines()

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
        coeff1 = (self.source * self.rec_vec).sum(axis=-1)[..., None] - (self.bounds * self.rec_vec[..., None, :2]).sum(axis=-1)
        coeff2 = ma.stack((self.rec_vec[..., 1], self.rec_vec[..., 1], self.rec_vec[..., 0], self.rec_vec[..., 0]), axis=-1)
        alpha = coeff2**2 + self.rec_vec[..., None, 2]**2
        betta = coeff2 * coeff1
        gamma = coeff1**2 - self.rec_vec[..., None, 2]**2 * (1 - self.bounds.sum(axis=-1)**2)
        delta = betta**2 - alpha * gamma

        self.source_lines = ma.concatenate((self.bounds + self.sol_m * ((betta + ma.sqrt(delta)) / alpha)[..., None],
                                            self.bounds + self.sol_m * ((betta - ma.sqrt(delta)) / alpha)[..., None]), axis=-2)

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
        delta_mask = (delta > 0).all(axis=-1)

        self.apply_mask(delta_mask)
        alpha, betta, delta = alpha[delta_mask], betta[delta_mask], delta[delta_mask]

        solution = np.concatenate((self.bounds + self.sol_m * ((betta + np.sqrt(delta)) / alpha)[..., None],
                                   self.bounds + self.sol_m * ((betta - np.sqrt(delta)) / alpha)[..., None]), axis=1)
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

class GradientDescent():
    """
    Gradient Descent algorithm class

    or_mat - reciprocal lattice orientation matrix
    center - reciprocal lattice center
    target_func - target function to minimize
    step_size - step size relative to point norm
    """
    def __init__(self, target_func, step_size=1e-6):
        self.trg_func, self.step_size = target_func, step_size

    def next_point(self, point):
        """
        Return next iterative point
        """
        next_point = point - self.step_size * self.trg_func.gradient(point)
        return next_point

    def run(self, start_point, max_iter=1000):
        """
        Return gradient descent result after a number of iterations

        start_point - starting point
        max_iter - number of iterations
        """
        points = [start_point]
        values = [self.trg_func.values(start_point)]
        while len(points) < max_iter:
            next_pt = self.next_point(points[-1])
            points.append(next_pt)
            values.append(self.trg_func.values(next_pt))
        return points, values

class TargetFunction(metaclass=ABCMeta):
    """
    Abstract Target Function class

    data - experimental data
    step_size - step size in numerical derivation
    """
    def __init__(self, data, step_size):
        self.data, self.step_size = data, step_size
        self._init_axes()

    def _init_axes(self):
        self.axes = np.zeros((self.step_size.size,) + self.step_size.shape)
        idxs = np.unravel_index(np.arange(0, self.axes.size, self.step_size.size + 1), self.axes.shape)
        self.axes[idxs] = 1

    def num_deriv(self, point, axis):
        """
        Calculate numerically target function partial derivative along the axis

        axis - axis of differentiation
        """
        step = self.step_size * axis
        d_func = self(point + step) - self(point - step)
        d_arg = 2 * step.sum()
        return d_func / d_arg

    def gradient(self, point):
        """
        Return target function gradient at a given point
        """
        grad = np.array([self.num_deriv(point, axis) for axis in self.axes])
        return grad.reshape(point.shape)

    def __call__(self, point):
        """
        Return target function value at the given point
        """
        return np.mean(self.values(point))

    @abstractmethod
    def values(self, point):
        pass

class IndexTF(TargetFunction, metaclass=ABCMeta):
    """
    Abstract indexing solution target function class

    data - experimental data
    step_size - step size in numerical derivation

    Point is a flattened array of basis vectors lengths and orientation matrix angles
    """
    mat_shape = (3, 3)

    def __init__(self, data, num_ap, step_size):
        super(IndexTF, self).__init__(data, step_size)
        self.num_ap = num_ap

    def rec_basis(self, point):
        """
        Return rectangular lattice basis vectors based on the point
        """
        return np.cos(point[3:]).reshape(self.mat_shape) * point[:3][:, None]

    @abstractmethod
    def values(self, point):
        pass

def qindex_point(rec_basis):
    """
    Return QIndexTF class point based on a reciprocal basis array
    """
    basis_sizes = np.sqrt((rec_basis**2).sum(axis=1))
    or_mat = rec_basis / basis_sizes[:, None]
    return np.concatenate((basis_sizes, np.arccos(or_mat).ravel()))

def orthqindex_point(rec_basis):
    """
    Return QIndexTF class point based on a reciprocal basis array
    """
    ort_basis = utils.gramm_schmidt(rec_basis)
    basis_sizes = np.sqrt((ort_basis**2).sum(axis=1))
    or_mat = ort_basis / basis_sizes[:, None]
    eul_angles = utils.euler_angles(or_mat)
    return np.concatenate((basis_sizes, eul_angles))

class QIndexTF(IndexTF):
    """
    Indexing solution target function class based on scattering vectors error

    data - experimental data
    step_size - step size in numerical derivation

    Point is a flattened array of basis vectors lengths and orientation matrix
    """
    def __init__(self, data, num_ap, step_size=1e-10 * np.ones(12)):
        super(QIndexTF, self).__init__(data, num_ap, step_size)

    def lattice(self, point):
        """
        Return reciprocal lattice object for a given point
        """
        return IndexLattice(rec_basis=self.rec_basis(point),
                            exp_vec=self.data.scat_vec.mean(axis=1),
                            num_ap=self.num_ap)

    def values(self, point):
        """
        Return target function value array for a given point
        """
        index_lat = self.lattice(point)
        norm = index_lat.rec_vec[:, :, None] - self.data.kout[:, None]
        norm_grid = np.abs(1 - np.ma.sqrt((norm**2).sum(axis=-1))).sum(axis=-1)

        pen_x = ma.where(np.abs(index_lat.source_lines[..., 0]) > self.num_ap[0],
                         np.abs(index_lat.source_lines[..., 0]) - self.num_ap[0],
                         0)
        pen_y = ma.where(np.abs(index_lat.source_lines[..., 1]) > self.num_ap[1],
                         np.abs(index_lat.source_lines[..., 1]) - self.num_ap[1],
                         0)

        tf_grid = norm_grid + 10 * (pen_x + pen_y).min(axis=-1)
        return tf_grid.min(axis=-1)

class RotQIndexTF(QIndexTF):
    """
    Indexing solution target function class based on scattering vectors error
    with orthogonal basis vectors implied

    data - experimental data
    step_size - step size in numerical derivation

    Point is a flattened array of basis vectors lengths and euler angles [phi1, Phi, phi2]
    Euler angles with Bunge convention are used
    """
    def __init__(self, data, num_ap, rec_basis, step_size=1e-10 * np.ones(12)):
        super(RotQIndexTF, self).__init__(data, num_ap, step_size)
        self.basis = rec_basis

    def rec_basis(self, point):
        """
        Return orthogonal orientation matrix based on euler angles
        """
        return self.basis.dot(utils.euler_matrix(point[0], point[1], point[2]).T)

# class ExpSetTF(TargetFunction):
#     """
#     Experimental geaometry refinement target function class

#     data - tilt series diffraction streaks positions in detector pixels
#     rec_basis - indexing solution of the first frame in tilt series
#     step_size - step size in numerical derivation

#     Point is as follows: [theta, phi, det_x, det_y, det_z]
#     theta, phi - axis of rotation angles
#     det_x, det_y, det_z - position of detector center in respect to the sample
#     """
#     pix_size = 75 * 1e-3

#     def __init__(self, data, rec_basis, frame_idx=1, step_size=1e-10 * np.ones(5)):
#         det_lines = data[frame_idx].raw_lines.mean(axis=1)
#         super(ExpSetTF, self).__init__(det_lines, step_size)
#         self.rec_basis, self.frame_idx = rec_basis, frame_idx

#     @property
#     def kin(self):
#         return np.tile(np.array([0, 0, 1]), (self.data.shape[0], 1))

#     def rot_axis(self, point):
#         """
#         Return an axis of rotation
#         """
#         return np.array([np.sin(point[0]) * np.cos(point[1]),
#                          np.sin(point[0]) * np.sin(point[1]),
#                          np.cos(point[0])])

#     def rot_matrix(self, point):
#         """
#         Return one degree rotation matrix
#         """
#         return utils.rotation_matrix(self.rot_axis(point), -np.radians(self.frame_idx))

#     def rec_vec(self, point):
#         """
#         Return outcoming wavevectors array of the examined frame
#         """
#         pts = self.data * self.pix_size - point[2:4]
#         angles = np.arctan2(pts[..., 1], pts[..., 0])
#         thetas = np.arctan(np.sqrt(pts[..., 0]**2 + pts[..., 1]**2) / point[4])
#         kout = np.stack((np.sin(thetas) * np.cos(angles),
#                          np.sin(thetas) * np.sin(angles),
#                          np.cos(thetas)), axis=-1)
#         return RecVectors(kout=kout, kin=self.kin)

#     def values(self, point):
#         """
#         Return target function array of values at the given point
#         """
#         rec_basis = self.rec_basis.dot(self.rot_matrix(point).T)
#         q_tf = QIndexTF(self.rec_vec(point))
#         return q_tf.values(qindex_point(rec_basis))
    