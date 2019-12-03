"""
model.py - convergent beam diffraction forward model
"""
from math import sqrt
from abc import ABCMeta, abstractmethod
import numpy as np
from . import utils
from .feat_detect import RecVectors

class RecLattice():
    """
    Reciprocal lattice class

    rec_basis - unit cell vectors (dimensionless)
    """
    def __init__(self, rec_basis):
        self.rec_basis = rec_basis
        self.basis_sizes = np.sqrt((rec_basis**2).sum(axis=1))
        self.inv_basis = np.linalg.inv(self.rec_basis)

    def hkl_grid(self, scat_vec):
        """
        Return a grid of all possible hkl indices of the scattering vectors array

        scat_vec - scattering vectors array of shape (N, 3)
        """
        hkl = scat_vec.dot(self.inv_basis)
        hkl_vals = np.stack((np.ceil(hkl), np.floor(hkl)), axis=1)
        hkl_grid = []
        for hkl in hkl_vals:
            h_idx, k_idx, l_idx = np.meshgrid(hkl[..., 0], hkl[..., 1], hkl[..., 2])
            hkl_grid.append(np.stack((h_idx.ravel(), k_idx.ravel(), l_idx.ravel()), axis=-1))
        return np.stack(hkl_grid)

    def scat_vec(self, scat_vec):
        """
        Return model scattering vectors based on an experimental scattering vectors array

        scat_vec - scattering vectors array of shape (N, 3)
        """
        hkl_idx = self.hkl_grid(scat_vec)
        return hkl_idx.dot(self.rec_basis)

    def sph_vectors(self, q_max):
        """
        Return flattened reciprocal lattice vectors inside a sphere

        q_max - reciprocal sphere radius
        """
        lat_size = np.rint(q_max / self.basis_sizes)
        h_idxs = np.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = np.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = np.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = np.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl_idx = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)
        rec_vec = hkl_idx.dot(self.rec_basis)
        rec_abs = np.sqrt((rec_vec * rec_vec).sum(axis=1))
        idxs = np.where((rec_abs != 0) & (rec_abs < q_max))
        return rec_vec[idxs], hkl_idx[idxs]

class IndexingSolution():
    def __init__(self, rec_basis, hkl_idxs):
        self.rec_basis, self.hkl_idxs = rec_basis, hkl_idxs
        self.qs_model = hkl_idxs.dot(rec_basis)

    @property
    def qabs_model(self):
        return np.sqrt((self.qs_model**2).sum(axis=1))

    def kin(self):
        """
        Return model outcoming wavevectors based on an experimental scattering vectors array

        scat_vec - scattering vectors array of shape (N, 3)
        """
        thetas = np.arccos(-self.qs_model[:, 2] / self.qabs_model)
        phis = np.arctan2(self.qs_model[:, 1], self.qs_model[:, 0])
        kin_x = -np.sin(thetas - np.arccos(self.qabs_model / 2)) * np.cos(phis)
        kin_y = -np.sin(thetas - np.arccos(self.qabs_model / 2)) * np.sin(phis)
        kin_z = np.cos(thetas - np.arccos(self.qabs_model / 2))
        return np.stack((kin_x, kin_y, kin_z), axis=1)

    def kout(self):
        """
        Return model outcoming wavevectors based on an experimental scattering vectors array

        scat_vec - scattering vectors array of shape (N, 3)
        """
        return self.qs_model + self.kin()

    def det_pts(self, exp_set):
        """
        Return predicted diffraction points at detector plane in pixels
        """
        kout = self.kout()
        phis = np.arctan2(kout[:, 1], kout[:, 0])
        thetas = np.arccos(kout[:, 2])
        det_x = exp_set.det_pos[2] * np.tan(thetas) * np.cos(phis)
        det_y = exp_set.det_pos[2] * np.tan(thetas) * np.sin(phis)
        return (np.stack((det_x, det_y), axis=1) + exp_set.det_pos[:2]) / exp_set.pix_size

class ABCModel(metaclass=ABCMeta):
    """
    Abstract convergent beam diffraction pattern generator class

    rec_lat - reciprocal lattice object
    num_ap - convergent beam numerical aperture
    """
    condition = None

    def __init__(self, rec_lat, num_ap, q_max):
        if q_max > 2:
            raise ValueError('q_max must be less than 2')
        self.rec_lat, self.num_ap = rec_lat, num_ap
        self.rec_vec, self.raw_hkl = rec_lat.sph_vectors(q_max)
        self.rec_abs = np.sqrt((self.rec_vec**2).sum(axis=-1))

    @property
    def rec_th(self):
        return np.arccos(-self.rec_vec[..., 2] / self.rec_abs)

    @property
    def rec_phi(self):
        return np.arctan2(self.rec_vec[..., 1], self.rec_vec[..., 0])

    @property
    def scat_vec(self):
        return self.rec_vec[self.condition]

    @property
    def scat_abs(self):
        return self.rec_abs[self.condition]

    @property
    def scat_th(self):
        return np.arccos(-self.scat_vec[..., 2] / self.scat_abs)

    @property
    def scat_phi(self):
        return np.arctan2(self.scat_vec[..., 1], self.scat_vec[..., 0])

    @property
    def hkl_idx(self):
        return self.raw_hkl[self.condition]

    @property
    def source(self):
        """
        Return source points that define incoming wavevectors for given reciprocal lattice
        """
        return np.stack((-np.sin(self.scat_th - np.arccos(self.scat_abs / 2)) * np.cos(self.scat_phi),
                         -np.sin(self.scat_th - np.arccos(self.scat_abs / 2)) * np.sin(self.scat_phi),
                         np.cos(self.scat_th - np.arccos(self.scat_abs / 2))), axis=-1)

    def laue_vectors(self):
        """
        Return reciprocal lattice points that take part in diffracton
        """
        return self.scat_vec, self.scat_abs

    @abstractmethod
    def source_lines(self):
        pass

    def out_wavevectors(self):
        """
        Return output wave vectors
        """
        return self.source_lines() + self.scat_vec[:, None]

    def det_pts(self, exp_set):
        """
        Return detector diffraction orders points for a detector at given distance

        exp_set - ExperrimentSettings class object
        """
        k_out = self.out_wavevectors()
        det_x = exp_set.det_pos[2] * np.tan(np.arccos(k_out[..., 2])) * np.cos(np.arctan2(k_out[..., 1], k_out[..., 0]))
        det_y = exp_set.det_pos[2] * np.tan(np.arccos(k_out[..., 2])) * np.sin(np.arctan2(k_out[..., 1], k_out[..., 0]))
        return (np.stack((det_x, det_y), axis=-1) + exp_set.det_pos[:2]) / exp_set.pix_size

class CircModel(ABCModel):
    """
    Circular aperture convergent beam diffraction pattern generator class

    rec_lat - reciprocal lattice object
    num_ap - convergent beam numerical aperture
    """
    def __init__(self, rec_lat, num_ap, q_max):
        super(CircModel, self).__init__(rec_lat, num_ap, q_max)
        self.condition = np.abs(np.sin(self.rec_th - np.arccos(self.rec_abs / 2))) < self.num_ap

    def source_lines(self):
        """
        Return source lines of a circular aperture lens
        """
        term1 = self.scat_abs**2 + 2 * self.scat_vec[..., 2] * sqrt(1 - self.num_ap**2)
        term2 = 2 * np.sqrt(self.scat_vec[..., 0]**2 + self.scat_vec[..., 1]**2) * self.num_ap
        source_pt1 = np.stack((-self.num_ap * np.cos(self.scat_phi + np.arccos(term1 / term2)),
                               -self.num_ap * np.sin(self.scat_phi + np.arccos(term1 / term2)),
                               np.repeat(sqrt(1 - self.num_ap**2), self.scat_vec.shape[0])), axis=-1)
        source_pt2 = np.stack((-self.num_ap * np.cos(self.scat_phi - np.arccos(term1 / term2)),
                               -self.num_ap * np.sin(self.scat_phi - np.arccos(term1 / term2)),
                               np.repeat(sqrt(1 - self.num_ap**2), self.scat_vec.shape[0])), axis=-1)
        return np.stack((source_pt1, source_pt2), axis=1)

class RectModel(ABCModel):
    """
    Rectangular aperture convergent beam diffraction pattern generator class

    rec_lat - reciprocal lattice object
    num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axis
    """
    sol_m = np.array([[[0, 1], [0, 1], [1, 0], [1, 0]]])

    def __init__(self, rec_lat, num_ap, q_max):
        super(RectModel, self).__init__(rec_lat, num_ap, q_max)
        cond_list = [np.abs(self.rec_vec[..., 1] / self.rec_vec[..., 0]) <= self.num_ap[1] / self.num_ap[0],
                     np.abs(self.rec_vec[..., 1] / self.rec_vec[..., 0]) >= self.num_ap[1] / self.num_ap[0]]
        func_list = [lambda rec_phi: self.num_ap[0] / np.cos(rec_phi),
                     lambda rec_phi: self.num_ap[1] / np.sin(rec_phi)]
        kin_bound = np.piecewise(self.rec_phi, cond_list, func_list)
        self.condition = np.abs(np.sin(self.rec_th - np.arccos(self.rec_abs / 2))) < np.abs(kin_bound)
        self.bounds = np.array([[[self.num_ap[0], 0],
                                 [-self.num_ap[0], 0],
                                 [0, self.num_ap[1]],
                                 [0, -self.num_ap[1]]]])
        self.init_source_lines()

    def init_source_lines(self):
        """
        Derive source lines of a rectangular aperture lens
        """
        coeff1 = (self.source * self.scat_vec).sum(axis=-1)[..., None] - (self.bounds * self.scat_vec[..., None, :2]).sum(axis=-1)
        coeff2 = np.stack((self.scat_vec[..., 1], self.scat_vec[..., 1], self.scat_vec[..., 0], self.scat_vec[..., 0]), axis=-1)
        alpha = coeff2**2 + self.scat_vec[..., None, 2]**2
        betta = coeff2 * coeff1
        gamma = coeff1**2 - self.scat_vec[..., None, 2]**2 * (1 - self.bounds.sum(axis=2)**2)
        delta = betta**2 - alpha * gamma
        delta_mask = np.where((delta > 0).all(axis=-1))[0]

        alpha, betta, delta = alpha[delta_mask], betta[delta_mask], delta[delta_mask]
        self.condition = np.where(self.condition)[0][delta_mask]

        solution = np.concatenate((self.bounds + self.sol_m * ((betta + np.sqrt(delta)) / alpha)[..., None],
                                   self.bounds + self.sol_m * ((betta - np.sqrt(delta)) / alpha)[..., None]), axis=1)
        solution = np.stack((solution[..., 0],
                             solution[..., 1],
                             np.sqrt(1 - solution[..., 0]**2 - solution[..., 1]**2)), axis=-1)
        ort_mask = np.abs(((self.source[:, None] - solution) * self.scat_vec[:, None]).sum(axis=-1)) < 1e-6
        sol_mask = (np.abs(solution[..., 0]) <= self.num_ap[0]) & (np.abs(solution[..., 1]) <= self.num_ap[1]) & ort_mask
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
    pix_size = 75 * 1e-3

    def kin(self, kout):
        return np.tile(np.array([0, 0, 1]), kout.shape[:-1] + (1,))

    def rec_basis(self, point):
        """
        Return rectangular lattice basis vectors based on the point
        """
        return np.cos(point[3:12]).reshape(self.mat_shape) * point[:3][:, None]

    def rec_vec(self, point):
        """
        Return outcoming wavevectors array of the examined frame
        """
        pts = self.data * self.pix_size - point[12:14]
        angles = np.arctan2(pts[..., 1], pts[..., 0])
        thetas = np.arctan(np.sqrt(pts[..., 0]**2 + pts[..., 1]**2) / point[14])
        kout = np.stack((np.sin(thetas) * np.cos(angles),
                         np.sin(thetas) * np.sin(angles),
                         np.cos(thetas)), axis=-1)
        return RecVectors(kout=kout, kin=self.kin(kout))

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
    def __init__(self, data, step_size=1e-10 * np.ones(15)):
        super(QIndexTF, self).__init__(data, step_size)

    def grid_values(self, point):
        """
        Return target function value array for all possible hkl indices
        """
        rec_vec = self.rec_vec(point)
        rec_lat = RecLattice(self.rec_basis(point))
        qs_model = rec_lat.scat_vec(rec_vec.scat_vec)
        norm = qs_model - rec_vec.kout[:, None]
        return (1 - np.sqrt((norm * norm).sum(axis=-1)))**2

    def values(self, point):
        """
        Return target function value array for a given point
        """
        return np.min(self.grid_values(point), axis=-1)

    def hkl_idxs(self, point):
        """
        Return the most optimal hkl indices based on target function values
        """
        rec_vec = self.rec_vec(point)
        rec_lat = RecLattice(self.rec_basis(point))
        hkl_grid = rec_lat.hkl_grid(rec_vec.scat_vec)
        ind = (np.arange(hkl_grid.shape[0]), np.argmin(self.grid_values(point), axis=-1))
        return hkl_grid[ind]

class QIndexStreaksTF(QIndexTF):
    def grid_values(self, point):
        """
        Return target function value array for all possible hkl indices
        """
        rec_vec = self.rec_vec(point)
        rec_lat = RecLattice(self.rec_basis(point))
        qs_model = rec_lat.scat_vec(rec_vec.scat_vec)
        norm = qs_model[:, :, None] - rec_vec.kout[:, None]
        return ((1 - np.sqrt((norm * norm).sum(axis=-1)))**2).sum(axis=-1)

class OrthQIndexTF(QIndexTF):
    """
    Indexing solution target function class based on scattering vectors error
    with orthogonal basis vectors implied

    data - experimental data
    step_size - step size in numerical derivation

    Point is a flattened array of basis vectors lengths and euler angles [phi1, Phi, phi2]
    Euler angles with Bunge convention are used
    """
    def __init__(self, data, step_size=1e-10 * np.ones(6)):
        super(OrthQIndexTF, self).__init__(data, step_size)

    def rec_basis(self, point):
        """
        Return orthogonal orientation matrix based on euler angles
        """
        return utils.euler_matrix(point[3], point[4], point[5]) * point[:3][:, None]

class ExpSetTF(TargetFunction):
    """
    Experimental geaometry refinement target function class

    data - tilt series diffraction streaks positions in detector pixels
    rec_basis - indexing solution of the first frame in tilt series
    step_size - step size in numerical derivation

    Point is as follows: [theta, phi, det_x, det_y, det_z]
    theta, phi - axis of rotation angles
    det_x, det_y, det_z - position of detector center in respect to the sample
    """
    pix_size = 75 * 1e-3

    def __init__(self, data, rec_basis, frame_idx=1, step_size=1e-10 * np.ones(5)):
        det_lines = data[frame_idx].raw_lines.mean(axis=1)
        super(ExpSetTF, self).__init__(det_lines, step_size)
        self.rec_basis, self.frame_idx = rec_basis, frame_idx

    @property
    def kin(self):
        return np.tile(np.array([0, 0, 1]), (self.data.shape[0], 1))

    def rot_axis(self, point):
        """
        Return an axis of rotation
        """
        return np.array([np.sin(point[0]) * np.cos(point[1]),
                         np.sin(point[0]) * np.sin(point[1]),
                         np.cos(point[0])])

    def rot_matrix(self, point):
        """
        Return one degree rotation matrix
        """
        return utils.rotation_matrix(self.rot_axis(point), -np.radians(self.frame_idx))

    def rec_vec(self, point):
        """
        Return outcoming wavevectors array of the examined frame
        """
        pts = self.data * self.pix_size - point[2:4]
        angles = np.arctan2(pts[..., 1], pts[..., 0])
        thetas = np.arctan(np.sqrt(pts[..., 0]**2 + pts[..., 1]**2) / point[4])
        kout = np.stack((np.sin(thetas) * np.cos(angles),
                         np.sin(thetas) * np.sin(angles),
                         np.cos(thetas)), axis=-1)
        return RecVectors(kout=kout, kin=self.kin)

    def values(self, point):
        """
        Return target function array of values at the given point
        """
        rec_basis = self.rec_basis.dot(self.rot_matrix(point).T)
        q_tf = QIndexTF(self.rec_vec(point))
        return q_tf.values(qindex_point(rec_basis))
    