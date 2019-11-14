"""
model.py - convergent beam diffraction forward model
"""
from math import sqrt
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class RecLattice():
    """
    Reciprocal lattice class

    or_mat - orientation matrix (dimensionless)
    center - reciprocal lattice center
    """
    def __init__(self, or_mat, center=np.zeros(3)):
        self.or_mat, self.center = or_mat, center
        self.or_norm = np.sqrt((self.or_mat * self.or_mat).sum(axis=1))
        self.or_inv = np.linalg.inv(self.or_mat)

    def hkl_idx(self, scat_vec):
        """
        Return hkl indices for an array of scattering vectors

        scat_vec - scattering vectors array of shape (N, 3)
        """
        hkl = (scat_vec  - self.center).dot(self.or_inv)
        return np.around(hkl)

    def scat_vec(self, scat_vec):
        """
        Return model scattering vectors based on an experimental scattering vectors array

        scat_vec - scattering vectors array of shape (N, 3)
        """
        hkl_idx = self.hkl_idx(scat_vec)
        return hkl_idx.dot(self.or_mat) + self.center

    def sph_vectors(self, q_max):
        """
        Return flattened reciprocal lattice vectors inside a sphere

        q_max - reciprocal sphere radius
        """
        lat_size = np.rint(q_max / self.or_norm)
        h_idxs = np.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = np.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = np.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = np.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl_idx = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)
        rec_vec = hkl_idx.dot(self.or_mat)
        rec_abs = np.sqrt((rec_vec * rec_vec).sum(axis=1))
        idxs = np.where((rec_abs != 0) & (rec_abs < q_max))
        return rec_vec[idxs], hkl_idx[idxs]

class ABCModel(metaclass=ABCMeta):
    """
    Abstract convergent beam diffraction pattern generator class

    rec_lat - reciprocal lattice object
    num_ap - convergent beam numerical aperture
    """
    def __init__(self, rec_lat, num_ap, q_max):
        if q_max > 2:
            raise ValueError('q_max must be less than 2')
        self.rec_lat, self.num_ap, self.center = rec_lat, num_ap, rec_lat.center
        self.rec_vec, self.raw_hkl = rec_lat.sph_vectors(q_max)
        self.rec_abs = np.sqrt((self.rec_vec * self.rec_vec).sum(axis=1))

    @property
    def alpha(self):
        return np.arctan2(self.rec_vec[:, 1], self.rec_vec[:, 0])

    @property
    def betta(self):
        return np.arccos(-self.rec_vec[:, 2] / self.rec_abs)

    @abstractproperty
    def condition(self):
        pass

    @property
    def q_x(self):
        return self.rec_vec[:, 0][self.condition]

    @property
    def q_y(self):
        return self.rec_vec[:, 1][self.condition]

    @property
    def q_z(self):
        return self.rec_vec[:, 2][self.condition]

    @property
    def q_abs(self):
        return self.rec_abs[self.condition]

    @property
    def theta(self):
        return np.arccos(-self.q_z / self.q_abs)

    @property
    def phi(self):
        return np.arctan2(self.q_y, self.q_x)

    @property
    def hkl_idx(self):
        return self.raw_hkl[self.condition]

    def laue_vectors(self):
        """
        Return reciprocal lattice points that take part in diffracton
        """
        return self.q_x, self.q_y, self.q_z, self.q_abs

    def source_pts(self):
        """
        Return source points that define incoming wavevectors for given reciprocal lattice
        """
        o_x = -np.sin(self.theta - np.arccos(self.q_abs / 2)) * np.cos(self.phi)
        o_y = -np.sin(self.theta - np.arccos(self.q_abs / 2)) * np.sin(self.phi)
        o_z = np.cos(self.theta - np.arccos(self.q_abs / 2))
        return o_x, o_y, o_z

    @abstractmethod
    def entry_pts(self): pass

    @abstractmethod
    def exit_pts(self): pass

    def out_wavevectors(self):
        """
        Return output wave vectors
        """
        onx, ony, onz = self.entry_pts()
        oxx, oxy, oxz = self.exit_pts()
        return (np.stack((self.q_x + onx, self.q_x + oxx), axis=1) + self.center[0],
                np.stack((self.q_y + ony, self.q_y + oxy), axis=1) + self.center[1],
                np.stack((self.q_z + onz, self.q_z + oxz), axis=1) + self.center[2])

    def detector_pts(self, det_dist):
        """
        Return detector diffraction orders points for a detector at given distance

        det_dist - detector distance
        """
        k_x, k_y, k_z = self.out_wavevectors()
        det_x = det_dist * np.tan(np.sqrt(2 - 2 * k_z)) * np.cos(np.arctan2(k_y, k_x))
        det_y = det_dist * np.tan(np.sqrt(2 - 2 * k_z)) * np.sin(np.arctan2(k_y, k_x))
        return det_x, det_y

class CircModel(ABCModel):
    """
    Circular aperture convergent beam diffraction pattern generator class

    rec_lat - reciprocal lattice object
    num_ap - convergent beam numerical aperture
    """
    @property
    def condition(self):
        return np.abs(np.sin(self.betta - np.arccos(self.rec_abs / 2))) < self.num_ap

    def entry_pts(self):
        """
        Return first point array of source lines
        """
        dphi = np.arccos((self.q_abs**2 + 2 * self.q_z * sqrt(1 - self.num_ap**2)) / (2 * np.sqrt(self.q_x**2 + self.q_y**2) * self.num_ap))
        return (-self.num_ap * np.cos(self.phi + dphi),
                -self.num_ap * np.sin(self.phi + dphi),
                np.repeat(sqrt(1 - self.num_ap**2), self.q_x.shape))

    def exit_pts(self):
        """
        Return second point array of source lines
        """
        dphi = np.arccos((self.q_abs**2 + 2 * self.q_z * sqrt(1 - self.num_ap**2)) / (2 * np.sqrt(self.q_x**2 + self.q_y**2) * self.num_ap))
        return (-self.num_ap * np.cos(self.phi - dphi),
                -self.num_ap * np.sin(self.phi - dphi),
                np.repeat(sqrt(1 - self.num_ap**2), self.q_x.shape))

class SquareModel(ABCModel):
    """
    Rectangular aperture convergent beam diffraction pattern generator class

    rec_lat - reciprocal lattice object
    num_ap - convergent beam numerical aperture
    """
    condition = None

    def __init__(self, rec_lat, num_ap, q_max):
        super(SquareModel, self).__init__(rec_lat, num_ap, q_max)
        self.condition = np.abs(np.sin(self.betta - np.arccos(self.rec_abs / 2))) < sqrt(2) * self.num_ap
        self._pts_x, self._pts_y, _mask = self._source_lines()
        self.condition = (np.where(self.condition)[0][_mask],)

    @property
    def bounds(self):
        return np.stack(([self.num_ap, 0],
                         [-self.num_ap, 0],
                         [0, self.num_ap],
                         [0, -self.num_ap]), axis=1)

    def _source_lines(self):
        boundary_prd = (np.multiply.outer(self.q_x, self.bounds[0]) +
                        np.multiply.outer(self.q_y, self.bounds[1]))
        source_x, source_y, source_z = self.source_pts()
        c_1 = (source_x * self.q_x + source_y * self.q_y + source_z * self.q_z)[:, np.newaxis] - boundary_prd
        c_2 = np.stack((self.q_y, self.q_y, self.q_x, self.q_x), axis=1)
        c_3 = np.stack(([0, 1], [0, 1], [1, 0], [1, 0]), axis=1)
        a_coeff, b_coeff, c_coeff = (c_2**2 + self.q_z[:, np.newaxis]**2,
                                     c_2 * c_1,
                                     c_1**2 - (self.q_z**2 * (1 - self.num_ap**2))[:, np.newaxis])
        delta = b_coeff**2 - a_coeff * c_coeff
        delta_mask = np.where((delta > 0).all(axis=1))
        a_masked, b_masked, delta_masked = a_coeff[delta_mask], b_coeff[delta_mask], delta[delta_mask]
        pts = np.concatenate((self.bounds + (c_3 * ((b_masked + np.sqrt(delta_masked)) / a_masked)[:, np.newaxis]),
                              self.bounds + (c_3 * ((b_masked - np.sqrt(delta_masked)) / a_masked)[:, np.newaxis])), axis=2)
        ort_mask = np.abs((source_x[delta_mask][:, np.newaxis] - pts[:, 0]) *
                          self.q_x[delta_mask][:, np.newaxis] +
                          (source_y[delta_mask][:, np.newaxis] - pts[:, 1]) *
                          self.q_y[delta_mask][:, np.newaxis] +
                          (source_z[delta_mask][:, np.newaxis] - np.sqrt(1 - pts[:, 0]**2 - pts[:, 1]**2)) *
                          self.q_z[delta_mask][:, np.newaxis]) < 1e-6
        pts_x = pts[:, 0][(np.abs(pts) <= self.num_ap).all(axis=1) & ort_mask].reshape(-1, 2)
        pts_y = pts[:, 1][(np.abs(pts) <= self.num_ap).all(axis=1) & ort_mask].reshape(-1, 2)
        return pts_x, pts_y, (delta_mask[0][((np.abs(pts) <= self.num_ap).all(axis=1) & ort_mask).any(axis=1)],)

    def entry_pts(self):
        """
        Return first point array of source lines
        """
        return (self._pts_x[:, 0],
                self._pts_y[:, 0],
                np.sqrt(1 - self._pts_x[:, 0]**2 - self._pts_y[:, 0]**2))

    def exit_pts(self):
        """
        Return second point array of source lines
        """
        return (self._pts_x[:, 1],
                self._pts_y[:, 1],
                np.sqrt(1 - self._pts_x[:, 1]**2 - self._pts_y[:, 1]**2))

class GradientDescent():
    """
    Gradient Descent algorithm class

    or_mat - reciprocal lattice orientation matrix
    center - reciprocal lattice center
    target_func - target function to minimize
    step_size - step size relative to point norm
    """
    def __init__(self, or_mat, center, target_func, step_size=1e-2):
        self.point = np.concatenate((or_mat, center[None, :]))
        self.trg_func, self.step_size = target_func, step_size
        or_step = np.tile(np.sqrt((or_mat * or_mat).sum(axis=1))[:, None] * step_size, (1, 3))
        center_step = or_step[:, 0].mean() * np.ones((1, 3))
        self.point_step = np.concatenate((or_step, center_step))
        self.axes = np.zeros((self.point.size,) + self.point.shape)
        idxs = np.unravel_index(np.arange(0, self.axes.size, self.point.size + 1), self.axes.shape)
        self.axes[idxs] = 1

    def derivative(self, axis):
        """
        Return target function partial derivative along the axis

        axis - axis of differentiation
        """
        point_step = self.point_step * axis
        form_val = self.trg_func(self.point + point_step)
        lat_val = self.trg_func(self.point - point_step)
        return (form_val - lat_val) / 2 / point_step.sum()

    def value(self):
        """
        Return target function value
        """
        return self.trg_func(self.point)

    def gradient(self):
        """
        Return target function gradient
        """
        return np.array([self.derivative(axis) for axis in self.axes]).reshape(self.point.shape)

    def next_point(self):
        """
        Return next iterative point
        """
        grad = self.gradient()
        next_point = self.point - np.sum((grad / self.point_step)**2)**-0.5 * grad
        return GradientDescent(or_mat=next_point[:3],
                               center=next_point[3],
                               target_func=self.trg_func,
                               step_size=self.step_size)

class TargetFunction():
    """
    Target function class

    data - experimental data
    """
    def __init__(self, data):
        self.data = data

    def __call__(self, point):
        """
        Return target function value at the given point
        """
        rec_lat = RecLattice(or_mat=point[:3], center=point[3])
        qs_model = rec_lat.scat_vec(self.data.scat_vec)
        norm = qs_model - self.data.kout
        delta_n = np.abs(1 - np.sqrt((norm * norm).sum(axis=1)))
        return np.median(delta_n)
    