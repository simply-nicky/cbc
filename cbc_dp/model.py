"""
model.py - convergent beam diffraction forward model
"""
from math import sqrt
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

class ABCModel(metaclass=ABCMeta):
    """
    Abstract convergent beam diffraction pattern generator class

    rec_lat - reciprocal lattice object
    num_ap - convergent beam numerical aperture
    """
    def __init__(self, rec_lat, num_ap):
        if rec_lat.q_max > 2:
            raise ValueError('q_max must be less than 2')
        self.rec_lat, self.num_ap = rec_lat, num_ap
        self.g_x, self.g_y, self.g_z, self.g_abs = rec_lat.vectors()

    @property
    def alpha(self):
        return np.arctan2(self.g_y, self.g_x)

    @property
    def betta(self):
        return np.arccos(-self.g_z / self.g_abs)

    @abstractproperty
    def condition(self):
        pass

    @property
    def q_x(self):
        return self.g_x[self.condition]

    @property
    def q_y(self):
        return self.g_y[self.condition]

    @property
    def q_z(self):
        return self.g_z[self.condition]

    @property
    def q_abs(self):
        return self.g_abs[self.condition]

    @property
    def theta(self):
        return np.arccos(-self.q_z / self.q_abs)

    @property
    def phi(self):
        return np.arctan2(self.q_y, self.q_x)

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
        return (np.stack((self.q_x + onx, self.q_x + oxx), axis=1),
                np.stack((self.q_y + ony, self.q_y + oxy), axis=1),
                np.stack((self.q_z + onz, self.q_z + oxz), axis=1))

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
        return np.abs(np.sin(self.betta - np.arccos(self.g_abs / 2))) < self.num_ap

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

    def __init__(self, rec_lat, num_ap):
        super(SquareModel, self).__init__(rec_lat, num_ap)
        self.condition = np.abs(np.sin(self.betta - np.arccos(self.g_abs / 2))) < sqrt(2) * self.num_ap
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

class GradientDescend():
    def __init__(self, model, start_point, data):
        self.model, self.point, self.data = model, start_point, data
