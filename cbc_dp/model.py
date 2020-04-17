"""
model.py - convergent beam diffraction forward model
"""
from abc import ABCMeta, abstractmethod
import configparser
import numpy as np
from . import utils

class RecBasis():
    """
    Reciprocal lattice basis vectors class

    rb_mat - basis vectors matrix
    """
    def __init__(self, rb_mat):
        self.rb_mat, self.inv_mat = rb_mat, utils.inverse_matrix(rb_mat)
        self.sizes = np.sqrt((rb_mat**2).sum(axis=-1))
        self.or_mat = rb_mat / self.sizes[:, None]

    @classmethod
    def import_ini(cls, ini_file):
        """
        Import RecBasis class object from an ini file
        """
        config = configparser.ConfigParser()
        config.read(ini_file)
        a_vec = np.array([config.getfloat('rec_basis', 'ax'),
                          config.getfloat('rec_basis', 'ay'),
                          config.getfloat('rec_basis', 'az')])
        b_vec = np.array([config.getfloat('rec_basis', 'bx'),
                          config.getfloat('rec_basis', 'by'),
                          config.getfloat('rec_basis', 'bz')])
        c_vec = np.array([config.getfloat('rec_basis', 'cx'),
                          config.getfloat('rec_basis', 'cy'),
                          config.getfloat('rec_basis', 'cz')])
        return cls(rb_mat=np.stack((a_vec, b_vec, c_vec)))

    def save_ini(self, out_path):
        """
        Save reciprocal basis to an ini file
        """
        config = configparser.ConfigParser()
        config['rec_basis'] = {'ax': self.rb_mat[0, 0],
                               'ay': self.rb_mat[0, 1],
                               'az': self.rb_mat[0, 2],
                               'bx': self.rb_mat[1, 0],
                               'by': self.rb_mat[1, 1],
                               'bz': self.rb_mat[1, 2],
                               'cx': self.rb_mat[2, 0],
                               'cy': self.rb_mat[2, 1],
                               'cz': self.rb_mat[2, 2]}
        with open(out_path, 'w') as ini_file:
            config.write(ini_file)

class ABCLattice(metaclass=ABCMeta):
    """
    Abstract reciprocal lattice in convergent beam diffraction class

    rec_basis - RecBasis class object
    """
    rec_vec, hkl_idxs = None, None

    def __init__(self, rec_basis):
        self.rec_basis = rec_basis
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
        lat_size = np.rint(self.q_max / self.rec_basis.sizes)
        h_idxs = np.arange(-lat_size[0], lat_size[0] + 1)
        k_idxs = np.arange(-lat_size[1], lat_size[1] + 1)
        l_idxs = np.arange(-lat_size[2], lat_size[2] + 1)
        h_grid, k_grid, l_grid = np.meshgrid(h_idxs, k_idxs, l_idxs)
        hkl_idxs = np.stack((h_grid.ravel(), k_grid.ravel(), l_grid.ravel()), axis=1)
        rec_vec = hkl_idxs.dot(self.rec_basis.rb_mat)
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
        self.rec_vec = self.hkl_idxs.dot(self.rec_basis.rb_mat)

class ABCModel():
    """
    Abstract convergent beam diffraction pattern generator class

    rec_lat - RecLattice class object
    """
    mask = None
    masked_attrs = ['rec_vec', 'hkl_idxs', 'rec_abs', 'rec_phi', 'rec_th', 'source']

    def __init__(self, rec_lat):
        self.rec_lat = rec_lat
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

    rec_lat - RecLattice class object
    num_ap - convergent beam numerical aperture
    """
    def __init__(self, rec_lat, num_ap):
        self.num_ap = num_ap
        super(CircModel, self).__init__(rec_lat)

    def _init_mask(self):
        self.mask = (np.abs(np.sin(self.rec_lat.rec_th - np.arccos(self.rec_lat.rec_abs / 2))) < self.num_ap)

    def source_lines(self):
        """
        Return source lines of a circular aperture lens
        """
        kin_z = np.sqrt(1 - self.num_ap**2)
        term1 = self.rec_abs**2 + 2 * self.rec_vec[..., 2] * kin_z
        term2 = 2 * np.sqrt(self.rec_vec[..., 0]**2 + self.rec_vec[..., 1]**2) * self.num_ap
        source_pt1 = np.stack((-self.num_ap * np.cos(self.rec_phi + np.arccos(term1 / term2)),
                               -self.num_ap * np.sin(self.rec_phi + np.arccos(term1 / term2)),
                               np.repeat(kin_z, self.rec_vec.shape[0])), axis=-1)
        source_pt2 = np.stack((-self.num_ap * np.cos(self.rec_phi - np.arccos(term1 / term2)),
                               -self.num_ap * np.sin(self.rec_phi - np.arccos(term1 / term2)),
                               np.repeat(kin_z, self.rec_vec.shape[0])), axis=-1)
        return np.stack((source_pt1, source_pt2), axis=1)

class RectModel(ABCModel):
    """
    Rectangular aperture convergent beam diffraction pattern generator class

    rec_lat - RecLattice class object
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    """
    def __init__(self, rec_lat, kin):
        self.kin = kin
        super(RectModel, self).__init__(rec_lat)

    def _init_mask(self):
        self._source_lines, self.mask = utils.model_source_lines(source=self.rec_lat.source,
                                                                 rec_vec=self.rec_lat.rec_vec,
                                                                 kin=self.kin)

    def source_lines(self):
        """
        Return source lines of a rectangular aperture lens
        """
        return self._source_lines
