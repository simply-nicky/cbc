"""
indexer.py - convergent beam crystallography indexer implementation
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import pygmo
from . import utils

class AbcCBI(metaclass=ABCMeta):
    """
    Abstract Convergent beam indexer (CBI) class

    lines - detected diffraction streaks positions at the detector [pixels]
    exp_set - FrameSetup class object
    num_ap = [na_x, na_y] - convergent beam numerical apertures in x- and y-axis
    rec_basis - Reciprocal lattice basis vectors matrix
    tol - tolerance defining vector bounds
    pen_coeff - fitness penalty coefficient
    """
    mat_shape = (3, 3)
    lower_b, upper_b = None, None

    def __init__(self, streaks, num_ap, pen_coeff):
        self.lines, self.exp_set = streaks.raw_lines * streaks.exp_set.pix_size, streaks.exp_set
        self.num_ap, self.pen_coeff = num_ap, pen_coeff

    @abstractmethod
    def rec_basis(self, vec):
        pass

    def kout_exp(self, vec):
        """
        Generate the experimentally measured deiffraction streaks outcoming wavevectors
        """
        return utils.kout_frame(lines=self.lines, x0=vec[0], y0=vec[1], z0=vec[2])

    @abstractmethod
    def voting_hkl(self, vec, kout_exp):
        pass

    @abstractmethod
    def voting_vectors(self, vec, kout_exp):
        pass

    def det_pts(self, kout, vec):
        """
        Return diffraction streaks locations at the detector plane
        """
        theta, phi = np.arccos(kout[..., 2]), np.arctan2(kout[..., 1], kout[..., 0])
        det_x = vec[2] * np.tan(theta) * np.cos(phi)
        det_y = vec[2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + vec[:2]) / self.exp_set.pix_size

    def idxs(self, vot_vec, kout_exp):
        """
        Return the indices of the optimal reciprocal lattice voting vectors
        """
        return utils.fitness_idxs(vot_vec=vot_vec, kout_exp=kout_exp, na_x=self.num_ap[0],
                                  na_y=self.num_ap[1], pen_coeff=self.pen_coeff)

    def get_bounds(self):
        """
        Return the TF argument vector bounds
        """
        return (self.lower_b, self.upper_b)

    def fitness(self, vec):
        """
        Return the target function value for a vector
        """
        kout_exp = self.kout_exp(vec)
        vot_vec = self.voting_vectors(vec, kout_exp)
        return [utils.fitness(vot_vec=vot_vec, kout_exp=kout_exp, na_x=self.num_ap[0],
                              na_y=self.num_ap[1], pen_coeff=self.pen_coeff)]

    def hkl_idxs(self, vec):
        """
        Return the hkl indices of the optimal reciprocal lattice voting vectors
        """
        kout_exp = self.kout_exp(vec)
        vot_vec = self.voting_vectors(vec, kout_exp)
        hkl_idxs = self.voting_hkl(vec, kout_exp)
        return hkl_idxs[self.idxs(vot_vec, kout_exp)]

    def gradient(self, d_vec):
        """
        Return the target function gradient value
        """
        return pygmo.estimate_gradient(self.fitness, d_vec)

    def get_name(self):
        return "A convergent beam indexing problem"

class FrameCBI(AbcCBI):
    def __init__(self, streaks, num_ap, rec_basis, tol, pen_coeff):
        super(FrameCBI, self).__init__(streaks, num_ap, pen_coeff)
        self._init_bounds(rec_basis, tol)

    @abstractmethod
    def _init_bounds(self, rec_basis, tol):
        pass

    def kout_exp(self, vec):
        """
        Generate the experimentally measured diffraction streaks outcoming wavevectors of a frame
        """
        return utils.kout_frame(lines=self.lines, x0=vec[0], y0=vec[1], z0=vec[2])

    def voting_vectors(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting points for the given experimental outcoming
        wavevectors kout_exp
        """
        return utils.voting_vectors_f(kout_exp=kout_exp.mean(axis=1), rec_basis=self.rec_basis(vec),
                                      na_x=self.num_ap[0], na_y=self.num_ap[1])

    def voting_hkl(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting hkl indices for the given experimental outcoming
        wavevectors kout_exp
        """
        return utils.voting_idxs_f(kout_exp=kout_exp.mean(axis=1), rec_basis=self.rec_basis(vec),
                                   na_x=self.num_ap[0], na_y=self.num_ap[1])

class ScanCBI(AbcCBI):
    def __init__(self, streaks, num_ap, rec_basis, thetas, tol, pen_coeff=10):
        super(ScanCBI, self).__init__(streaks, num_ap, pen_coeff)
        self.frame_idxs, self.thetas, self.size = streaks.frame_idxs, thetas, thetas.size
        self._init_bounds(rec_basis.ravel(), thetas, tol)

    def _init_bounds(self, rec_basis, thetas, tol):
        pt0_lb = np.repeat((1 - np.array(tol[0])) * self.exp_set.det_pos, self.size)
        pt0_ub = np.repeat((1 + np.array(tol[0])) * self.exp_set.det_pos, self.size)
        th_lb, th_ub = thetas - tol[1], thetas + tol[1]
        rot_lb, rot_ub = np.pi / 2 - tol[2] * np.ones(2), np.pi / 2 + tol[2] * np.ones(2)
        rb_bounds = np.stack(((1 - tol[1]) * rec_basis, (1 + tol[1]) * rec_basis))
        self.lower_b = np.concatenate((pt0_lb, th_lb, rot_lb, rb_bounds.min(axis=0)))
        self.upper_b = np.concatenate((pt0_ub, th_ub, rot_ub, rb_bounds.max(axis=0)))

    def rec_basis(self, vec):
        """
        Return rectangular lattice basis vectors for a vector
        """
        return vec[4 * self.size + 2:].reshape(self.mat_shape)

    def kout_exp(self, vec):
        """
        Generate the experimentally measured diffraction streaks outcoming wavevectors of a scan
        """
        return utils.kout_scan(lines=self.lines, frame_idxs=self.frame_idxs, x0=vec[0:self.size], y0=vec[self.size:2*self.size], z0=vec[2*self.size:3*self.size])

    def voting_vectors(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting points for the given experimental outcoming
        wavevectors kout_exp
        """
        return utils.voting_vectors_s(kout_exp=kout_exp.mean(axis=1), rec_basis=self.rec_basis(vec),
                                      thetas=vec[3*self.size:4*self.size], frame_idxs=self.frame_idxs,
                                      alpha=vec[4*self.size], betta=vec[4*self.size+1],
                                      na_x=self.num_ap[0], na_y=self.num_ap[1])

    def voting_hkl(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting hkl indices for the given experimental outcoming
        wavevectors kout_exp
        """
        return utils.voting_idxs_s(kout_exp=kout_exp.mean(axis=1), rec_basis=self.rec_basis(vec),
                                   thetas=vec[3*self.size:4*self.size], frame_idxs=self.frame_idxs,
                                   alpha=vec[4*self.size], betta=vec[4*self.size+1],
                                   na_x=self.num_ap[0], na_y=self.num_ap[1])

class FCBI(FrameCBI):
    """
    Convergent beam crystallography indexer class
    Argument vector is comprised of detector relative position
    and full reciprocal basis vectors matrix

    lines                           - detected diffraction streaks positions at the detector
                                      [mm]
    exp_set                         - FrameSetup class object
    num_ap = [na_x, na_y]           - convergent beam numerical apertures in x- and y-axis
    rec_basis                       - Reciprocal lattice basis vectors matrix
    tol = (pos_tol, rb_tol)         - relative detector position and
                                      reciprocal basis matrix tolerances [0.0 - 1.0]
    pen_coeff                       - fitness penalty coefficient
    """
    def __init__(self, streaks, num_ap, rec_basis, tol=([0.03, 0.03, 0.075], 0.12), pen_coeff=10):
        super(FCBI, self).__init__(streaks, num_ap, rec_basis.ravel(), tol, pen_coeff)

    def _init_bounds(self, rec_basis, tol):
        rb_bounds = np.stack(((1 - tol[1]) * rec_basis, (1 + tol[1]) * rec_basis))
        self.lower_b = np.concatenate(((1 - np.array(tol[0])) * self.exp_set.det_pos, rb_bounds.min(axis=0)))
        self.upper_b = np.concatenate(((1 + np.array(tol[0])) * self.exp_set.det_pos, rb_bounds.max(axis=0)))

    def rec_basis(self, vec):
        """
        Return rectangular lattice basis vectors for a vector
        """
        return vec[3:].reshape(self.mat_shape)

    def get_extra_info(self):
        return "Dimensions: 12 in total\n3 - detector position\n9 - reciprocal lattice basis vectors matrix"

class RCBI(FrameCBI):
    """
    Convergent beam crystallography indexer class
    Argument vector is comprised of detector relative position,
    flattened array of basis vectors lengths, and euler angles [phi1, Phi, phi2]
    Euler angles with Bunge convention are used

    lines                               - detected diffraction streaks positions at the detector
                                          [mm]
    exp_set                             - FrameSetup class object
    num_ap = [na_x, na_y]               - convergent beam numerical apertures in x- and y-axis
    rec_basis                           - Reciprocal lattice basis vectors matrix
    tol = (pos_tol, size_tol, ang_tol)  - relative detector position, basis vector lengths,
                                          and rotation angles tolerances [0.0 - 1.0]
    pen_coeff                           - fitness penalty coefficient
    """
    def __init__(self, streaks, num_ap, rec_basis, tol=([0.03, 0.03, 0.075], 0.1, np.radians(5)), pen_coeff=10):
        super(RCBI, self).__init__(streaks, num_ap, rec_basis, tol, pen_coeff)

    def _init_bounds(self, rec_basis, tol):
        self.rec_sizes = np.sqrt((rec_basis**2).sum(axis=-1))
        self.or_mat = rec_basis / self.rec_sizes[:, None]
        self.lower_b = np.concatenate(((1 - np.array(tol[0])) * self.exp_set.det_pos, (1 - tol[1]) * self.rec_sizes, -tol[2] * np.ones(3)))
        self.upper_b = np.concatenate(((1 + np.array(tol[0])) * self.exp_set.det_pos, (1 + tol[1]) * self.rec_sizes, tol[2] * np.ones(3)))

    def rec_basis(self, vec):
        """
        Return orthogonal orientation matrix based on euler angles
        """
        return self.or_mat.dot(utils.euler_matrix(vec[6], vec[7], vec[8]).T) * vec[3:6, None]

    def get_extra_info(self):
        return "Dimensions: 9 in total\n3 - detector position\n3 - reciprocal lattice basis vectors lengths\n3 - orientation matrix angles"