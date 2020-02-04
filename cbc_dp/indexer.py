"""
indexer.py - convergent beam crystallography indexer implementation
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import pygmo
from . import utils
from .model import RectModel

class AbcCBI(metaclass=ABCMeta):
    """
    Abstract Convergent beam indexer (CBI) class

    lines - detected diffraction streaks positions at the detector [pixels]
    exp_set - FrameSetup class object
    num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axis
    rec_basis - Reciprocal lattice basis vectors matrix
    tol - tolerance defining vector bounds
    pen_coeff - fitness penalty coefficient
    """
    mat_shape = (3, 3)
    lower_b, upper_b = None, None

    def __init__(self, lines, exp_set, num_ap, rec_basis, tol, pen_coeff):
        self.lines, self.exp_set, self.num_ap, self.pen_coeff = lines, exp_set, num_ap, pen_coeff
        self._init_bounds(rec_basis, tol)

    @abstractmethod
    def _init_bounds(self, rec_basis, tol):
        pass

    @abstractmethod
    def rec_basis(self, vec):
        pass

    def kout_exp(self, vec):
        """
        Generate the experimentally measured deiffraction streaks outcoming wavevectors
        """
        return utils.kout(self.lines * self.exp_set.pix_size, self.exp_set.det_pos, vec)

    def det_pts(self, kout, vec):
        """
        Return diffraction streaks locations at the detector plane
        """
        theta, phi = np.arccos(kout[..., 2]), np.arctan2(kout[..., 1], kout[..., 0])
        det_x = self.exp_set.det_pos[2] * (1 + vec[2]) * np.tan(theta) * np.cos(phi)
        det_y = self.exp_set.det_pos[2] * (1 + vec[2]) * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + self.exp_set.det_pos[:2] * (1 + vec[:2])) / self.exp_set.pix_size

    def voting_vectors(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting points for the given experimental outcoming
        wavevectors kout_exp
        """
        return utils.voting_vectors(kout_exp=kout_exp.mean(axis=1),
                                    rec_basis=self.rec_basis(vec),
                                    num_ap_x=self.num_ap[0],
                                    num_ap_y=self.num_ap[1])

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
        return [utils.fitness(vot_vec=vot_vec,
                              kout_exp=kout_exp,
                              num_ap=np.sqrt(self.num_ap[0]**2 + self.num_ap[1]**2),
                              pen_coeff=self.pen_coeff)]

    def idxs(self, vec):
        """
        Return the indices of the optimal reciprocal lattice voting vectors
        """
        kout_exp = self.kout_exp(vec)
        vot_vec = self.voting_vectors(vec, kout_exp)
        return utils.fitness_idxs(vot_vec=vot_vec,
                                  kout_exp=kout_exp,
                                  num_ap=np.sqrt(self.num_ap[0]**2 + self.num_ap[1]**2),
                                  pen_coeff=self.pen_coeff)

    def gradient(self, d_vec):
        """
        Return the target function gradient value
        """
        return pygmo.estimate_gradient(self.fitness, d_vec)

    def get_name(self):
        return "A convergent beam indexing problem"

class FCBI(AbcCBI):
    """
    Convergent beam crystallography indexer class
    Argument vector is comprised of detector relative position
    and full reciprocal basis vectors matrix

    lines                           - detected diffraction streaks positions at the detector
                                      [mm]
    exp_set                         - FrameSetup class object
    num_ap = [num_ap_x, num_ap_y]   - convergent beam numerical apertures in x- and y-axis
    rec_basis                       - Reciprocal lattice basis vectors matrix
    tol = (pos_tol, rb_tol)         - relative detector position and
                                      reciprocal basis matrix tolerances [0.0 - 1.0]
    pen_coeff                       - fitness penalty coefficient
    """
    def __init__(self, lines, exp_set, num_ap, rec_basis, tol=(0.1, 0.05), pen_coeff=10):
        super(FCBI, self).__init__(lines, exp_set, num_ap, rec_basis.ravel(), tol, pen_coeff)

    def _init_bounds(self, rec_basis, tol):
        rb_bounds = np.stack(((1 - tol[1]) * rec_basis, (1 + tol[1]) * rec_basis))
        self.lower_b = np.concatenate((-tol[0] * np.ones(3), rb_bounds.min(axis=0)))
        self.upper_b = np.concatenate((tol[0] * np.ones(3), rb_bounds.max(axis=0)))

    def rec_basis(self, vec):
        """
        Return rectangular lattice basis vectors for a vector
        """
        return vec[3:].reshape(self.mat_shape)

    def get_extra_info(self):
        return "Dimensions: 12 in total\n3 - detector position\n9 - reciprocal lattice basis vectors matrix"

class RCBI(AbcCBI):
    """
    Convergent beam crystallography indexer class
    Argument vector is comprised of detector relative position,
    flattened array of basis vectors lengths, and euler angles [phi1, Phi, phi2]
    Euler angles with Bunge convention are used

    lines                               - detected diffraction streaks positions at the detector
                                          [mm]
    exp_set                             - FrameSetup class object
    num_ap = [num_ap_x, num_ap_y]       - convergent beam numerical apertures in x- and y-axis
    rec_basis                           - Reciprocal lattice basis vectors matrix
    tol = (pos_tol, size_tol, ang_tol)  - relative detector position, basis vector lengths,
                                          and orientation matrix angles tolerances [0.0 - 1.0]
    pen_coeff                           - fitness penalty coefficient
    """
    def __init__(self, lines, exp_set, num_ap, rec_basis, tol=(0.05, 0.1, np.radians(5)), pen_coeff=10):
        super(RCBI, self).__init__(lines, exp_set, num_ap, rec_basis, tol, pen_coeff)

    def _init_bounds(self, rec_basis, tol):
        self.rec_sizes = np.sqrt((rec_basis**2).sum(axis=-1))
        self.or_mat = rec_basis / self.rec_sizes[:, None]
        self.lower_b = np.concatenate((-tol[0] * np.ones(3), (1 - tol[1]) * self.rec_sizes, -tol[2] * np.ones(3)))
        self.upper_b = np.concatenate((tol[0] * np.ones(3), (1 + tol[1]) * self.rec_sizes, tol[2] * np.ones(3)))

    def rec_basis(self, vec):
        """
        Return orthogonal orientation matrix based on euler angles
        """
        return self.or_mat.dot(utils.euler_matrix(vec[6], vec[7], vec[8]).T) * vec[3:6, None]

    def get_extra_info(self):
        return "Dimensions: 9 in total\n3 - detector position\n3 - reciprocal lattice basis vectors lengths\n3 - orientation matrix angles"

class IndexSolution():
    """
    Refined indexing solution class

    target_func - indexing refinement target function
    sol - refined indexing solution
    """
    def __init__(self, target_func, sol):
        self.target_func, self.sol = target_func, sol
        self.rec_basis, self.exp_set = target_func.rec_basis(sol), target_func.exp_set(sol)

    @property
    def num_ap(self):
        return self.target_func.num_ap

    def rec_vectors(self):
        """
        Return the optimal reciprocal lattice vectors
        """
        return self.target_func.voting_vectors(self.sol)

    def q_max(self):
        """
        Return the maximal reciprocal vectors norm value
        """
        return np.max(np.sqrt((self.rec_vectors**2).sum(axis=-1)))

    def model(self):
        """
        Generate diffraction pattern based on indexing solution

        Return detector plane indexing points and streaks positions
        """
        rect_model = RectModel(rec_basis=self.rec_basis,
                               num_ap=self.num_ap,
                               q_max=self.q_max())
        model_pts = rect_model.det_pts(self.exp_set)
        model_lines = rect_model.det_lines(self.exp_set)
        return model_pts, model_lines
