"""
indexer.py - convergent beam crystallography indexer implementation
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import pygmo
from scipy.ndimage import gaussian_filter, label, maximum_position
from . import utils
from .grouper import TiltGroups

class FrameStreaks():
    """
    Detected diffraction streaks on one frame class

    lines - detected lines
    zero - zero output wavevector point
    exp_set - ScanSetup class object
    """
    def __init__(self, lines, exp_set):
        self.raw_lines, self.exp_set = lines, exp_set
        self.lines = self.raw_lines * self.exp_set.pix_size - self.exp_set.det_pos[:2]

    @property
    def kin(self):
        return np.tile(np.array([0., 0., 1.]), (self.size, 1))

    @property
    def size(self):
        return self.lines.shape[0]

    @property
    def taus(self):
        taus = (self.lines[:, 1] - self.lines[:, 0])
        return taus / np.sqrt(taus[:, 0]**2 + taus[:, 1]**2)[:, np.newaxis]

    def __iter__(self):
        for line in self.raw_lines:
            yield line

    def __getitem__(self, idx):
        return FrameStreaks(self.raw_lines[idx], self.exp_set)

    def index_pts(self):
        """
        Return indexing points
        """
        products = self.lines[:, 0, 1] * self.taus[:, 0] - self.lines[:, 0, 0] * self.taus[:, 1]
        return np.stack((-self.taus[:, 1] * products, self.taus[:, 0] * products), axis=1)

    def kout(self, theta):
        """
        Return reciprocal vectors of detected diffraction streaks.

        theta - angle of rotation
        """
        rot_m = self.exp_set.rotation_matrix(theta)
        kout = self.exp_set.kout_exp(self.lines.mean(axis=1))
        return RecVectors(kout=kout.dot(rot_m.T),
                          kin=self.kin.dot(rot_m.T))

    def kout_streaks(self, theta):
        """
        Return reciprocal vectors of detected diffraction streaks.

        theta - angle of rotation
        """
        rot_m = self.exp_set.rotation_matrix(theta)
        kout = self.exp_set.kout_exp(self.lines)
        return RecVectors(kout=kout.dot(rot_m.T),
                          kin=np.tile(self.kin[:, None], (1, 2, 1)).dot(rot_m.T))

    def kout_index(self, theta):
        """
        Return reciprocal points of indexing points.

        theta - angle of rotation
        """
        index_pts = self.index_pts()
        rot_m = self.exp_set.rotation_matrix(theta)
        kout = self.exp_set.kout_exp(index_pts)
        return RecVectors(kout=kout.dot(rot_m.T), kin=self.kin.dot(rot_m.T))

    def full_index_refine(self, rec_basis, num_ap, pos_tol=(0.007, 0.014, 0.06), rb_tol=0.12):
        """
        Return a population of reciprocal lattice basis vectors matrix refinement problem

        rec_basis - reciprocal lattice basis vectors matrix
        num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axes
        pos_tol - relative sample position tolerance
        rb_tol - lattice basis vectors matrix tolerance
        """
        full_tf = FCBI(streaks=self,
                       rec_basis=rec_basis,
                       num_ap=num_ap,
                       tol=(pos_tol, rb_tol))
        return pygmo.problem(full_tf)

    def rot_index_refine(self, rec_basis, num_ap, pos_tol=(0.007, 0.014, 0.06), size_tol=0.05, ang_tol=0.09):
        """
        Return a population of reciprocal lattice rotation refinement problem

        rec_basis - reciprocal lattice basis vectors matrix
        num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axes
        pos_tol - relative sample position tolerance
        size_tol - lattice basis vectors length tolerance
        ang_tol - rotation anlges tolerance
        """
        rot_tf = RCBI(streaks=self,
                      rec_basis=rec_basis,
                      num_ap=num_ap,
                      tol=(pos_tol, size_tol, ang_tol))
        return pygmo.problem(rot_tf)

class ScanStreaks(FrameStreaks):
    """
    Detected diffraction streaks of a rotational scan class

    raw_lines - detected lines
    exp_set - ScanSetup class object
    frame_idxs - frame indices
    """
    def __init__(self, raw_lines, exp_set, frame_idxs):
        self.frame_idxs = frame_idxs
        super(ScanStreaks, self).__init__(raw_lines, exp_set)

    @classmethod
    def import_series(cls, strks_list):
        """
        Import ScanStreaks object from list of FrameStreaks

        strks_list - detected lines FrameStreaks class object list
        """
        frame_idxs = np.concatenate([idx * np.ones(strks.size, dtype=np.int)
                                     for idx, strks in enumerate(strks_list)])
        raw_lines = np.concatenate([strks.raw_lines for strks in strks_list])
        return cls(raw_lines, strks_list[0].exp_set, frame_idxs)

    def extract(self, idxs):
        """
        Return a ScanStreaks object of the given streaks subset
        """
        return ScanStreaks(self.raw_lines[idxs], self.exp_set, self.frame_idxs[idxs])

    def __getitem__(self, frame_idx):
        if isinstance(frame_idx, int):
            idxs = np.where(self.frame_idxs == frame_idx)
            return FrameStreaks(self.raw_lines[idxs], self.exp_set)
        elif isinstance(frame_idx, np.ndarray) and np.issubdtype(frame_idx.dtype, np.integer):
            idxs = np.where((self.frame_idxs[:, None] == frame_idx[None, :]).any(axis=1))
            return self.extract(idxs)
        else:
            raise IndexError('Only integers and integer arrays are valid indices')

    def __iter__(self):
        uniq_idxs = np.unique(self.frame_idxs)
        for frame_idx in uniq_idxs:
            idxs = np.where(self.frame_idxs == frame_idx)
            yield FrameStreaks(self.raw_lines[idxs], self.exp_set)

    def kout(self, theta):
        """
        Return reciprocal vectors of detected diffraction streaks in rotational scan.

        theta - angles of rotation
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout(theta=theta_val)
            kout_list.append(rec_vec.kout)
            kin_list.append(rec_vec.kin)
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def kout_ref(self, theta, pixels=25):
        """
        Return refined reciprocal vectors of detected diffraction streaks in rotational scan.

        thetas - angles of rotation
        pixels - pixel distance between two adjacent streaks considered to be collapsed
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout(theta=theta_val)
            kout_list.append(rec_vec.kout)
            kin_list.append(rec_vec.kin)
        groups = TiltGroups(kout=kout_list,
                            kin=kin_list,
                            threshold=self.exp_set.pixtoq(pixels))
        ref_kin = groups.ref_kin()
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(ref_kin))

    def kout_streaks(self, theta):
        """
        Return reciprocal streaks of detected diffraction streaks in rotational scan.

        thetas - angles of rotation
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout_streaks(theta=theta_val)
            kout_list.append(rec_vec.kout)
            kin_list.append(rec_vec.kin)
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def kout_index(self, theta):
        """
        Return reciprocal vectors of indexing points in rotational scan.

        thetas - angles of rotation
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout_index(theta=theta_val)
            kout_list.append(rec_vec.kout)
            kin_list.append(rec_vec.kin)
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def full_index_refine(self, rec_basis, theta, num_ap, n_isl=20, pop_size=50,
                          gen_num=2000, pos_tol=(0.007, 0.014, 0.06), rb_tol=0.12):
        """
        Return refinement problems archipelago

        rec_basis - preliminary reciprocal lattice basis vectors matrix
        theta - angles of rotation
        num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axes
        n_isl - number of islands of one frame
        pop_size - population size
        gen_num - maximum generations number of the refinement algorithm
        pos_tol - relative sample position tolerance
        rb_tol - lattice basis vectors matrix tolerance
        """
        archi = pygmo.archipelago()
        for frame_strks, theta_val in zip(iter(self), theta):
            frame_basis = rec_basis.dot(self.exp_set.rotation_matrix(-theta_val).T)
            prob = frame_strks.rot_index_refine(rec_basis=frame_basis,
                                                num_ap=num_ap,
                                                pos_tol=pos_tol,
                                                rb_tol=rb_tol)
            pops = [pygmo.population(size=pop_size, prob=prob, b=pygmo.mp_bfe()) for _ in range(n_isl)]
            for pop in pops:
                archi.push_back(algo=pygmo.de(gen_num), pop=pop)
        return archi

    def rot_index_refine(self, rec_basis, theta, num_ap, n_isl=20, pop_size=50, gen_num=2000,
                         pos_tol=(0.007, 0.014, 0.06), size_tol=0.05, ang_tol=0.09):
        """
        Return refinement problems archipelago

        rec_basis - preliminary reciprocal lattice basis vectors matrix
        theta - angles of rotation
        num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axes
        n_isl - number of islands of one frame
        pop_size - population size
        gen_num - maximum generations number of the refinement algorithm
        pos_tol - relative sample position tolerance
        size_tol - lattice basis vectors length tolerance
        ang_tol - rotation anlges tolerance
        """
        archi = pygmo.archipelago()
        for frame_strks, theta_val in zip(iter(self), theta):
            frame_basis = rec_basis.dot(self.exp_set.rotation_matrix(-theta_val).T)
            prob = frame_strks.rot_index_refine(rec_basis=frame_basis,
                                                num_ap=num_ap,
                                                pos_tol=pos_tol,
                                                size_tol=size_tol,
                                                ang_tol=ang_tol)
            pops = [pygmo.population(size=pop_size, prob=prob, b=pygmo.mp_bfe()) for _ in range(n_isl)]
            for pop in pops:
                archi.push_back(algo=pygmo.de(gen_num), pop=pop)
        return archi

    def save(self, out_file):
        """
        Save detected diffraction streaks to an HDF5 file

        out_file - h5py file object
        """
        streaks_group = out_file.create_group('streaks')
        streaks_group.create_dataset('frame_idxs', data=self.frame_idxs)
        streaks_group.create_dataset('lines', data=self.raw_lines)
        self.exp_set.save(out_file)

class RecVectors():
    """
    Diffraction peaks in reciprocal space class

    kout - outcoming wavevectors
    kin - incoming wavevectors
    """
    def __init__(self, kout, kin):
        self.kout, self.kin, self.scat_vec = kout, kin, kout - kin

    @property
    def size(self):
        return self.scat_vec.shape[0]

    @property
    def range(self):
        return self.scat_vec.max(axis=0) - self.scat_vec.min(axis=0)

    def grid(self, size):
        """
        Return reciprocal vectors in a grid of shape (size, size, size)

        size - grid size
        """
        return utils.make_grid(points=self.scat_vec,
                               values=np.ones(self.size, dtype=np.float64),
                               size=size)

    def fft(self, size):
        """
        Return fast fourier transform of reciprocal vectors

        size - output grid size
        """
        grid = self.grid(size)
        return np.abs(np.fft.fftshift(np.fft.fftn(grid))**2)

    def fft_peaks(self, size=100, gauss_sigma=1.5, threshold=5):
        """
        Return fourier transform peak positions sorted by norm value

        size - fourier transform grid size
        gauss_sigma - gaussian filter sigma value
        threshold - peak detetion threshold
        """
        fft_grid = self.fft(size)
        fft_blur = gaussian_filter(fft_grid.astype(np.float64), gauss_sigma)
        peak_mask = fft_blur > fft_blur.mean() + threshold * fft_blur.std()
        peak_labels, peak_num = label(peak_mask)
        peaks = np.array(maximum_position(fft_blur,
                                          labels=peak_labels,
                                          index=np.arange(1, peak_num + 1)))
        peaks = peaks - size / 2 * np.ones(3)
        sort_mask = (peaks * peaks).sum(axis=1).argsort()
        return peaks[sort_mask[1:]]

    def index(self, size=100, gauss_sigma=1.5, threshold=5):
        """
        Return indexing solution

        size - fourier transform grid size
        gauss_sigma - gaussian filter sigma value
        threshold - peak detetion threshold
        """
        peaks = self.fft_peaks(size, gauss_sigma, threshold)
        axes = peaks[None, 0]
        idxs = utils.find_reduced(peaks, axes)[0]
        axes = np.concatenate((axes, peaks[None, idxs[0]]))
        idxs = utils.find_reduced(peaks, axes)[0]
        return np.concatenate((axes, peaks[None, idxs[0]])) * self.range**-1

class AbcCBI(metaclass=ABCMeta):
    """
    Abstract Convergent beam indexer (CBI) class

    streaks - FrameStreaks or ScanStreaks class object
    num_ap = [na_x, na_y] - convergent beam numerical apertures in x- and y-axis
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

    @abstractmethod
    def kout_exp(self, vec):
        pass

    @abstractmethod
    def voting_hkl(self, vec, kout_exp):
        pass

    @abstractmethod
    def voting_vectors(self, vec, kout_exp):
        pass

    def det_pts(self, kout_x, kout_y, vec):
        """
        Return diffraction streaks locations at the detector plane
        """
        theta, phi = np.arccos(np.sqrt(1 - kout_x**2 - kout_y**2)), np.arctan2(kout_y, kout_x)
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

    def rec_vectors(self, vec):
        """
        Return optimal reciprocal lattice vectors
        """
        return self.hkl_idxs(vec).dot(self.rec_basis(vec))

    def gradient(self, d_vec):
        """
        Return the target function gradient value
        """
        return pygmo.estimate_gradient(self.fitness, d_vec)

    def get_name(self):
        return "A convergent beam indexing problem"

class FrameCBI(AbcCBI):
    """
    Abstract frame refinement class

    streaks - FrameStreaks class object
    num_ap = [na_x, na_y] - convergent beam numerical apertures in x- and y-axis
    rec_basis - Reciprocal lattice basis vectors matrix
    tol - tolerance defining vector bounds
    pen_coeff - fitness penalty coefficient
    """
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

    def good_idxs(self, vec, na_ext):
        """
        Return indices of good streaks lying inside extended pupil bounds na_ext
        """
        return utils.reduce_streaks(kout_exp=self.kout_exp(vec), hkl_idxs=self.hkl_idxs(vec),
                                    rec_basis=self.rec_basis(vec), na_x=self.num_ap[0], na_y=self.num_ap[1],
                                    na_ext_x=na_ext[0], na_ext_y=na_ext[1], pen_coeff=self.pen_coeff)

    def i_sigma(self, vec, cor_data, background, structure, width, na_ext):
        """
        Return good streaks intensities and Poisson noise

        vec - refinement vector
        cor_data - background subtracted diffraction pattern image
        background - background image
        structure - binary structure for binary dilation
        width - diffraction streaks width
        """
        kin = self.kout_exp(vec) - self.rec_vectors(vec)[:, None]
        source_streaks = self.det_pts(kin[..., 0], kin[..., 1], vec)
        idxs = self.good_idxs(vec, na_ext)
        return utils.i_sigma_frame(streaks=self.lines[idxs], source_streaks=source_streaks[idxs], cor_data=cor_data,
                                   background=background, structure=structure, width=width)

class ScanCBI(AbcCBI):
    """
    Abstract scan refinement class

    streaks - ScanStreaks class object
    num_ap = [na_x, na_y] - convergent beam numerical apertures in x- and y-axis
    rec_basis - Reciprocal lattice basis vectors matrix
    thetas - angles of rotation
    tol = [pos_tol, th_tol, rot_tol, rb_tol] - tolerance defining vector bounds
    pen_coeff - fitness penalty coefficient
    """
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