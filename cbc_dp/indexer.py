"""
indexer.py - convergent beam crystallography indexer implementation
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import pygmo
from scipy.ndimage import gaussian_filter, label, maximum_position
from .utils import find_reduced, make_grid, scan_ps, scan_rb, fcbi_rb, rcbi_rb
from .utils import kout_frame, kin_frame, vot_vec_frame, vot_idxs_frame, fit_frame, fit_idxs_frame
from .utils import kout_scan, kin_scan, vot_vec_scan, vot_idxs_scan, fit_scan, fit_idxs_scan
from .grouper import TiltGroups
from .model import RecBasis

class FrameStreaks():
    """
    Detected diffraction streaks on one frame class

    lines - detected lines
    zero - zero output wavevector point
    exp_set - ScanSetup class object
    """
    def __init__(self, lines, exp_set):
        self.raw_lines, self.exp_set = lines, exp_set

    @property
    def kin(self):
        return np.tile(np.array([0., 0., 1.]), (self.size, 1))

    @property
    def size(self):
        return self.raw_lines.shape[0]

    def __iter__(self):
        for line in self.raw_lines:
            yield line

    def __getitem__(self, idx):
        return FrameStreaks(self.raw_lines[idx], self.exp_set)

    def index_pts(self):
        """
        Return indexing points
        """
        lines = self.raw_lines * self.exp_set.pix_size - self.exp_set.smp_pos[:2]
        taus = lines[:, 1] - lines[:, 0]
        taus = taus / np.sqrt(taus[:, 0]**2 + taus[:, 1]**2)[:, np.newaxis]
        products = lines[:, 0, 1] * taus[:, 0] - lines[:, 0, 0] * taus[:, 1]
        return np.stack((-taus[:, 1] * products, taus[:, 0] * products), axis=1)

    def kout(self):
        """
        Return reciprocal vectors of detected diffraction streaks.
        """
        kout = self.exp_set.kout_exp(self.raw_lines.mean(axis=1))
        return RecVectors(kout=kout, kin=self.kin)

    def kout_streaks(self):
        """
        Return reciprocal vectors of detected diffraction streaks.
        """
        kout = self.exp_set.kout_exp(self.raw_lines)
        return RecVectors(kout=kout, kin=np.tile(self.kin[:, None], (1, 2, 1)))

    def kout_index(self):
        """
        Return reciprocal points of indexing points.
        """
        index_pts = self.index_pts()
        kout = self.exp_set.kout_exp(index_pts)
        return RecVectors(kout=kout, kin=self.kin)

    def full_index(self, rec_basis, frame_idx, f_tol, smp_tol, rb_tol, ang_tol):
        """
        Return a population of reciprocal lattice basis vectors matrix refinement problem

        rec_basis - RecBasis class object
        frame_idx - frame index
        f_tol - focus position tolerance
        smp_tol - relative sample position tolerance
        rb_tol - lattice basis vectors matrix tolerance
        ang_tol - rotation angles tolerance
        """
        full_tf = FCBI(streaks=self, frame_idx=frame_idx, rec_basis=rec_basis,
                       f_tol=f_tol, smp_tol=smp_tol, rb_tol=rb_tol, ang_tol=ang_tol)
        return pygmo.problem(full_tf)

    def rot_index(self, rec_basis, frame_idx, f_tol, smp_tol, rb_tol, ang_tol):
        """
        Return a population of reciprocal lattice rotation refinement problem

        rec_basis - RecBasis class object
        frame_idx - frame index
        f_tol - focus position tolerance
        smp_tol - relative sample position tolerance
        rb_tol - lattice basis vectors matrix tolerance
        ang_tol - rotation angles tolerance
        """
        rot_tf = RCBI(streaks=self, rec_basis=rec_basis, frame_idx=frame_idx,
                      f_tol=f_tol, smp_tol=smp_tol, rb_tol=rb_tol, ang_tol=ang_tol)
        return pygmo.problem(rot_tf)

class ScanStreaks(FrameStreaks):
    """
    Detected diffraction streaks of a rotational scan class

    raw_lines - detected lines
    exp_set - ScanSetup class object
    frame_idxs - frame indices
    """
    err_str = 'Only integers, slice, integer arrays, and list of integers are valid indices'

    def __init__(self, raw_lines, exp_set, frame_idxs):
        self.frame_idxs = frame_idxs
        self.frames, self.idxs = np.unique(self.frame_idxs, return_index=True)
        self.idxs = np.append(self.idxs, raw_lines.shape[0])
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
        if isinstance(frame_idx, (int, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)):
            return FrameStreaks(self.raw_lines[self.idxs[frame_idx]:self.idxs[frame_idx + 1]],
                                self.exp_set)
        elif isinstance(frame_idx, (slice, np.ndarray, list)):
            try:
                start_idxs, stop_idxs = self.idxs[frame_idx], self.idxs[1:][frame_idx]
            except IndexError as err:
                raise IndexError(self.err_str) from err
            streaks, frame_idxs = [], []
            for start, stop in zip(start_idxs, stop_idxs):
                streaks.append(self.raw_lines[start:stop])
                frame_idxs.append(self.frame_idxs[start:stop])
            return ScanStreaks(np.concatenate(streaks), self.exp_set, np.concatenate(frame_idxs))
        else:
            raise IndexError(self.err_str)

    def __iter__(self):
        for start, stop in zip(self.idxs[:-1], self.idxs[1:]):
            yield FrameStreaks(self.raw_lines[start:stop], self.exp_set)

    def kout(self):
        """
        Return reciprocal vectors of detected diffraction streaks in rotational scan.

        theta - angles of rotation
        """
        kout_list, kin_list = [], []
        for frame_idx, frame_strks in enumerate(iter(self)):
            rot_m = self.exp_set.rotation_matrix(frame_idx)
            rec_vec = frame_strks.kout()
            kout_list.append(rec_vec.kout.dot(rot_m.T))
            kin_list.append(rec_vec.kin.dot(rot_m.T))
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def kout_ref(self, pixels=25):
        """
        Return refined reciprocal vectors of detected diffraction streaks in rotational scan.

        pixels - pixel distance between two adjacent streaks considered to be collapsed
        """
        kout_list, kin_list = [], []
        for frame_idx, frame_strks in enumerate(iter(self)):
            rot_m = self.exp_set.rotation_matrix(frame_idx)
            rec_vec = frame_strks.kout()
            kout_list.append(rec_vec.kout.dot(rot_m.T))
            kin_list.append(rec_vec.kin.dot(rot_m.T))
        groups = TiltGroups(kout=kout_list,
                            kin=kin_list,
                            threshold=self.exp_set.pixtoq(pixels))
        ref_kin = groups.ref_kin()
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(ref_kin))

    def kout_streaks(self):
        """
        Return reciprocal streaks of detected diffraction streaks in rotational scan.
        """
        kout_list, kin_list = [], []
        for frame_idx, frame_strks in enumerate(iter(self)):
            rot_m = self.exp_set.rotation_matrix(frame_idx)
            rec_vec = frame_strks.kout_streaks()
            kout_list.append(rec_vec.kout.dot(rot_m.T))
            kin_list.append(rec_vec.kin.dot(rot_m.T))
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def kout_index(self):
        """
        Return reciprocal vectors of indexing points in rotational scan.
        """
        kout_list, kin_list = [], []
        for frame_idx, frame_strks in enumerate(iter(self)):
            rot_m = self.exp_set.rotation_matrix(frame_idx)
            rec_vec = frame_strks.kout_index()
            kout_list.append(rec_vec.kout.dot(rot_m.T))
            kin_list.append(rec_vec.kin.dot(rot_m.T))
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def full_index(self, rec_basis, n_isl, pop_size, gen_num,
                   f_tol, smp_tol, rb_tol, ang_tol):
        """
        Return refinement problems archipelago

        rec_basis - RecBasis class object
        n_isl - number of islands of one frame
        pop_size - population size
        gen_num - maximum generations number of the refinement algorithm
        f_tol - focus position tolerance
        smp_tol - relative sample position tolerance
        rb_tol - lattice basis vectors matrix tolerance
        ang_tol - rotation angles tolerance
        """
        archi = pygmo.archipelago()
        for frame_idx, frame_strks in zip(self.frames, iter(self)):
            prob = frame_strks.full_index(rec_basis=rec_basis, frame_idx=frame_idx, f_tol=f_tol,
                                          smp_tol=smp_tol, rb_tol=rb_tol, ang_tol=ang_tol)
            pops = [pygmo.population(size=pop_size, prob=prob, b=pygmo.mp_bfe()) for _ in range(n_isl)]
            for pop in pops:
                archi.push_back(algo=pygmo.de(gen_num), pop=pop)
        return archi

    def rot_index(self, rec_basis, n_isl, pop_size, gen_num,
                   f_tol, smp_tol, rb_tol, ang_tol):
        """
        Return refinement problems archipelago

        rec_basis - RecBasis class object
        n_isl - number of islands of one frame
        pop_size - population size
        gen_num - maximum generations number of the refinement algorithm
        f_tol - focus position tolerance
        smp_tol - relative sample position tolerance
        rb_tol - lattice basis vectors matrix tolerance
        ang_tol - rotation angles tolerance
        """
        archi = pygmo.archipelago()
        algo = pygmo.de(gen_num)
        for frame_idx, frame_strks in zip(self.frames, iter(self)):
            prob = frame_strks.rot_index(rec_basis=rec_basis, frame_idx=frame_idx, f_tol=f_tol,
                                         smp_tol=smp_tol, rb_tol=rb_tol, ang_tol=ang_tol)
            pops = [pygmo.population(size=pop_size, prob=prob, b=pygmo.mp_bfe()) for _ in range(n_isl)]
            for pop in pops:
                archi.push_back(algo=algo, pop=pop)
        return archi

    def scan_index(self, rec_basis, n_isl, pop_size, gen_num,
                   f_tol, smp_tol, rb_tol, ang_tol):
        """
        Return refinement problems archipelago

        rec_basis - RecBasis class object
        n_isl - number of islands of one frame
        pop_size - population size
        gen_num - maximum generations number of the refinement algorithm
        f_tol - focus position tolerance
        smp_tol - relative sample position tolerance
        rb_tol - lattice basis vectors matrix tolerance
        ang_tol - rotation angles tolerance
        """
        archi = pygmo.archipelago()
        prob = ScanCBI(streaks=self, rec_basis=rec_basis, f_tol=f_tol, smp_tol=smp_tol,
                       rb_tol=rb_tol, ang_tol=ang_tol)
        ps = scan_ps(pop_size, self.frames.size)
        pops = [pygmo.population(prob, size=ps, b=pygmo.mp_bfe()) for _ in range(n_isl)]
        algo = pygmo.moead(gen=gen_num)
        for pop in pops:
            archi.push_back(algo=algo, pop=pop)
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
        return make_grid(points=self.scat_vec, size=size,
                         values=np.ones(self.size, dtype=np.float64))

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
        idxs = find_reduced(peaks, axes)[0]
        axes = np.concatenate((axes, peaks[None, idxs[0]]))
        idxs = find_reduced(peaks, axes)[0]
        return np.concatenate((axes, peaks[None, idxs[0]])) * self.range**-1

class AbcCBI(metaclass=ABCMeta):
    """
    Abstract Convergent beam indexer (CBI) class

    streaks - FrameStreaks or ScanStreaks class object
    pen_coeff - fitness penalty coefficient
    """
    mat_shape = (3, 3)
    lower_b, upper_b, info_text = None, None, None

    def __init__(self, streaks, pen_coeff):
        self.lines = streaks.raw_lines * streaks.exp_set.pix_size
        self.pix_size, self.pen_coeff = streaks.exp_set.pix_size, pen_coeff

    @abstractmethod
    def rb_mat(self, vec):
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

    @abstractmethod
    def fit(self, vec, vot_vec, kout_exp):
        pass

    @abstractmethod
    def kin_bounds(self, vec):
        pass

    @abstractmethod
    def fit_idxs(self, vec, vot_vec, kout_exp):
        pass

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
        return self.fit(vec, vot_vec, kout_exp)

    def hkl_idxs(self, vec):
        """
        Return the hkl indices of the optimal reciprocal lattice voting vectors
        """
        kout_exp = self.kout_exp(vec)
        vot_vec = self.voting_vectors(vec, kout_exp)
        hkl_idxs = self.voting_hkl(vec, kout_exp)
        return hkl_idxs[self.fit_idxs(vec, vot_vec, kout_exp)]

    def rec_vectors(self, vec):
        """
        Return optimal reciprocal lattice vectors
        """
        return self.hkl_idxs(vec).dot(self.rb_mat(vec))

    def gradient(self, d_vec):
        """
        Return the target function gradient value
        """
        return pygmo.estimate_gradient(self.fitness, d_vec)

    def det_kin(self, kin_x, kin_y, vec):
        """
        Return diffraction streaks locations at the detector plane
        """
        theta, phi = np.arccos(np.sqrt(1 - kin_x**2 - kin_y**2)), np.arctan2(kin_y, kin_x)
        det_x = vec[2] * np.tan(theta) * np.cos(phi)
        det_y = vec[2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + vec[:2]) / self.pix_size

    def get_name(self):
        return "A convergent beam indexing problem"

    def get_extra_info(self):
        return self.info_text

class FrameCBI(AbcCBI):
    """
    Abstract frame refinement class

    streaks - FrameStreaks class object
    frame_idx - frame index
    rec_basis - RecBasis class object
    tol - tolerance defining vector bounds
    pen_coeff - fitness penalty coefficient
    """
    def __init__(self, streaks, frame_idx, rec_basis, tol, pen_coeff):
        super(FrameCBI, self).__init__(streaks, pen_coeff)
        self.pupil = streaks.exp_set.pupil_bounds(frame_idx)
        eul_ang = streaks.exp_set.euler_angles(frame_idx)
        self._init_bounds(smp_pos=streaks.exp_set.smp_pos, f_pos=streaks.exp_set.f_pos,
                          rec_basis=rec_basis, eul_ang=eul_ang, tol=tol)

    @abstractmethod
    def _init_bounds(self, smp_pos, f_pos, rec_basis, eul_ang, tol):
        pass

    def rec_basis(self, vec):
        """
        Return RecBasis class object
        """
        return RecBasis(rb_mat=self.rb_mat(vec))

    def kin_bounds(self, vec):
        """
        Return pupil bounds in reciprocal space
        """
        return kin_frame(self.pupil, vec[:3])

    def kout_exp(self, vec):
        """
        Generate the experimentally measured diffraction streaks outcoming wavevectors of a frame
        """
        return kout_frame(streaks=self.lines, pt0=vec[3:6])

    def voting_vectors(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting points for the given experimental outcoming
        wavevectors kout_exp
        """
        return vot_vec_frame(kout_exp=kout_exp.mean(axis=1), rec_basis=self.rb_mat(vec),
                             kin=self.kin_bounds(vec))

    def voting_hkl(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting hkl indices for the given experimental outcoming
        wavevectors kout_exp
        """
        return vot_idxs_frame(kout_exp=kout_exp.mean(axis=1), rec_basis=self.rb_mat(vec),
                              kin=self.kin_bounds(vec))

    def fit(self, vec, vot_vec, kout_exp):
        """
        Return fitness value for the given voting vectors array vot_vec and
        experimental outcoming wavevectors kout_exp
        """
        return [fit_frame(vot_vec=vot_vec, kout_exp=kout_exp, kin=self.kin_bounds(vec),
                          pen_coeff=self.pen_coeff)]

    def fit_idxs(self, vec, vot_vec, kout_exp):
        """
        Return the indices of the optimal reciprocal lattice voting vectors
        """
        return fit_idxs_frame(vot_vec=vot_vec, kout_exp=kout_exp,
                              kin=self.kin_bounds(vec), pen_coeff=self.pen_coeff)

    def det_kout(self, kout_x, kout_y, vec):
        """
        Return diffraction streaks locations at the detector plane
        """
        theta, phi = np.arccos(np.sqrt(1 - kout_x**2 - kout_y**2)), np.arctan2(kout_y, kout_x)
        det_x = vec[5] * np.tan(theta) * np.cos(phi)
        det_y = vec[5] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + vec[3:5]) / self.pix_size

class ScanCBI(AbcCBI):
    """
    Abstract scan refinement class (incomplete)

    streaks - ScanStreaks class object
    rec_basis - RecBasis class object
    tol = [pos_tol, th_tol, rot_tol, rb_tol] - tolerance defining vector bounds
    pen_coeff - fitness penalty coefficient
    """
    info_text_str = ("Dimensions: {total:d} in total",
                     "9 - reciprocal lattice basis vectors matrix",
                     "{smp_pos:d} - sample coordinates",
                     "{eul_ang:d} - Euler angles")

    def __init__(self, streaks, rec_basis, f_tol, smp_tol, rb_tol, ang_tol, pen_coeff=1.):
        super(ScanCBI, self).__init__(streaks, pen_coeff)
        self.pupil = (streaks.exp_set.pupil.reshape((-1, 2, 2)) * self.pix_size)[streaks.frames]
        self.frames, self.idxs = streaks.frames, streaks.idxs
        tol = (f_tol, smp_tol, rb_tol, ang_tol)
        eul_ang = streaks.exp_set.eul_ang[self.frames].ravel()
        self._init_bounds(smp_pos=streaks.exp_set.smp_pos, f_pos=streaks.exp_set.f_pos,
                          rec_basis=rec_basis, eul_ang=eul_ang, tol=tol)

    def _init_bounds(self, smp_pos, f_pos, rec_basis, eul_ang, tol):
        pt0_lb = np.tile((1 - np.array(tol[1])) * smp_pos, self.frames.size)
        pt0_ub = np.tile((1 + np.array(tol[1])) * smp_pos, self.frames.size)
        self.lower_b = np.concatenate(((1 - np.array(tol[0])) * f_pos,
                                       rec_basis.rb_mat.ravel() - tol[2],
                                       pt0_lb, eul_ang - tol[3]))
        self.upper_b = np.concatenate(((1 + np.array(tol[0])) * f_pos,
                                       rec_basis.rb_mat.ravel() + tol[2],
                                       pt0_ub, eul_ang + tol[3]))

    @property
    def info_text(self):
        return "\n".join(self.info_text_str).format(total=self.lower_b.size,
                                                    smp_pos=3 * self.frames.size,
                                                    eul_ang=3 * self.frames.size)

    def kin_bounds(self, vec):
        """
        Return pupil bounds in reciprocal space
        """
        return kin_scan(self.pupil, vec[:3])

    def rb_mat(self, vec):
        """
        Return rectangular lattice basis vectors matrix
        """
        return vec[3:12].reshape(self.mat_shape)

    def rb_scan(self, vec):
        """
        Return rectangular lattice basis matrices vectors of a scan
        """
        return scan_rb(eul_ang=vec[12 + 3 * self.frames.size:],
                       frames=self.frames, rb_mat=vec[3:12])

    def kout_exp(self, vec):
        """
        Generate the experimentally measured diffraction streaks outcoming wavevectors of a scan
        """
        return kout_scan(streaks=self.lines, pts0=vec[12:12 + 3 * self.frames.size], idxs=self.idxs)

    def voting_vectors(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting points for the given experimental outcoming
        wavevectors kout_exp
        """
        return vot_vec_scan(kout_exp=kout_exp.mean(axis=1), rec_basis=self.rb_scan(vec),
                            kin=self.kin_bounds(vec), idxs=self.idxs)

    def voting_hkl(self, vec, kout_exp):
        """
        Return the reciprocal lattice voting hkl indices for the given experimental outcoming
        wavevectors kout_exp
        """
        return vot_idxs_scan(kout_exp=kout_exp.mean(axis=1), rec_basis=self.rb_scan(vec),
                             kin=self.kin_bounds(vec), idxs=self.idxs)

    def fit(self, vec, vot_vec, kout_exp):
        """
        Return fitness value for the given voting vectors array vot_vec and
        experimental outcoming wavevectors kout_exp
        """
        return fit_scan(vot_vec=vot_vec, kout_exp=kout_exp, kin=self.kin_bounds(vec),
                        idxs=self.idxs, pen_coeff=self.pen_coeff)

    def fit_idxs(self, vec, vot_vec, kout_exp):
        """
        Return the indices of the optimal reciprocal lattice voting vectors
        """
        return fit_idxs_scan(vot_vec=vot_vec, kout_exp=kout_exp, kin=self.kin_bounds(vec),
                             idxs=self.idxs, pen_coeff=self.pen_coeff)

    def get_nobj(self):
        """
        Return fitness vector dimension
        """
        return self.frames.size

    def det_kout(self, idx, kout_x, kout_y, vec):
        """
        Return diffraction streaks locations at the detector plane
        """
        theta, phi = np.arccos(np.sqrt(1 - kout_x**2 - kout_y**2)), np.arctan2(kout_y, kout_x)
        det_x = vec[12 + 3 * idx + 2] * np.tan(theta) * np.cos(phi)
        det_y = vec[12 + 3 * idx + 2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + vec[12 + 3 * idx:12 + 3 * idx + 2]) / self.pix_size

class FCBI(FrameCBI):
    """
    Convergent beam crystallography indexer class.
    Argument vector is comprised of detector relative position,
    reciprocal lattice basis vector lengths, and Euler angles for every basis vector.
    Euler angles with Bunge convention are used.

    streaks                             - FrameStreaks class object
    rec_basis                           - RecBasis class object
    rot_mat                             - rotation matrix
    (f_tol, smp_tol, rb_tol, ang_tol)   - focus, sample coordinates,
                                          reciporcal basis vector lengths,
                                          and rotation angles tolerances [0.0 - 1.0]
    pen_coeff                           - fitness penalty coefficient
    """
    info_text = "\n".join(("Dimensions: 15 in total", "3 - sample coordinates",
                           "3 - reciprocal lattice basis vector lengths",
                           "9 - Euler angles for every basis vector"))

    def __init__(self, streaks, frame_idx, rec_basis, f_tol, smp_tol, rb_tol, ang_tol, pen_coeff=1.):
        tol = (f_tol, smp_tol, rb_tol, ang_tol)
        super(FCBI, self).__init__(streaks, frame_idx, rec_basis, tol, pen_coeff)

    def _init_bounds(self, smp_pos, f_pos, rec_basis, eul_ang, tol):
        self.or_mat = rec_basis.or_mat
        self.lower_b = np.concatenate(((1 - np.array(tol[0])) * f_pos,
                                       (1 - np.array(tol[1])) * smp_pos,
                                       (1 - tol[2]) * rec_basis.sizes,
                                       np.tile(eul_ang + tol[2], 3)))
        self.upper_b = np.concatenate(((1 + np.array(tol[0])) * f_pos,
                                       (1 + np.array(tol[1])) * smp_pos,
                                       (1 + tol[2]) * rec_basis.sizes,
                                       np.tile(eul_ang + tol[2], 3)))

    def rb_mat(self, vec):
        """
        Return rectangular lattice basis vectors for a vector
        """
        return fcbi_rb(or_mat=self.or_mat, rb_sizes=vec[6:9], eul_ang=vec[9:])

class RCBI(FrameCBI):
    """
    Convergent beam crystallography indexer class.
    Argument vector is comprised of detector relative position,
    reciprocal lattice basis vector lengths, and Euler angles for basis vectors matrix.
    Euler angles with Bunge convention are used

    streaks                             - FrameStreaks class object
    rec_basis                           - RecBasis class object
    rot_mat                             - rotation matrix
    (f_tol, smp_tol, rb_tol, ang_tol)   - focus, sample coordinates,
                                          reciporcal basis vector lengths,
                                          and rotation angles tolerances [0.0 - 1.0]
    pen_coeff                           - fitness penalty coefficient
    """
    info_text = "\n".join(("Dimensions: 9 in total", "3 - sample coordinates",
                           "3 - reciprocal lattice basis vectors lengths",
                           "3 - orientation matrix angles"))

    def __init__(self, streaks, frame_idx, rec_basis, f_tol, smp_tol, rb_tol, ang_tol, pen_coeff=1.):
        tol = (f_tol, smp_tol, rb_tol, ang_tol)
        super(RCBI, self).__init__(streaks, frame_idx, rec_basis, tol, pen_coeff)

    def _init_bounds(self, smp_pos, f_pos, rec_basis, eul_ang, tol):
        self.or_mat = rec_basis.or_mat
        self.lower_b = np.concatenate(((1 - np.array(tol[0])) * f_pos,
                                       (1 - np.array(tol[1])) * smp_pos,
                                       (1 - tol[2]) * rec_basis.sizes,
                                       eul_ang - tol[3]))
        self.upper_b = np.concatenate(((1 + np.array(tol[0])) * f_pos,
                                       (1 + np.array(tol[1])) * smp_pos,
                                       (1 + tol[2]) * rec_basis.sizes,
                                       eul_ang + tol[3]))

    def rb_mat(self, vec):
        """
        Return orthogonal orientation matrix based on euler angles
        """
        return rcbi_rb(or_mat=self.or_mat, rb_sizes=vec[6:9], eul_ang=vec[9:])
    