"""
feat_detect.py - feature detection on convergent diffraction pattern module
"""
from abc import ABCMeta, abstractmethod
import numpy as np
import pygmo
from skimage.transform import probabilistic_hough_line
from scipy.ndimage import gaussian_filter, label, maximum_position
from cv2 import createLineSegmentDetector
from . import utils
from .grouper import TiltGroups
from .indexer import FCBI

class FrameSetup():
    """
    Detector frame experimental setup class

    pix_size - detector pixel size
    det_pos - detector position in respect to the sample [mm]
    """
    def __init__(self, pix_size, det_pos):
        self.pix_size, self.det_pos = pix_size, det_pos

    def pixtoq(self, pixels):
        """
        Convert detector distance in pixels to distance in reciprocal space
        """
        return pixels * self.pix_size / self.det_pos[2]

    def kout_exp(self, det_pts):
        """
        Project detected diffraction streaks to reciprocal space vectors in [a.u.]

        det_pts - detector points [mm]
        """
        angles = np.arctan2(det_pts[..., 1], det_pts[..., 0])
        thetas = np.arctan(np.sqrt(det_pts[..., 0]**2 + det_pts[..., 1]**2) / self.det_pos[2])
        return np.stack((np.sin(thetas) * np.cos(angles),
                         np.sin(thetas) * np.sin(angles),
                         np.cos(thetas)), axis=-1)

    def det_pts(self, k_out):
        """
        Return diffraction streaks locations at the detector plane

        k_out - outcoming wavevectors
        """
        theta, phi = np.arccos(k_out[..., 2]), np.arctan2(k_out[..., 1], k_out[..., 0])
        det_x = self.det_pos[2] * np.tan(theta) * np.cos(phi)
        det_y = self.det_pos[2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + self.det_pos[:2]) / self.pix_size

    def save(self, out_file):
        """
        Save experiment settings to an HDF5 file

        out_file - h5py File object
        """
        out_group = out_file.create_group("experiment_settings")
        out_group.create_dataset('pix_size', data=self.pix_size)
        out_group.create_dataset('det_pos', data=self.det_pos)

class ScanSetup(FrameSetup):
    """
    Detector tilt scan experimental setup class

    pix_size - detector pixel size
    det_pos - detector position in respect to the sample [mm]
    rot_axis - axis of rotation
    """
    def __init__(self, pix_size, det_pos, rot_axis):
        super(ScanSetup, self).__init__(pix_size, det_pos)
        self.axis = rot_axis

    def rotation_matrix(self, theta):
        """
        Return roational matrix for a given angle of rotation
        """
        return utils.rotation_matrix(self.axis, theta)

    def save(self, out_file):
        """
        Save experiment settings to an HDF5 file

        out_file - h5py File object
        """
        out_group = out_file.create_group("experiment_settings")
        out_group.create_dataset('pix_size', data=self.pix_size)
        out_group.create_dataset('det_pos', data=self.det_pos)
        out_group.create_dataset('rot_axis', data=self.axis)

class LineDetector(metaclass=ABCMeta):
    """
    Abstract line detector class
    """
    @staticmethod
    @abstractmethod
    def _refiner(lines, width):
        pass

    @abstractmethod
    def _detector(self, frame):
        pass

    def det_frame_raw(self, frame):
        return np.array([[[x0, y0], [x1, y1]] for (x0, y0), (x1, y1) in self._detector(frame)])

    def det_frame(self, frame, exp_set, width=10):
        """
        Return FrameStreaks class object of detected lines

        frame - diffraction pattern
        exp_set - FrameSetup class object
        width - line width [pixels]
        """
        raw_lines = self.det_frame_raw(frame).astype(float)
        lines = self._refiner(raw_lines, width)
        lines = self._refiner(lines, width)
        return FrameStreaks(lines, exp_set)

    def det_scan_raw(self, scan):
        """
        Return Hough Line Transofrm raw detected lines for given scan
        """
        return [self.det_frame_raw(frame) for frame in scan]

    def det_scan(self, data, exp_set, width=10):
        """
        Return ScanStreaks class obect of detected lines

        scan - rotational scan
        exp_set - FrameSetup class object
        width - line width [pixels]
        """
        return ScanStreaks.import_series([self.det_frame(frame, exp_set, width) for frame in data])

class HoughLineDetector(LineDetector):
    """
    Hough line transform line detector class

    threshold - line detection threshold
    line_length - maximal line length
    line_gap - maximal line gap to consider two lines as one
    dth - angle parameter spacing
    """
    def __init__(self, threshold=10, line_length=15, line_gap=3, dth=np.pi/500):
        self.threshold, self.line_length, self.line_gap = threshold, line_length, line_gap
        self.thetas = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi / dth), endpoint=True)

    _refiner = staticmethod(utils.hl_refiner)

    def _detector(self, frame):
        return probabilistic_hough_line(frame,
                                        threshold=self.threshold,
                                        line_length=self.line_length,
                                        line_gap=self.line_gap,
                                        theta=self.thetas)

class LineSegmentDetector(LineDetector):
    """
    Line Segment Detector line detection class

    scale - scale of the input image
    sigma_scale - the sigma of gaussian filter
    lag_eps - detection threshold

    To read about the underlying theory see the article:
    http://www.ipol.im/pub/art/2012/gjmr-lsd/?utm_source=doi
    """
    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0):
        self.detector = createLineSegmentDetector(_scale=scale,
                                                  _sigma_scale=sigma_scale,
                                                  _log_eps=log_eps)

    _refiner = staticmethod(utils.lsd_refiner)

    def _detector(self, frame):
        cap = np.mean(frame[frame != 0]) + np.std(frame[frame != 0])
        img = utils.arraytoimg(np.clip(frame, 0, cap))
        return self.detector.detect(img)[0][:, 0].reshape((-1, 2, 2)).astype(np.float64)

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
        return np.tile(np.array([0, 0, 1]), (self.size, 1))

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

    def i_sigma(self, hkl_idxs, cor_data, bgd):
        """
        Return detected streaks intensities (I) and Poisson noise (sigma)

        cor_data - background subtracted data
        bgd - background data
        hkl_idxs - hkl indices
        """
        hkl_uniq = np.unique(hkl_idxs, axis=0)
        i_list, s_list = [], []
        for hkl in hkl_uniq:
            idxs = np.where((hkl_idxs == hkl).all(axis=1))
            mask = utils.streaks_mask(lines=self.raw_lines[idxs], structure=utils.STRUCT,
                                      width=3, shape_x=bgd.shape[0], shape_y=bgd.shape[1])
            i_tot, sigma = utils.i_sigma(mask, cor_data, bgd)
            i_list.append(i_tot)
            s_list.append(sigma)
        return hkl_uniq, np.array(i_list), np.array(s_list)

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
                          kin=np.tile(self.kin[:, None], (1, 2, 1)))

    def kout_index(self, theta):
        """
        Return reciprocal points of indexing points.

        theta - angle of rotation
        """
        index_pts = self.index_pts()
        rot_m = self.exp_set.rotation_matrix(theta)
        kout = self.exp_set.kout_exp(index_pts)
        return RecVectors(kout=kout.dot(rot_m.T), kin=self.kin.dot(rot_m.T))

    def index_refine(self, rec_basis, num_ap, tol=(0.05, 0.015)):
        """
        Return refinement problem population

        rec_basis - reciprocal lattice basis vectors matrix
        num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axes
        tol - tolerance defining refined solution bounds
        """
        full_tf = FCBI(lines=self.raw_lines,
                       exp_set=self.exp_set,
                       rec_basis=rec_basis,
                       num_ap=num_ap,
                       tol=tol)
        return pygmo.problem(full_tf)

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

    def i_sigma(self, hkl_idxs, cor_data, bgd):
        """
        Return detected streaks intensities (I) and Poisson noise (sigma)

        cor_data - background subtracted data
        bgd - background data
        hkl_idxs - hkl indices
        """
        frame_idxs = np.unique(self.frame_idxs)
        i_dict, s_dict, hkl_dict = {}, {}, {}
        for frame_idx in frame_idxs:
            idxs = np.where(self.frame_idxs == frame_idx)
            streaks = FrameStreaks(self.raw_lines[idxs], self.exp_set)
            hkl_arr, i_arr, s_arr = streaks.i_sigma(hkl_idxs=hkl_idxs[idxs],
                                                    cor_data=cor_data[frame_idx],
                                                    bgd=bgd[frame_idx])
            hkl_dict[frame_idx] = hkl_arr
            i_dict[frame_idx] = i_arr
            s_dict[frame_idx] = s_arr
        return hkl_dict, i_dict, s_dict

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

    def index_refine(self, theta, num_ap, n_isl=20, pop_size=50, gen_num=2000, tol=(0.05, 0.015)):
        """
        Return refinement problems archipelago

        num_ap = [num_ap_x, num_ap_y] - convergent beam numerical apertures in x- and y-axes
        n_isl - number of islands of one frame
        pop_size - population size
        tol - tolerance defining refined solution bounds
        """
        rec_basis = utils.rec_basis(self.kout_ref(theta=theta).index())
        archi = pygmo.archipelago()
        for frame_strks, theta_val in zip(iter(self), theta):
            frame_basis = rec_basis.dot(self.exp_set.rotation_matrix(-theta_val).T)
            prob = frame_strks.index_refine(rec_basis=frame_basis,
                                            num_ap=num_ap,
                                            tol=tol)
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

    def save_intensities(self, out_file, raw_data, background):
        """
        Save detected diffraction streaks and intensities to an HDF5 file

        out_file - h5py file object
        raw_data - raw diffraction patterns
        backgroun - background noises
        """
        self.save(out_file)
        data_group = out_file.create_group('streaks_data')
        data_group.create_dataset('intensities', data=self.intensities(raw_data))
        data_group.create_dataset('snr', data=self.snr(raw_data, background))

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
    