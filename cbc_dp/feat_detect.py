"""
feat_detect.py - feature detection on convergent diffraction pattern module
"""
from itertools import accumulate
from operator import add
from abc import ABCMeta, abstractmethod
import numpy as np
from skimage.transform import probabilistic_hough_line
from skimage.draw import line_aa
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, maximum_position
from cv2 import createLineSegmentDetector
from . import utils
from .grouper import TiltGroups

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
    def _refiner(lines, taus, d_tau, d_n):
        pass

    @abstractmethod
    def _detector(self, frame):
        pass

    def det_frame_raw(self, frame):
        return np.array([[[x0, y0], [x1, y1]] for (x0, y0), (x1, y1) in self._detector(frame)])

    def det_frame(self, frame, exp_set, d_tau=1.5, d_n=1.):
        """
        Return FrameStreaks class object of detected lines

        frame - diffraction pattern
        exp_set - FrameSetup class object
        d_tau - tangent detection error
        d_n - radial detection error
        """
        frame_strks = FrameStreaks(self.det_frame_raw(frame), exp_set)
        ref_lines = type(self)._refiner(lines=frame_strks.lines,
                                        taus=frame_strks.taus,
                                        d_tau=d_tau, d_n=d_n)
        return FrameStreaks((ref_lines + exp_set.det_pos[:2]) / exp_set.pix_size, exp_set)

    def det_scan_raw(self, scan):
        """
        Return Hough Line Transofrm raw detected lines for given scan
        """
        return [self.det_frame_raw(frame) for frame in scan]

    def det_scan(self, data, exp_set, d_tau=1.5, d_n=1.):
        """
        Return ScanStreaks class obect of detected lines

        scan - rotational scan
        exp_set - FrameSetup class object
        d_tau - tangent detection error [pixels]
        d_n - radial detection error [pixels]
        """
        return ScanStreaks.import_series([self.det_frame(frame, exp_set, d_tau, d_n) for frame in data])

class HoughLineDetector(LineDetector):
    """
    Hough line transform line detector class

    threshold - line detection threshold
    line_length - maximal line length
    line_gap - maximal line gap to consider two lines as one
    dth - angle parameter spacing
    """
    def __init__(self, threshold, line_length, line_gap, dth):
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

    def intensities(self, raw_data):
        """
        Return detected streaks intensities

        raw_data - raw diffraction pattern
        """
        ints = []
        for line in self.raw_lines.astype(np.int):
            rows, columns, val = line_aa(line[0, 1], line[0, 0], line[1, 1], line[1, 0])
            ints.append((raw_data[rows, columns] * val).sum())
        return np.array(ints)

    def snr(self, raw_data, background):
        """
        Return detected streaks signal to noise ratio

        raw_data - raw diffraction pattern
        background - noise background
        """
        signal = self.intensities(raw_data)
        noise = []
        for line in self.raw_lines.astype(np.int):
            rows, columns, val = line_aa(line[0, 1], line[0, 0], line[1, 1], line[1, 0])
            noise.append(np.abs(background[rows, columns] * val).sum())
        return signal / np.array(noise)

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

class ScanStreaks(FrameStreaks):
    """
    Detected diffraction streaks of a rotational scan class

    strks_list - detected lines FrameStreaks class object list
    """
    def __init__(self, raw_lines, exp_set, shapes):
        self.shapes = shapes
        super(ScanStreaks, self).__init__(raw_lines, exp_set)

    @classmethod
    def import_series(cls, strks_list):
        """
        Import ScanStreaks object from list of FrameStreaks
        """
        shapes = [0] + list(accumulate([strks.size for strks in strks_list], add))
        raw_lines = np.concatenate([strks.raw_lines for strks in strks_list])
        return cls(raw_lines, strks_list[0].exp_set, np.array(shapes))

    def __getitem__(self, index):
        starts = self.shapes[:-1][index]
        stops = self.shapes[1:][index]
        try:
            return FrameStreaks(self.raw_lines[starts:stops], self.exp_set)
        except (IndexError, TypeError):
            return ScanStreaks.import_series([FrameStreaks(self.raw_lines[start:stop], self.exp_set)
                                              for start, stop in zip(starts, stops)])

    def get_index(self, idxs):
        """
        Return elements frame indexes
        """
        frame_idxs = np.searchsorted(self.shapes, idxs, side='right') - 1
        uniq_idxs = np.unique(frame_idxs)
        return {idx: idxs[np.where(frame_idxs == idx)] - self.shapes[idx] for idx in uniq_idxs}

    def extract(self, idxs):
        """
        Return a ScanStreaks object of the given streaks subset
        """
        index = self.get_index(idxs)
        strks_list = []
        for frame_idx in index:
            frame = self.__getitem__(frame_idx)
            strks_list.append(frame[index[frame_idx]])
        return ScanStreaks.import_series(strks_list)

    def __iter__(self):
        for idx in range(self.shapes.size - 1):
            yield self.__getitem__(idx)

    def intensities(self, raw_data):
        """
        Return detected streaks intensities

        data - raw diffraction patterns
        """
        ints = []
        for frame_strks, frame in zip(iter(self), raw_data):
            frame_ints = frame_strks.intensities(frame)
            ints.append(frame_ints)
        return np.concatenate(ints)

    def snr(self, raw_data, background):
        """
        Return detected streaks signal to noise ratio

        raw_data - raw diffraction pattern
        background - noise backgrounds
        """
        snr_vals = []
        for frame_strks, frame, bgd in zip(iter(self), raw_data, background):
            frame_snr = frame_strks.snr(frame, bgd)
            snr_vals.append(frame_snr)
        return np.concatenate(snr_vals)

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

    def save(self, out_file):
        """
        Save detected diffraction streaks to an HDF5 file

        out_file - h5py file object
        """
        streaks_group = out_file.create_group('streaks')
        streaks_group.create_dataset('counts', data=self.shapes)
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
    