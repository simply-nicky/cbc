"""
feat_detect.py - feature detection on convergent diffraction pattern module
"""
from itertools import accumulate
from operator import add
from abc import ABCMeta, abstractmethod
import numpy as np
import numba as nb
from skimage.transform import probabilistic_hough_line
from skimage.draw import line_aa
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, maximum_position
from cv2 import createLineSegmentDetector
from . import utils
from .grouper import TiltGroups

class ExperimentSettings():
    """
    Experimment parameters class

    axis - axis of rotation
    pix_size - detector pixel size
    det_pos - detector position in respect to the sample [mm]
    """
    def __init__(self, axis, pix_size, det_pos, wavelength):
        self.axis, self.pix_size, self.det_pos = axis, pix_size, det_pos
        self.wavelength = wavelength

    def rotation_matrix(self, theta):
        """
        Return roational matrix for a given angle of rotation
        """
        return utils.rotation_matrix(self.axis, theta)

    def pixtoq(self, pixels):
        """
        Convert detector distance in pixels to distance in reciprocal space
        """
        return pixels * self.pix_size / self.det_pos[2]

    def laue_raw(self, det_pixels):
        """
        Project detected diffraction streaks to reciprocal space vectors in [a.u.]

        det_pixels - detector points: [fs, ss]
        """
        pts = det_pixels * self.pix_size - self.det_pos[:2]
        angles = np.arctan2(pts[..., 1], pts[..., 0])
        thetas = np.arctan(np.sqrt(pts[..., 0]**2 + pts[..., 1]**2) / self.det_pos[2])
        return np.stack((np.sin(thetas) * np.cos(angles),
                         np.sin(thetas) * np.sin(angles),
                         np.cos(thetas)), axis=-1)

    def laue(self, det_pts):
        """
        Project detected diffraction streaks to reciprocal space vectors in [a.u.]

        det_pts - detector points [mm]
        """
        angles = np.arctan2(det_pts[..., 1], det_pts[..., 0])
        thetas = np.arctan(np.sqrt(det_pts[..., 0]**2 + det_pts[..., 1]**2) / self.det_pos[2])
        return np.stack((np.sin(thetas) * np.cos(angles),
                         np.sin(thetas) * np.sin(angles),
                         np.cos(thetas)), axis=-1)

    def save(self, out_file):
        """
        Save experiment settings to an HDF5 file

        out_file - h5py File object
        """
        out_group = out_file.create_group("experiment_settings")
        out_group.create_dataset('pix_size', self.pix_size)
        out_group.create_dataset('det_dist', self.det_pos)
        out_group.create_dataset('rot_axis', self.axis)
        out_group.create_dataset('wavelength', self.wavelength)

class LineDetector(metaclass=ABCMeta):
    """
    Abstract line detector class
    """
    @staticmethod
    @abstractmethod
    def _refiner(lines, taus, drtau, drn):
        pass

    @abstractmethod
    def _detector(self, frame):
        pass

    def det_frame_raw(self, frame):
        return np.array([[[x0, y0], [x1, y1]] for (x0, y0), (x1, y1) in self._detector(frame)])

    def det_frame(self, frame, exp_set, drtau=1.5, drn=0.75):
        """
        Return FrameStreaks class object of detected lines

        frame - diffraction pattern
        exp_set - ExperimentalSettings class object
        drtau - tangent detection error
        drn - radial detection error
        """
        frame_strks = FrameStreaks(self.det_frame_raw(frame), exp_set)
        ref_lines = type(self)._refiner(lines=frame_strks.lines,
                                        taus=frame_strks.taus,
                                        drtau=drtau, drn=drn)
        return FrameStreaks((ref_lines + exp_set.det_pos[:2]) / exp_set.pix_size, exp_set)

    def det_scan_raw(self, scan):
        """
        Return Hough Line Transofrm raw detected lines for given scan
        """
        return [self.det_frame_raw(frame) for frame in scan]

    def det_scan(self, data, exp_set, drtau=1.5, drn=0.75):
        """
        Return ScanStreaks class obect of detected lines

        scan - rotational scan
        exp_set - ExperimentalSettings class object
        drtau - tangent detection error
        drn - radial detection error
        """
        return ScanStreaks([self.det_frame(frame, exp_set, drtau, drn) for frame in data])

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

    @staticmethod
    @nb.njit(nb.float64[:, :, :](nb.float64[:, :, :],
                                 nb.float64[:, :],
                                 nb.float64,
                                 nb.float64))
    def _refiner(lines, taus, drtau, drn):
        hl_lines = np.empty(lines.shape, dtype=np.float64)
        pts = (lines[:, 0] + lines[:, 1]) / 2
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        idxs = []
        count = 0
        for idx in range(lines.shape[0]):
            if idx not in idxs:
                new_line = np.empty((2, 2), dtype=np.float64)
                proj0 = lines[idx, 0, 0] * taus[idx, 0] + lines[idx, 0, 1] * taus[idx, 1]
                proj1 = lines[idx, 1, 0] * taus[idx, 0] + lines[idx, 1, 1] * taus[idx, 1]
                if proj0 < proj1:
                    new_line = lines[idx]
                else:
                    new_line = lines[idx, ::-1]
                for idx2 in range(lines.shape[0]):
                    if idx == idx2:
                        continue
                    elif abs((angles[idx] - angles[idx2]) * radii[idx]) < drtau and abs(radii[idx] - radii[idx2]) < drn:
                        idxs.append(idx2)
                        proj20 = lines[idx2, 0, 0] * taus[idx, 0] + lines[idx2, 0, 1] * taus[idx, 1]
                        proj21 = lines[idx2, 1, 0] * taus[idx, 0] + lines[idx2, 1, 1] * taus[idx, 1]
                        if proj20 < proj0:
                            new_line[0] = lines[idx2, 0]
                        elif proj20 > proj1:
                            new_line[1] = lines[idx2, 0]
                        if proj21 < proj0:
                            new_line[0] = lines[idx2, 1]
                        elif proj21 > proj1:
                            new_line[1] = lines[idx2, 1]
                hl_lines[count] = new_line
                count += 1
        return hl_lines[:count]

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

    @staticmethod
    @nb.njit(nb.float64[:, :, :](nb.float64[:, :, :],
                                 nb.float64[:, :],
                                 nb.float64,
                                 nb.float64))
    def _refiner(lines, taus, drtau, drn):
        pts = (lines[:, 0] + lines[:, 1]) / 2
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        lsd_lines = np.empty(lines.shape, dtype=np.float64)
        idxs = []
        count = 0
        for idx in range(lines.shape[0]):
            if idx not in idxs:
                new_line = np.empty((2, 2), dtype=np.float64)
                proj0 = lines[idx, 0, 0] * taus[idx, 0] + lines[idx, 0, 1] * taus[idx, 1]
                proj1 = lines[idx, 1, 0] * taus[idx, 0] + lines[idx, 1, 1] * taus[idx, 1]
                if proj0 < proj1:
                    new_line = lines[idx]
                else:
                    new_line[0] = lines[idx, ::-1]
                for idx2 in range(lines.shape[0]):
                    if idx == idx2:
                        continue
                    elif abs((angles[idx] - angles[idx2]) * radii[idx]) < drtau and abs(radii[idx] - radii[idx2]) < drn:
                        idxs.append(idx2)
                        proj20 = lines[idx2, 0, 0] * taus[idx, 0] + lines[idx2, 0, 1] * taus[idx, 1]
                        proj21 = lines[idx2, 1, 0] * taus[idx, 0] + lines[idx2, 1, 1] * taus[idx, 1]
                        if proj20 < proj21:
                            new_line[0] = (lines[idx2, 0] + new_line[0]) / 2
                            new_line[1] = (lines[idx2, 1] + new_line[1]) / 2
                        else:
                            new_line[0] = (lines[idx2, 1] + new_line[0]) / 2
                            new_line[1] = (lines[idx2, 0] + new_line[1]) / 2
                lsd_lines[count] = new_line
                count += 1
        return lsd_lines[:count]

    def _detector(self, frame):
        cap = np.mean(frame[frame != 0]) + np.std(frame[frame != 0])
        img = utils.arraytoimg(np.clip(frame, 0, cap))
        return self.detector.detect(img)[0][:, 0].reshape((-1, 2, 2)).astype(np.float64)

class FrameStreaks():
    """
    Detected diffraction streaks on one frame class

    lines - detected lines
    zero - zero output wavevector point
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

        exp_set - ExperimentSettings class object
        theta - angle of rotation
        """
        rot_m = self.exp_set.rotation_matrix(theta)
        kout = self.exp_set.laue(self.lines.mean(axis=1))
        return RecVectors(kout=kout.dot(rot_m.T),
                          kin=self.kin.dot(rot_m.T))

    def kout_streaks(self, theta):
        """
        Return reciprocal vectors of detected diffraction streaks.

        exp_set - ExperimentSettings class object
        theta - angle of rotation
        """
        rot_m = self.exp_set.rotation_matrix(theta)
        kout = self.exp_set.laue(self.lines)
        return RecVectors(kout=kout.dot(rot_m.T),
                          kin=np.tile(self.kin[:, None], (1, 2, 1)))

    def kout_index(self, theta):
        """
        Return reciprocal points of indexing points.

        exp_set - ExperimentSettings class object
        theta - angle of rotation
        """
        index_pts = self.index_pts()
        rot_m = self.exp_set.rotation_matrix(theta)
        kout = self.exp_set.laue(index_pts)
        return RecVectors(kout=kout.dot(rot_m.T), kin=self.kin.dot(rot_m.T))

class ScanStreaks(FrameStreaks):
    """
    Detected diffraction streaks of a rotational scan class

    strks_list - detected lines FrameStreaks class object list
    """
    def __init__(self, strks_list):
        shapes = [0] + list(accumulate([strks.size for strks in strks_list], add))
        self.shapes = np.array(shapes)
        raw_lines = np.concatenate([strks.raw_lines for strks in strks_list])
        super(ScanStreaks, self).__init__(raw_lines, strks_list[0].exp_set)

    def __getitem__(self, index):
        starts = self.shapes[:-1][index]
        stops = self.shapes[1:][index]
        try:
            return FrameStreaks(self.raw_lines[starts:stops], self.exp_set)
        except (IndexError, TypeError):
            return ScanStreaks([FrameStreaks(self.raw_lines[start:stop], self.exp_set)
                                for start, stop in zip(starts, stops)])

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

        exp_set - ExperimentSettings class object
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

        exp_set - ExperimentSettings class object
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

        exp_set - ExperimentSettings class object
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

        exp_set - ExperimentSettings class object
        thetas - angles of rotation
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout_index(theta=theta_val)
            kout_list.append(rec_vec.kout)
            kin_list.append(rec_vec.kin)
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def save(self, raw_data, background, out_file):
        """
        Save detected diffraction streaks to an HDF5 file

        raw_data - raw diffraction patterns
        backgroun - background noises
        out_file - h5py file object
        """
        out_group = out_file.create_group('streaks')
        out_group.create_dataset('counts', data=self.shapes)
        out_group.create_dataset('lines', data=self.lines)
        out_group.create_dataset('intensities', data=self.intensities(raw_data))
        out_group.create_dataset('snr', data=self.snr(raw_data, background))
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
    