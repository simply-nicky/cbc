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

class LineDetector(metaclass=ABCMeta):
    """
    Abstract line detector class
    """
    @staticmethod
    @abstractmethod
    def _refiner(lines, angles, radii, taus, drtau, drn):
        pass

    @abstractmethod
    def _detector(self, frame):
        pass

    def det_frame_raw(self, frame):
        return np.array([[[x0, y0], [x1, y1]] for (x0, y0), (x1, y1) in self._detector(frame)])

    def det_frame(self, frame, zero, drtau, drn):
        """
        Return FrameStreaks class object of detected lines

        frame - diffraction pattern
        zero - zero output wavevector point
        drtau - tangent detection error
        drn - radial detection error
        """
        frame_strks = FrameStreaks(self.det_frame_raw(frame), zero)
        pts = frame_strks.dlines.mean(axis=1)
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        radii = np.sqrt(np.sum(pts**2, axis=1))
        ref_lines = type(self)._refiner(lines=frame_strks.lines, angles=angles,
                                        radii=radii, taus=frame_strks.taus,
                                        drtau=drtau, drn=drn)
        return FrameStreaks(ref_lines, zero)

    def det_scan_raw(self, scan):
        """
        Return Hough Line Transofrm raw detected lines for given scan
        """
        return [self.det_frame_raw(frame) for frame in scan]

    def det_scan(self, data, zero, drtau, drn):
        """
        Return ScanStreaks class obect of detected lines

        scan - rotational scan
        zero - zero output wavevector point
        drtau - tangent detection error
        drn - radial detection error
        """
        return ScanStreaks([self.det_frame(frame, zero, drtau, drn) for frame in data])

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
    @nb.njit(nb.int64[:, :, :](nb.int64[:, :, :],
                               nb.float64[:],
                               nb.float64[:],
                               nb.float64[:, :],
                               nb.float64,
                               nb.float64))
    def _refiner(lines, angles, radii, taus, drtau, drn):
        new_lines = np.empty(lines.shape, dtype=np.int64)
        idxs = []
        count = 0
        for idx in range(lines.shape[0]):
            if idx not in idxs:
                new_line = np.empty((2, 2), dtype=np.int64)
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
                new_lines[count] = new_line
                count += 1
        return new_lines[:count]

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
                                 nb.float64[:],
                                 nb.float64[:],
                                 nb.float64[:, :],
                                 nb.float64,
                                 nb.float64))
    def _refiner(lines, angles, radii, taus, drtau, drn):
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

class ExperimentSettings():
    """
    Experimment parameters class

    axis - axis of rotation
    pix_size - detector pixel size
    det_dist - distance between the detector and the sample
    """
    def __init__(self, axis, pix_size, det_dist, wavelength):
        self.axis, self.pix_size, self.det_dist = axis, pix_size, det_dist
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
        return pixels * self.pix_size / self.det_dist

    def convert_rec(self, rec_vec):
        """
        Convert dimensionless reciprocal vectors to [m^-1]
        """
        return rec_vec / self.wavelength

    def rec_project(self, det_pts, zero):
        """
        Project detected diffraction streaks to reciprocal space vectors in [a.u.]

        det_pts - detector points: [fs, ss]
        zero - zero output wavevector
        """
        pts = det_pts - zero
        angles = np.arctan2(pts[..., 1], pts[..., 0])
        thetas = np.arctan(self.pix_size * np.sqrt(pts[..., 0]**2 + pts[..., 1]**2) / self.det_dist)
        q_x = np.sin(thetas) * np.cos(angles)
        q_y = np.sin(thetas) * np.sin(angles)
        return np.stack((q_x, q_y, np.sqrt(1 - q_x**2 - q_y**2)), axis=-1)

    def save(self, out_file):
        """
        Save experiment settings to an HDF5 file
        """
        out_group = out_file.create_group("experiment_settings")
        out_group.create_dataset('pix_size', self.pix_size)
        out_group.create_dataset('det_dist', self.det_dist)
        out_group.create_dataset('rot_axis', self.axis)
        out_group.create_dataset('wavelength', self.wavelength)

class FrameStreaks():
    """
    Detected diffraction streaks on one frame class

    lines - detected lines
    zero - zero output wavevector point
    """
    def __init__(self, lines, zero):
        self.lines, self.zero = lines, zero
        self.dlines = lines - zero

    @property
    def kin(self):
        return np.tile(np.array([0, 0, 1]), (self.size, 1))

    @property
    def size(self):
        return self.lines.shape[0]

    @property
    def x_coord(self):
        return self.dlines[..., 0]

    @property
    def y_coord(self):
        return self.dlines[..., 1]

    @property
    def radii(self):
        return np.sqrt(self.x_coord**2 + self.y_coord**2)

    @property
    def angles(self):
        return np.arctan2(self.y_coord, self.x_coord)

    @property
    def taus(self):
        taus = (self.lines[:, 1] - self.lines[:, 0])
        return taus / np.sqrt(taus[:, 0]**2 + taus[:, 1]**2)[:, np.newaxis]

    def __iter__(self):
        for line in self.lines:
            yield line.astype(np.int64)

    def index_pts(self):
        """
        Return indexing points
        """
        products = self.dlines[:, 0, 1] * self.taus[:, 0] - self.dlines[:, 0, 0] * self.taus[:, 1]
        return -self.taus[:, 1] * products + self.zero[0], self.taus[:, 0] * products + self.zero[1]

    def intensities(self, raw_data):
        """
        Return detected streaks intensities

        raw_data - raw diffraction pattern
        """
        ints = []
        for line in iter(self):
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
        for line in iter(self):
            rows, columns, val = line_aa(line[0, 1], line[0, 0], line[1, 1], line[1, 0])
            noise.append(np.abs(background[rows, columns] * val).sum())
        return signal / np.array(noise)

    def kout(self, exp_set, theta):
        """
        Return reciprocal vectors of detected diffraction streaks.

        exp_set - ExperimentSettings class object
        theta - angle of rotation
        """
        rot_m = exp_set.rotation_matrix(theta)
        kout = exp_set.rec_project(det_pts=self.lines.mean(axis=1), zero=self.zero)
        return RecVectors(kout=kout.dot(rot_m.T), kin=self.kin.dot(rot_m.T))

    def kout_streaks(self, exp_set, theta):
        """
        Return reciprocal vectors of detected diffraction streaks.

        exp_set - ExperimentSettings class object
        theta - angle of rotation
        """
        rot_m = exp_set.rotation_matrix(theta)
        kout = exp_set.rec_project(det_pts=self.lines, zero=self.zero)
        kin = kout - kout.mean(axis=1)[:, None] + self.kin[:, None]
        return RecVectors(kout=kout.dot(rot_m.T), kin=kin.dot(rot_m.T))

    def kout_index(self, exp_set, theta):
        """
        Return reciprocal points of indexing points.

        exp_set - ExperimentSettings class object
        theta - angle of rotation
        """
        index_pts = self.index_pts()
        rot_m = exp_set.rotation_matrix(theta)
        kout = exp_set.rec_project(det_pts=index_pts, zero=self.zero)
        return RecVectors(kout=kout.dot(rot_m.T), kin=self.kin.dot(rot_m.T))

class ScanStreaks(FrameStreaks):
    """
    Detected diffraction streaks of a rotational scan class

    strks_list - detected lines FrameStreaks class object list
    """
    def __init__(self, strks_list):
        shapes = [0] + list(accumulate([strks.size for strks in strks_list], add))
        self.shapes = np.array(shapes)
        lines = np.concatenate([strks.lines for strks in strks_list])
        super(ScanStreaks, self).__init__(lines, strks_list[0].zero)

    def __getitem__(self, index):
        starts = self.shapes[:-1][index]
        stops = self.shapes[1:][index]
        if isinstance(index, int):
            return FrameStreaks(self.lines[starts:stops], self.zero)
        else:
            return ScanStreaks([FrameStreaks(self.lines[start:stop], self.zero)
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

    def kout(self, exp_set, theta):
        """
        Return reciprocal vectors of detected diffraction streaks in rotational scan.

        exp_set - ExperimentSettings class object
        theta - angles of rotation
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout(exp_set=exp_set, theta=theta_val)
            kout_list.append(rec_vec.kout)
            kin_list.append(rec_vec.kin)
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def kout_ref(self, exp_set, theta, pixels):
        """
        Return refined reciprocal vectors of detected diffraction streaks in rotational scan.

        exp_set - ExperimentSettings class object
        thetas - angles of rotation
        pixels - pixel distance between two adjacent streaks considered to be collapsed
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout(exp_set=exp_set, theta=theta_val)
            kout_list.append(rec_vec.kout)
            kin_list.append(rec_vec.kin)
        groups = TiltGroups(kout=kout_list,
                            kin=kin_list,
                            threshold=exp_set.pixtoq(pixels))
        ref_kin = groups.ref_kin()
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(ref_kin))

    def kout_streaks(self, exp_set, theta):
        """
        Return reciprocal streaks of detected diffraction streaks in rotational scan.

        exp_set - ExperimentSettings class object
        thetas - angles of rotation
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout_streaks(exp_set=exp_set, theta=theta_val)
            kout_list.append(rec_vec.kout)
            kin_list.append(rec_vec.kin)
        return RecVectors(kout=np.concatenate(kout_list),
                          kin=np.concatenate(kin_list))

    def kout_index(self, exp_set, theta):
        """
        Return reciprocal vectors of indexing points in rotational scan.

        exp_set - ExperimentSettings class object
        thetas - angles of rotation
        """
        kout_list, kin_list = [], []
        for frame_strks, theta_val in zip(iter(self), theta):
            rec_vec = frame_strks.kout_index(exp_set=exp_set, theta=theta_val)
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
        out_group.create_dataset('zero', data=self.zero)
        out_group.create_dataset('intensities', data=self.intensities(raw_data))
        out_group.create_dataset('snr', data=self.snr(raw_data, background))

class RecVectors():
    """
    Diffraction peaks in reciprocal space class

    kout - outcoming wavevectors
    kin - incoming wavevectors
    """
    def __init__(self, kout, kin):
        self.kout, self.kin, self.scat_vec = kout, kin, kout - kin
        self.center = np.zeros(3)

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
        return utils.make_grid(self.scat_vec, size)

    def fft(self, size):
        """
        Return fast fourier transform of reciprocal vectors

        size - output grid size
        """
        grid = self.grid(size)
        return np.abs(np.fft.fftshift(np.fft.fftn(grid))**2)

    def bin(self, a_rec, b_rec, c_rec):
        """
        Bin reciprocal vectors in HKL indices
        """
        h_idx = (self.scat_vec - self.center).dot(a_rec) / a_rec.dot(a_rec)
        k_idx = (self.scat_vec - self.center).dot(b_rec) / b_rec.dot(b_rec)
        l_idx = (self.scat_vec - self.center).dot(c_rec) / c_rec.dot(c_rec)
        return np.stack((h_idx, k_idx, l_idx))

    def index(self, size=100, gauss_sigma=4, threshold=3):
        """
        Find indexing solution

        size - fourier transform grid size
        gauss_sigma - gaussian filter sigma value
        threshold - peak detetion threshold
        """
        fft_grid = self.fft(size)
        fft_blur = gaussian_filter(fft_grid.astype(np.float64), gauss_sigma)
        peak_mask = fft_blur > fft_blur.mean() + threshold * fft_blur.std()
        peak_labels, peak_num = label(peak_mask)
        peak_centers = np.array(maximum_position(fft_blur,
                                                 labels=peak_labels,
                                                 index=np.arange(1, peak_num + 1)))
        peak_centers = peak_centers - size / 2 * np.ones(3)
        axes_mask = np.array([center.dot(center) for center in peak_centers]).argsort()[:7]
        axes = (peak_centers * self.range**-1)[axes_mask]
        return axes

    # def save(self, out_file):
    #     """
    #     Save reciprocal points to an HDF5 file

    #     out_file - h5py file object
    #     """
    #     self.exp_set.save(out_file)
    #     out_file.create_dataset('rec_vectors', data=self.scat_vec)
