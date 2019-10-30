"""
data.py - data processing main classes
"""
import numpy as np
import numba as nb
import concurrent.futures
from . import utils
from math import sqrt, sin, cos, pi, atan2
from itertools import accumulate
from multiprocessing import cpu_count
from skimage.transform import probabilistic_hough_line
from skimage.draw import line_aa, circle
from cv2 import createLineSegmentDetector
from abc import ABCMeta, abstractmethod, abstractproperty

class ABCPropagator(object, metaclass=ABCMeta):
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
    def condition(self): pass

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
        return self.q_x, self.q_y, self.q_z, self.q_abs
    
    def source_pts(self):
        o_x = -np.sin(self.theta - np.arccos(self.q_abs / 2)) * np.cos(self.phi)
        o_y = -np.sin(self.theta - np.arccos(self.q_abs / 2)) * np.sin(self.phi)
        o_z = np.cos(self.theta - np.arccos(self.q_abs / 2))
        return o_x, o_y, o_z

    @abstractmethod
    def entry_pts(self): pass

    @abstractmethod
    def exit_pts(self): pass

    def out_wavevectors(self):
        onx, ony, onz = self.entry_pts()
        oxx, oxy, oxz = self.exit_pts()
        return (np.stack((self.q_x + onx, self.q_x + oxx), axis=1),
                np.stack((self.q_y + ony, self.q_y + oxy), axis=1),
                np.stack((self.q_z + onz, self.q_z + oxz), axis=1))

    def detector_pts(self, det_dist):
        k_x, k_y, k_z = self.out_wavevectors()
        det_x = det_dist * np.tan(np.sqrt(2 - 2 * k_z)) * np.cos(np.arctan2(k_y, k_x))
        det_y = det_dist * np.tan(np.sqrt(2 - 2 * k_z)) * np.sin(np.arctan2(k_y, k_x))
        return det_x, det_y

class CircPropagator(ABCPropagator):
    @property
    def condition(self):
        return np.abs(np.sin(self.betta - np.arccos(self.g_abs / 2))) < self.num_ap

    def entry_pts(self):
        dphi = np.arccos((self.q_abs**2 + 2 * self.q_z * sqrt(1 - self.num_ap**2)) / (2 * np.sqrt(self.q_x**2 + self.q_y**2) * self.num_ap))
        return (-self.num_ap * np.cos(self.phi + dphi),
                -self.num_ap * np.sin(self.phi + dphi),
                np.repeat(sqrt(1 - self.num_ap**2), self.q_x.shape))

    def exit_pts(self):
        dphi = np.arccos((self.q_abs**2 + 2 * self.q_z * sqrt(1 - self.num_ap**2)) / (2 * np.sqrt(self.q_x**2 + self.q_y**2) * self.num_ap))
        return (-self.num_ap * np.cos(self.phi - dphi),
                -self.num_ap * np.sin(self.phi - dphi),
                np.repeat(sqrt(1 - self.num_ap**2), self.q_x.shape))

class SquarePropagator(ABCPropagator):
    condition = None

    def __init__(self, rec_lat, num_ap):
        super(SquarePropagator, self).__init__(rec_lat, num_ap)
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
        return self._pts_x[:, 0], self._pts_y[:, 0], np.sqrt(1 - self._pts_x[:, 0]**2 - self._pts_y[:, 0]**2)

    def exit_pts(self):
        return self._pts_x[:, 1], self._pts_y[:, 1], np.sqrt(1 - self._pts_x[:, 1]**2 - self._pts_y[:, 1]**2)

class LineDetector(object, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _refiner(lines, angles, rs, taus, drtau, drn): pass

    @abstractmethod
    def _detector(self, frame): pass

    def det_frame_raw(self, frame):
        return np.array([[[x0, y0], [x1, y1]] for (x0, y0), (x1, y1) in self._detector(frame)])

    def det_frame(self, frame, zero, drtau, drn):
        lines = FrameStreaks(self.det_frame_raw(frame), zero)
        return FrameStreaks(type(self)._refiner(lines.lines,
                                                lines.angles,
                                                lines.radii,
                                                lines.taus,
                                                drtau,
                                                drn), zero)

    def det_scan_raw(self, data):
        return [self.det_frame_raw(frame) for frame in data]

    def det_scan(self, data, zero, drtau, drn):
        return ScanStreaks([self.det_frame(frame, zero, drtau, drn) for frame in data])

class HoughLineDetector(LineDetector):
    def __init__(self, threshold, line_length, line_gap, dth):
        self.trhd, self.ll, self.lg = threshold, line_length, line_gap
        self.thetas = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi / dth), endpoint=True)

    @staticmethod
    @nb.njit(nb.int64[:, :, :](nb.int64[:, :, :],
                               nb.float64[:],
                               nb.float64[:],
                               nb.float64[:, :],
                               nb.float64,
                               nb.float64))
    def _refiner(lines, angles, rs, taus, drtau, drn):
        newlines = np.empty(lines.shape, dtype=np.int64)
        idxs = []
        count = 0
        for idx in range(lines.shape[0]):
            if idx not in idxs:
                newline = np.empty((2, 2), dtype=np.float64)
                proj0 = lines[idx, 0, 0] * taus[idx, 0] + lines[idx, 0, 1] * taus[idx, 1]
                proj1 = lines[idx, 1, 0] * taus[idx, 0] + lines[idx, 1, 1] * taus[idx, 1]
                if proj0 < proj1: newline[0] = lines[idx, 0]; newline[1] = lines[idx, 1]
                else: newline[0] = lines[idx, 1]; newline[1] = lines[idx, 0]
                for idx2 in range(lines.shape[0]):
                    if idx == idx2: continue
                    elif abs((angles[idx] - angles[idx2]) * rs[idx]) < drtau and abs(rs[idx] - rs[idx2]) < drn:
                        idxs.append(idx2)
                        proj20 = lines[idx2, 0, 0] * taus[idx, 0] + lines[idx2, 0, 1] * taus[idx, 1]
                        proj21 = lines[idx2, 1, 0] * taus[idx, 0] + lines[idx2, 1, 1] * taus[idx, 1]
                        if proj20 < proj0: newline[0] = lines[idx2, 0]
                        elif proj20 > proj1: newline[1] = lines[idx2, 0]
                        if proj21 < proj0: newline[0] = lines[idx2, 1]
                        elif proj21 > proj1: newline[1] = lines[idx2, 1]           
                newlines[count] = newline
                count += 1
        return newlines[:count]

    def _detector(self, frame):
        return probabilistic_hough_line(frame,
                                        threshold=self.trhd,
                                        line_length=self.ll,
                                        line_gap=self.lg,
                                        theta=self.thetas)

class LineSegmentDetector(LineDetector):
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
    def _refiner(lines, angles, rs, taus, drtau, drn):
        lsdlines = np.empty(lines.shape, dtype=np.float64)
        idxs = []
        count = 0
        for idx in range(lines.shape[0]):
            if idx not in idxs:
                newline = np.empty((2, 2), dtype=np.float64)
                proj0 = lines[idx, 0, 0] * taus[idx, 0] + lines[idx, 0, 1] * taus[idx, 1]
                proj1 = lines[idx, 1, 0] * taus[idx, 0] + lines[idx, 1, 1] * taus[idx, 1]
                if proj0 < proj1: newline[0] = lines[idx, 0]; newline[1] = lines[idx, 1]
                else: newline[0] = lines[idx, 1]; newline[1] = lines[idx, 0]
                for idx2 in range(lines.shape[0]):
                    if idx == idx2: continue
                    elif abs((angles[idx] - angles[idx2]) * rs[idx]) < drtau and abs(rs[idx] - rs[idx2]) < drn:
                        idxs.append(idx2)
                        proj20 = lines[idx2, 0, 0] * taus[idx, 0] + lines[idx2, 0, 1] * taus[idx, 1]
                        proj21 = lines[idx2, 1, 0] * taus[idx, 0] + lines[idx2, 1, 1] * taus[idx, 1]
                        if proj20 < proj21:
                            newline[0] = (lines[idx2, 0] + newline[0]) / 2
                            newline[1] = (lines[idx2, 1] + newline[1]) / 2
                        else:
                            newline[0] = (lines[idx2, 1] + newline[0]) / 2
                            newline[1] = (lines[idx2, 0] + newline[1]) / 2
                lsdlines[count] = newline
                count += 1
        return lsdlines[:count]

    def _detector(self, frame):
        cap = np.mean(frame[frame != 0]) + np.std(frame[frame != 0])
        img = utils.arraytoimg(np.clip(frame, 0, cap))
        return self.detector.detect(img)[0][:, 0].reshape((-1, 2, 2)).astype(np.float64)

class FrameStreaks(object):
    def __init__(self, lines, zero):
        self.lines, self.zero = lines, zero
        self.dlines = lines - zero
        self.pts = self.dlines.mean(axis=1)

    @property
    def size(self):
        return self.lines.shape[0]

    @property
    def xs(self):
        return self.pts[:, 0]

    @property
    def ys(self):
        return self.pts[:, 1]

    @property
    def radii(self):
        return np.sqrt(self.xs**2 + self.ys**2)

    @property
    def angles(self):
        return np.arctan2(self.ys, self.xs)

    @property
    def taus(self):
        taus = (self.lines[:, 1] - self.lines[:, 0])
        return taus / np.sqrt(taus[:, 0]**2 + taus[:, 1]**2)[:, np.newaxis]

    def __iter__(self):
        for line in self.lines: yield line.astype(np.int64)

    def index_pts(self):
        ts = self.dlines[:, 0, 1] * self.taus[:, 0] - self.dlines[:, 0, 0] * self.taus[:, 1]
        return -self.taus[:, 1] * ts + self.zero[0], self.taus[:, 0] * ts + self.zero[1]

    def intensities(self, frame):
        ints = []
        for line in iter(self):
            rows, columns, val = line_aa(line[0, 1], line[0, 0], line[1, 1], line[1, 0])
            ints.append((frame[rows, columns] * val).sum())
        return np.array(ints)

    def snr(self, frame, background):
        signal = self.intensities(frame)
        noise = []
        for line in iter(self):
            radius = np.sqrt(np.sum((line[0] - line[1])**2)) / 2
            center = line.mean(axis=0)
            rows, columns = circle(center[0], center[1], radius, shape=frame.shape)
            noise.append(background[rows, columns].sum())
        return signal / np.array(noise)

class ScanStreaks(object):
    def __init__(self, streakslist):
        self.strkslist = streakslist

    @property
    def shapes(self):
        return np.array(list(accumulate([strks.size for strks in self.strkslist], lambda x, y: x + y)))

    @property
    def zero(self):
        return self.__getitem__(0).zero

    @staticmethod
    @nb.njit(nb.float64[:, :](nb.float64[:, :], nb.int64[:], nb.float64))
    def _refiner(qs, shapes, dk):
        b = len(shapes)
        out = np.empty(qs.shape, dtype=np.float64)
        idxs = []; jj = 0; count = 0
        for i in range(shapes[b - 2]):
            if i == shapes[jj]: jj += 1
            if i in idxs: continue
            qslist = []
            for j in range(shapes[jj], shapes[jj + 1]):
                if sqrt((qs[i, 0] - qs[j, 0])**2 + (qs[i, 1] - qs[j, 1])**2 + (qs[i, 2] - qs[j, 2])**2) < dk:
                    qslist.append(qs[i]); idxs.append(i)
                    break
            else:
                out[count] = qs[i]; count += 1
                continue
            for k in range(jj, b - 1):
                skip = True; q = qslist[-1]
                for l in range(shapes[k], shapes[k + 1]):
                    if sqrt((q[0] - qs[l, 0])**2 + (q[1] - qs[l, 1])**2 + (q[2] - qs[l, 2])**2) < dk:
                        skip = False; qslist.append(qs[l]); idxs.append(l)
                if skip: break
            qsum = np.copy(qslist[0])
            for q in qslist[1:]:
                qsum += q
            out[count] = qsum / len(qslist); count += 1
        return out[:count]

    def __getitem__(self, index): return self.strkslist[index]

    def __iter__(self):
        for strks in self.strkslist: yield strks

    def rec_vectors(self, axis, thetas, pix_size, det_dist):
        qs_list = []
        for strks, theta in zip(iter(self), thetas):
            k_x = np.arctan(pix_size * strks.radii / det_dist) * np.cos(strks.angles)
            k_y = np.arctan(pix_size * strks.radii / det_dist) * np.sin(strks.angles)
            rotm = utils.rotation_matrix(axis, theta)
            q_x, q_y, q_z = utils.rotate(rotm, k_x, k_y, np.sqrt(1 - k_x**2 - k_y**2) - 1)
            qs_list.append(np.stack((q_x, q_y, q_z), axis=1))
        return ReciprocalPeaks(np.concatenate(qs_list))

    def index_vectors(self, axis, thetas, pix_size, det_dist):
        qs_list = []
        for strks, theta in zip(iter(self), thetas):
            xs, ys = strks.index_pts()
            radii = np.sqrt(xs**2 + ys**2)
            angles = np.arctan2(ys, xs)
            k_x = np.arctan(pix_size * radii / det_dist) * np.cos(angles)
            k_y = np.arctan(pix_size * radii / det_dist) * np.sin(angles)
            rotm = utils.rotation_matrix(axis, theta)
            q_x, q_y, q_z = utils.rotate(rotm, k_x, k_y, np.sqrt(1 - k_x**2 - k_y**2) - 1)
            qs_list.append(np.stack((q_x, q_y, q_z), axis=1))
        return ReciprocalPeaks(np.concatenate(qs_list))

    def refined_rec_vectors(self, axis, thetas, pix_size, det_dist, dk):
        _qs = self.rec_vectors(axis, thetas, pix_size, det_dist).qs
        return ReciprocalPeaks(ScanStreaks._refiner(_qs, self.shapes, dk))

    def refined_index_vectors(self, axis, thetas, pix_size, det_dist, dk):
        _qs = self.index_vectors(axis, thetas, pix_size, det_dist).qs
        return ReciprocalPeaks(ScanStreaks._refiner(_qs, self.shapes, dk))

    def save(self, data, outfile):
        lines_group = outfile.create_group('bragg_lines')
        ints_group = outfile.create_group('bragg_intensities')
        for idx, (streaks, frame) in enumerate(zip(iter(self), data)):
            lines_group.create_dataset(str(idx), data=streaks.lines)
            ints_group.create_dataset(str(idx), data=streaks.intensities(frame))

class ReciprocalPeaks(object):
    def __init__(self, qs):
        self.qs = qs

    @staticmethod
    @nb.njit(nb.uint64[:, :, :](nb.float64[:,:], nb.float64, nb.int64), parallel=True)
    def _corgrid_func(qs, q_max, size):
        a = qs.shape[0]
        corgrid = np.zeros((size, size, size), dtype=np.uint64)
        ks = np.linspace(-q_max, q_max, size)
        for i in nb.prange(a):
            for j in range(i + 1, a):
                dk = qs[i] - qs[j]
                if abs(dk[0]) < q_max and abs(dk[1]) < q_max and abs(dk[2]) < q_max:
                    ii = np.searchsorted(ks, dk[0])
                    jj = np.searchsorted(ks, dk[1])
                    kk = np.searchsorted(ks, dk[2])
                    corgrid[ii, jj, kk] += 1
        return corgrid

    @property
    def range(self):
        return self.qs.max(axis=0) - self.qs.min(axis=0)    

    @staticmethod
    @nb.njit(nb.float64[:, :](nb.float64[:,:], nb.float64), parallel=True)
    def _cor_func(qs, q_max):
        a = qs.shape[0]
        cor = np.empty((int(a * (a - 1) / 2), 3), dtype=np.float64)
        count = 0
        for i in nb.prange(a):
            for j in range(i + 1, a):
                dk = qs[i] - qs[j]
                if abs(dk[0]) < q_max and abs(dk[1]) < q_max and abs(dk[2]) < q_max:
                    cor[count] = dk
                    count += 1
        return cor[:count]

    @staticmethod
    @nb.njit(nb.int64[:, :, :](nb.float64[:, :], nb.int64), parallel=True)
    def _grid(qs, size):
        a = qs.shape[0]
        grid = np.zeros((size, size, size), dtype=np.int64)
        xs = np.linspace(qs[:, 0].min(), qs[:, 0].max(), size)
        ys = np.linspace(qs[:, 1].min(), qs[:, 1].max(), size)
        zs = np.linspace(qs[:, 2].min(), qs[:, 2].max(), size)
        xs[0] -= (xs[-1] - xs[0]) / 10; xs[-1] += (xs[-1] - xs[0]) / 10
        ys[0] -= (ys[-1] - ys[0]) / 10; ys[-1] += (ys[-1] - ys[0]) / 10
        zs[0] -= (zs[-1] - zs[0]) / 10; zs[-1] += (zs[-1] - zs[0]) / 10
        for i in nb.prange(a):
            ii = np.searchsorted(xs, qs[i, 0])
            jj = np.searchsorted(ys, qs[i, 1])
            kk = np.searchsorted(zs, qs[i, 2])
            grid[ii, jj, kk] += 1
        return grid

    def correlation_grid(self, q_max, size):
        return ReciprocalPeaks._corgrid_func(self.qs, q_max, size)

    def correlation(self, q_max):
        return ReciprocalPeaks._cor_func(self.qs, q_max)

    def grid(self, size):
        return ReciprocalPeaks._grid(self.qs, size)

    def save_pts(self, out_file):
        out_file.create_dataset('rec_vectors', data=self.qs)

@nb.njit(nb.float64[:, :](nb.float64[:, :]))
def NMS(image):
    a, b = image.shape
    res = np.zeros((a, b), dtype=np.float64)
    for i in range(1, a - 1):
        for j in range(1, b - 1):
            phase = atan2(image[i + 1, j] - image[i - 1, j], image[i, j+1] - image[i, j - 1])
            if (phase >= 0.875 * pi or phase < -0.875 * pi) or (phase >= -0.125 * pi and phase < 0.125 * pi):
                if image[i, j] >= image[i, j + 1] and image[i, j] >= image[i, j - 1]:
                    res[i, j] = image[i, j]
            if (phase >= 0.625 * pi and phase < 0.875 * pi) or (phase >= -0.375 * pi and phase < -0.125 * pi):
                if image[i, j] >= image[i - 1, j + 1] and image[i, j] >= image[i + 1, j - 1]:
                    res[i, j] = image[i, j]
            if (phase >= 0.375 * pi and phase < 0.625 * pi) or (phase >= -0.625 * pi and phase < -0.375 * pi):
                if image[i, j] >= image[i - 1, j] and image[i, j] >= image[i + 1, j]:
                    res[i, j] = image[i, j]
            if (phase >= 0.125 * pi and phase < 0.375 * pi) or (phase >= -0.875 * pi and phase < -0.625 * pi):
                if image[i, j] >= image[i - 1, j - 1] and image[i, j] >= image[i + 1, j + 1]:
                    res[i, j] = image[i, j]
    return res
