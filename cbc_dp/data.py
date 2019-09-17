import numpy as np, numba as nb, concurrent.futures
from . import utils
from math import sqrt, sin, cos, pi, atan2
from itertools import accumulate
from multiprocessing import cpu_count
from skimage.transform import probabilistic_hough_line
from skimage.draw import line_aa
from cv2 import createLineSegmentDetector
from abc import ABCMeta, abstractmethod

class RecLattice(object):
    """
    Reciprocal lattice class

    arec, brec, crec - basis vectors of the reciprocal lattice [mm^-1]
    wavelength - light carrier  wavelength [mm]
    qmax - maximum lattice vector in dimensionless units
    """
    def __init__(self, arec, brec, crec, qmax, wavelength=1.14e-7):
        self.a, self.b, self.c, self.qmax = arec * wavelength, brec * wavelength, crec * wavelength, qmax

    def vectors(self):
        Na, Nb, Nc = self.qmax // np.sqrt(self.a.dot(self.a)), self.qmax // np.sqrt(self.b.dot(self.b)), self.qmax // np.sqrt(self.c.dot(self.c))
        arng, brng, crng = np.arange(-Na, Na), np.arange(-Nb, Nb), np.arange(-Nc, Nc)
        na, nb, nc = np.meshgrid(arng, brng, crng)
        pts = np.multiply.outer(self.a, na) + np.multiply.outer(self.b, nb) + np.multiply.outer(self.c, nc)
        mask = (np.sqrt(pts[0]**2 + pts[1]**2 + pts[2]**2) < self.qmax) & (np.sqrt(pts[0]**2 + pts[1]**2 + pts[2]**2) != 0)
        return pts[0][mask].ravel(), pts[1][mask].ravel(), pts[2][mask].ravel(), np.sqrt(pts[0]**2 + pts[1]**2 + pts[2]**2)[mask].ravel()

class ConvLines(object):
    def __init__(self, reclat, NA):
        if reclat.qmax > 2: raise ValueError('qmax must be less than 2')
        self.reclat, self.NA = reclat, NA
        self.gx, self.gy, self.gz, self.gabs = reclat.vectors()

    @property
    def betta(self): return np.arccos(-self.gz / self.gabs)

    @property
    def condition(self): return (2 * np.cos(self.betta + self.NA) < self.gabs) & (2 * np.cos(self.betta - self.NA) > self.gabs)

    @property
    def qx(self): return self.gx[self.condition]
    
    @property
    def qy(self): return self.gy[self.condition]

    @property
    def qz(self): return self.gz[self.condition]

    @property
    def qabs(self): return self.gabs[self.condition]

    @property
    def theta(self): return np.arccos(-self.qx / self.qabs)

    @property
    def phi(self): return np.arctan2(self.qy, self.qx)

    def lauevectors(self):
        return self.qx, self.qy, self.qz, self.qabs
    
    def sourcepts(self):
        ox = -np.sin(self.theta - np.arccos(self.qabs / 2)) * np.cos(self.phi)
        oy = -np.sin(self.theta - np.arccos(self.qabs / 2)) * np.sin(self.phi)
        oz = np.cos(self.phi)
        return ox, oy, oz

    def entrypts(self):
        dphi = np.arccos((self.qabs**2 + 2 * self.qz * cos(self.NA)) / (2 * np.sqrt(self.qx**2 + self.qy**2) * sin(self.NA)))
        return -sin(self.NA) * np.cos(self.phi + dphi), -sin(self.NA) * np.sin(self.phi + dphi), np.repeat(cos(self.NA), self.qx.shape)

    def exitpts(self):
        dphi = np.arccos((self.qabs**2 + 2 * self.qz * cos(self.NA)) / (2 * np.sqrt(self.qx**2 + self.qy**2) * sin(self.NA)))
        return -sin(self.NA) * np.cos(self.phi - dphi), -sin(self.NA) * np.sin(self.phi - dphi), np.repeat(cos(self.NA), self.qx.shape)

    def outputwavevectors(self):
        onx, ony, onz = self.entrypts()
        oxx, oxy, oxz = self.exitpts()
        return np.stack((self.qx + onx, self.qx + oxx), axis=1), np.stack((self.qy + ony, self.qy + oxy), axis=1), np.stack((self.qz + onz, self.qz + oxz), axis=1)

    def detectorpts(self, detdist):
        kx, ky, kz = self.outputwavevectors()
        x = detdist * np.tan(np.sqrt(2 - 2 * kz)) * np.cos(np.arctan2(ky, kx))
        y = detdist * np.tan(np.sqrt(2 - 2 * kz)) * np.sin(np.arctan2(ky, kx))
        return x, y

class LineDetector(object, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def _refiner(lines, angles, rs, taus, drtau, drn): pass
    
    @abstractmethod
    def _detector(self, frame): pass

    def detectFrameRaw(self, frame):
        return np.array([[[x0, y0], [x1, y1]] for (x0, y0), (x1, y1) in self._detector(frame)])

    def detectFrame(self, frame, zero, drtau, drn):
        lines = FrameStreaks(self.detectFrameRaw(frame), zero)
        return FrameStreaks(type(self)._refiner(lines.lines, lines.angles, lines.radii, lines.taus, drtau, drn), zero)

    def detectScanRaw(self, data): return [self.detectFrameRaw(frame) for frame in data]

    def detectScan(self, data, zero, drtau, drn): return ScanStreaks([self.detectFrame(frame, zero, drtau, drn) for frame in data])

class HoughLineDetector(LineDetector):
    def __init__(self, threshold, line_length, line_gap, dth):
        self.trhd, self.ll, self.lg = threshold, line_length, line_gap
        self.thetas = np.linspace(-np.pi / 2, np.pi / 2, int(np.pi / dth), endpoint=True)

    @staticmethod
    @nb.njit(nb.int64[:, :, :](nb.int64[:, :, :], nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64, nb.float64))
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
        return probabilistic_hough_line(frame, threshold=self.trhd, line_length=self.ll, line_gap=self.lg, theta=self.thetas)

class LineSegmentDetector(LineDetector):
    def __init__(self, scale=0.8, sigma_scale=0.6, log_eps=0):
        self.detector = createLineSegmentDetector(_scale=scale, _sigma_scale=sigma_scale, _log_eps=log_eps)
    
    @staticmethod
    @nb.njit(nb.float64[:, :, :](nb.float64[:, :, :], nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64, nb.float64))
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
    def size(self): return self.lines.shape[0]

    @property
    def xs(self): return self.pts[:, 0]

    @property
    def ys(self): return self.pts[:, 1]

    @property
    def radii(self): return np.sqrt(self.xs**2 + self.ys**2)

    @property
    def angles(self): return np.arctan2(self.ys, self.xs)

    @property
    def taus(self):
        taus = (self.lines[:, 1] - self.lines[:, 0])
        return taus / np.sqrt(taus[:,0]**2 + taus[:,1]**2)[:, np.newaxis]

    def __iter__(self):
        for line in self.lines: yield line.astype(np.int64)

    def indexpoints(self):
        ts = self.dlines[:, 0, 1] * self.taus[:, 0] - self.dlines[:, 0, 0] * self.taus[:, 1]
        return np.stack((-self.taus[:, 1] * ts + self.zero[0], self.taus[:, 0] * ts + self.zero[1]), axis=1)

    def intensities(self, frame):
        ints = []
        for line in iter(self):
            rr, cc, val = line_aa(line[0, 1], line[0, 0], line[1, 1], line[1, 0])
            ints.append((frame[rr, cc] * val).sum())
        return np.array(ints)

class ScanStreaks(object):
    def __init__(self, streakslist):
        self.strkslist = streakslist

    @property
    def shapes(self): return np.array(list(accumulate([strks.size for strks in self.strkslist], lambda x, y: x + y)))

    @property
    def zero(self): return self.__getitem__(0).zero

    @staticmethod
    @nb.njit(nb.float64[:,:](nb.float64[:,:],  nb.int64[:], nb.float64))
    def _refiner(qs, shapes, dk):
        b = len(shapes)
        out = np.empty(qs.shape, dtype=np.float64)
        idxs = []; jj = 0; count = 0
        for i in range(shapes[b - 2]):
            if i == shapes[jj]: jj += 1
            if i in idxs: continue
            qslist = []
            for j in range(shapes[jj], shapes[jj + 1]):
                if sqrt((qs[i,0] - qs[j,0])**2 + (qs[i,1] - qs[j,1])**2 + (qs[i,2] - qs[j,2])**2) < dk:
                    qslist.append(qs[i]); idxs.append(i)
                    break
            else:
                out[count] = qs[i]; count += 1
                continue
            for k in range(jj, b - 1):
                skip = True; q = qslist[-1]
                for l in range(shapes[k], shapes[k + 1]):
                    if sqrt((q[0] - qs[l,0])**2 + (q[1] - qs[l,1])**2 + (q[2] - qs[l,2])**2) < dk:
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

    def qs(self, axis, thetas, pixsize, detdist):
        qslist = []
        for strks, theta in zip(iter(self), thetas):
            kxs = np.arctan(pixsize * strks.radii / detdist) * np.cos(strks.angles)
            kys = np.arctan(pixsize * strks.radii / detdist) * np.sin(strks.angles)
            rotm = utils.rotation_matrix(axis, theta)
            qxs, qys, qzs = utils.rotate(rotm, kxs, kys, np.sqrt(1 - kxs**2 - kys**2) - 1)
            qslist.append(np.stack((qxs, qys, qzs), axis=1))
        return ReciprocalPeaks(np.concatenate(qslist))

    def refined_qs(self, axis, thetas, pixsize, detdist, dk):
        qs = self.qs(axis, thetas, pixsize, detdist).qs
        return ReciprocalPeaks(ScanStreaks._refiner(qs, self.shapes, dk))

    def save(self, data, outfile):
        linesgroup = outfile.create_group('bragg_lines')
        intsgroup = outfile.create_group('bragg_intensities')
        for idx, (streaks, frame) in enumerate(zip(iter(self), data)):
            linesgroup.create_dataset(str(idx), data=streaks.lines)
            intsgroup.create_dataset(str(idx), data=streaks.intensities(frame))

class ReciprocalPeaks(object):
    def __init__(self, qs):
        self.qs = qs

    @staticmethod
    @nb.njit(nb.uint64[:, :, :](nb.float64[:,:], nb.float64, nb.int64), parallel=True)
    def _corgrid_func(qs, qmax, size):
        a = qs.shape[0]
        corgrid = np.zeros((size, size, size), dtype=np.uint64)
        ks = np.linspace(-qmax, qmax, size)
        for i in nb.prange(a):
            for j in range(i + 1, a):
                dk = qs[i] - qs[j]
                if abs(dk[0]) < qmax and abs(dk[1]) < qmax and abs(dk[2]) < qmax:
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
    def _cor_func(qs, qmax):
        a = qs.shape[0]
        cor = np.empty((int(a * (a - 1) / 2), 3), dtype=np.float64)
        count = 0
        for i in nb.prange(a):
            for j in range(i + 1, a):
                dk = qs[i] - qs[j]
                if abs(dk[0]) < qmax and abs(dk[1]) < qmax and abs(dk[2]) < qmax:
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

    def correlation_grid(self, qmax, size):
        return ReciprocalPeaks._corgrid_func(self.qs, qmax, size)

    def correlation(self, qmax):
        return ReciprocalPeaks._cor_func(self.qs, qmax)

    def grid(self, size):
        return ReciprocalPeaks._grid(self.qs, size)

@nb.njit(nb.float64[:, :](nb.float64[:, :]))
def NMS(image):
    a, b = image.shape
    res = np.zeros((a, b), dtype=np.float64)
    for i in range(1, a - 1):
        for j in range(1, b - 1):
            phase = atan2(image[i+1, j] - image[i-1, j], image[i, j+1] - image[i,j-1])
            if (phase >= 0.875 * pi or phase < -0.875 * pi) or (phase >= -0.125 * pi and phase < 0.125 * pi):
                if image[i,j] >= image[i,j+1] and image[i,j] >= image[i,j-1]:
                    res[i,j] = image[i,j]
            if (phase >= 0.625 * pi and phase < 0.875 * pi) or (phase >= -0.375 * pi and phase < -0.125 * pi):
                if image[i,j] >= image[i-1,j+1] and image[i,j] >= image[i+1,j-1]:
                    res[i,j] = image[i,j]
            if (phase >= 0.375 * pi and phase < 0.625 * pi) or (phase >= -0.625 * pi and phase < -0.375 * pi):
                if image[i,j] >= image[i-1,j] and image[i,j] >= image[i+1,j]:
                    res[i,j] = image[i,j]
            if (phase >= 0.125 * pi and phase < 0.375 * pi) or (phase >= -0.875 * pi and phase < -0.625 * pi):
                if image[i,j] >= image[i-1,j-1] and image[i,j] >= image[i+1,j+1]:
                    res[i,j] = image[i,j]
    return res