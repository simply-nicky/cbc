"""
feat_detect.py - feature detection on convergent diffraction pattern module
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from skimage.transform import probabilistic_hough_line
from cv2 import createLineSegmentDetector
from .utils import INIParser, scan_eul_ang, rotation_matrix, hl_refiner, lsd_refiner, arraytoimg
from .indexer import FrameStreaks, ScanStreaks

class ScanSetup(INIParser):
    """
    Detector tilt scan experimental setup class

    pix_size - detector pixel size [mm]
    smp_pos - sample position relative to the detector [mm]
    f_pos - focus position relative to the detector [mm]
    pupil - pupil outline at the detector plane for every frame [pixels]
    axis - axis of rotation
    thetas - angles of rotation
    """
    section = 'exp_geom'

    def __init__(self, pix_size, smp_pos, f_pos, pupil, axis, thetas):
        self.data_dict = {'pix_size': pix_size, 'smp_pos': smp_pos, 'f_pos': f_pos,
                          'axis': axis, 'pupil': pupil.ravel(), 'thetas': thetas}
        self.eul_ang = scan_eul_ang(axis, -thetas)

    @classmethod
    def from_frame_setup(cls, frame_setup, pupil, axis, thetas):
        """
        Return ScanSetup object initialized with a FrameSetup object

        frame_setup - FrameSetup object
        pupil - pupil outline at the detector plane for every frame [pixels]
        axis - axis of rotation
        thetas - angles of rotation
        """
        return cls(pix_size=frame_setup.pix_size, smp_pos=frame_setup.smp_pos, 
                   f_pos=frame_setup.f_pos, pupil=pupil, axis=axis, thetas=thetas)

    @classmethod
    def import_ini(cls, geom_file):
        """
        Import ScanSetup class object from an ini file

        geom_file - path to a file
        """
        ini_parser = cls.read_ini(geom_file)
        pix_size = ini_parser.getfloat(cls.section, 'pix_size')
        smp_pos = ini_parser.getfloatarr(cls.section, 'smp_pos')
        f_pos = ini_parser.getfloatarr(cls.section, 'f_pos')
        pupil = ini_parser.getintarr(cls.section, 'pupil')
        axis = ini_parser.getfloatarr(cls.section, 'axis')
        thetas = ini_parser.getfloatarr(cls.section, 'thetas')
        return cls(pix_size=pix_size, smp_pos=smp_pos, f_pos=f_pos,
                   pupil=pupil, axis=axis, thetas=thetas)

    @property
    def scan_size(self):
        return self.thetas.size

    def pixtoq(self, pixels):
        """
        Convert detector distance in pixels to distance in reciprocal space
        """
        return pixels * self.pix_size / self.smp_pos[2]

    def kout_exp(self, streaks):
        """
        Project detected diffraction streaks to reciprocal space vectors in [a.u.]

        streaks - detector points [pixels]
        """
        delta_x = streaks[..., 0] * self.pix_size - self.smp_pos[0]
        delta_y = streaks[..., 1] * self.pix_size - self.smp_pos[1]
        phis = np.arctan2(delta_y, delta_x)
        thetas = np.arctan(np.sqrt(delta_x**2 + delta_y**2) / self.smp_pos[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def det_kout(self, k_out):
        """
        Return outcoming wavevector's corresponding locations at the detector plane

        k_out - outcoming wavevectors
        """
        theta, phi = np.arccos(k_out[..., 2]), np.arctan2(k_out[..., 1], k_out[..., 0])
        det_x = self.smp_pos[2] * np.tan(theta) * np.cos(phi)
        det_y = self.smp_pos[2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + self.smp_pos[:2]) / self.pix_size

    def det_kin(self, k_in):
        """
        Return incoming wavevector's corresponding locations at the detector plane

        k_in - incoming wavevectors
        """
        theta, phi = np.arccos(k_in[..., 2]), np.arctan2(k_in[..., 1], k_in[..., 0])
        det_x = self.f_pos[2] * np.tan(theta) * np.cos(phi)
        det_y = self.f_pos[2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + self.f_pos[:2]) / self.pix_size

    def pupil_bounds(self, frame_idx):
        """
        Return pupil bounds at the detector
        """
        return self.pupil[4 * frame_idx:4 * frame_idx + 4].reshape((2, 2)) * self.pix_size

    def rotation_matrix(self, frame_idx, inverse=False):
        """
        Return roational matrix for a given frame index
        """
        theta = -self.thetas[frame_idx] if inverse else self.thetas[frame_idx]
        return rotation_matrix(self.axis, theta)

    def euler_angles(self, frame_idx):
        """
        Return Euler angles for a given frame index of the inversed rotation
        """
        return self.eul_ang[frame_idx]

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

    _refiner = staticmethod(hl_refiner)

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

    _refiner = staticmethod(lsd_refiner)

    def _detector(self, frame):
        cap = np.mean(frame[frame != 0]) + np.std(frame[frame != 0])
        img = arraytoimg(np.clip(frame, 0, cap))
        return self.detector.detect(img)[0][:, 0].reshape((-1, 2, 2)).astype(np.float64)
    