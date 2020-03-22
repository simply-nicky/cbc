"""
feat_detect.py - feature detection on convergent diffraction pattern module
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from skimage.transform import probabilistic_hough_line
from cv2 import createLineSegmentDetector
from . import utils
from .indexer import FrameStreaks, ScanStreaks

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