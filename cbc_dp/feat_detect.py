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

    pix_size - detector pixel size [mm]
    smp_pos - sample position relative to the detector [mm]
    z_f - distance between the focus and the detector [mm]
    pupil - pupil outline at the detector plane [pixels]
    beam_pos - unfocussed beam position at the detector plane [pixels]
    """
    def __init__(self, pix_size, smp_pos, z_f, pupil, beam_pos):
        self.pix_size, self.smp_pos = pix_size, smp_pos
        self.z_f, self.pupil, self.beam_pos = z_f, pupil * pix_size, beam_pos * pix_size
        self.kin = (self.pupil - self.beam_pos) / self.z_f

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
        dx = streaks[..., 0] * self.pix_size - self.smp_pos[0]
        dy = streaks[..., 1] * self.pix_size - self.smp_pos[1]
        phis = np.arctan2(dy, dx)
        thetas = np.arctan(np.sqrt(dx**2 + dy**2) / self.smp_pos[2])
        return np.stack((np.sin(thetas) * np.cos(phis),
                         np.sin(thetas) * np.sin(phis),
                         np.cos(thetas)), axis=-1)

    def det_pts(self, k_out):
        """
        Return diffraction streaks locations at the detector plane

        k_out - outcoming wavevectors
        """
        theta, phi = np.arccos(k_out[..., 2]), np.arctan2(k_out[..., 1], k_out[..., 0])
        det_x = self.smp_pos[2] * np.tan(theta) * np.cos(phi)
        det_y = self.smp_pos[2] * np.tan(theta) * np.sin(phi)
        return (np.stack((det_x, det_y), axis=-1) + self.smp_pos[:2]) / self.pix_size

    def save(self, out_file):
        """
        Save experiment settings to an HDF5 file

        out_file - h5py File object
        """
        out_group = out_file.create_group("experiment_settings")
        out_group.create_dataset('pix_size', data=self.pix_size)
        out_group.create_dataset('sample_pos', data=self.smp_pos)
        out_group.create_dataset('z_f', data=self.z_f)
        out_group.create_dataset('pupil', data=self.pupil)
        out_group.create_dataset('beam_pos', data=self.beam_pos)

class ScanSetup(FrameSetup):
    """
    Detector tilt scan experimental setup class

    pix_size - detector pixel size [mm]
    smp_pos - sample position relative to the detector [mm]
    z_f - distance between the focus and the detector [mm]
    pupil - pupil outline at the detector plane [pixels]
    beam_pos - unfocussed beam position at the detector plane [pixels]
    rot_axis - axis of rotation
    thetas - angles of rotation
    """
    def __init__(self, pix_size, smp_pos, z_f, pupil, beam_pos, rot_axis, thetas):
        super(ScanSetup, self).__init__(pix_size, smp_pos, z_f, pupil, beam_pos)
        self.axis, self.thetas = rot_axis, thetas
        self.eul_ang = utils.euler_angles_scan(self.axis, -self.thetas)

    @classmethod
    def from_frame_setup(cls, frame_setup, rot_axis, thetas):
        """
        Return ScanSetup object initialized with a FrameSetup object

        frame_setup - FrameSetup object
        rot_axis - axis of rotation
        thetas - angles of rotation
        """
        return cls(pix_size=frame_setup.pix_size, smp_pos=frame_setup.smp_pos, thetas=thetas,
                   z_f=frame_setup.z_f, pupil=frame_setup.pupil / frame_setup.pix_size,
                   beam_pos=frame_setup.beam_pos / frame_setup.pix_size, rot_axis=rot_axis)

    @property
    def scan_size(self):
        return self.thetas.size

    def rotation_matrix(self, frame_idx, inverse=False):
        """
        Return roational matrix for a given frame index
        """
        theta = -self.thetas[frame_idx] if inverse else self.thetas[frame_idx]
        return utils.rotation_matrix(self.axis, theta)

    def euler_angles(self, frame_idx):
        """
        Return Euler angles for a given frame index of the inversed rotation
        """
        return self.eul_ang[frame_idx]

    def save(self, out_file):
        """
        Save experiment settings to an HDF5 file

        out_file - h5py File object
        """
        out_group = out_file.create_group("experiment_settings")
        out_group.create_dataset('pix_size', data=self.pix_size)
        out_group.create_dataset('sample_pos', data=self.smp_pos)
        out_group.create_dataset('z_f', data=self.z_f)
        out_group.create_dataset('pupil', data=self.pupil)
        out_group.create_dataset('beam_pos', data=self.beam_pos)
        out_group.create_dataset('rot_axis', data=self.axis)
        out_group.create_dataset('thetas', data=self.thetas)

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