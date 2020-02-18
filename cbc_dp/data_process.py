"""
wrapper.py - Eiger 4M detector data processing from Petra P06 beamline module
"""
import os
import concurrent.futures
from abc import ABCMeta, abstractmethod, abstractproperty
from scipy.ndimage import median_filter, binary_dilation, binary_fill_holes, label, labeled_comprehension
from scipy import constants
import numpy as np
import h5py
from . import utils
from .feat_detect import LineSegmentDetector

class Measurement(metaclass=ABCMeta):
    """
    Abstract Eiger 4M measurements data class

    prefix - experiment prefix ('b12_1', 'b12_2', 'opal', 'imaging', 'alignment')
    scan_num - scan number
    """
    mask = utils.HOTMASK

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = prefix, scan_num
        self.raw_data = self._init_raw()
        self.data = self.mask * self.raw_data

    @abstractproperty
    def mode(self):
        pass

    @abstractproperty
    def size(self):
        pass

    @abstractmethod
    def _init_raw(self):
        pass

    @abstractmethod
    def _save_data(self, outfile):
        pass

    @property
    def path(self):
        return os.path.join(utils.RAW_PATH,
                            utils.PREFIXES[self.prefix],
                            utils.MEAS_PATH[self.mode].format(self.scan_num))

    @property
    def nxsfilepath(self):
        return self.path + '.nxs'

    @property
    def command(self):
        return utils.scan_command(self.nxsfilepath)

    @property
    def data_path(self):
        return os.path.join(self.path, utils.EIGER_PATH)

    @property
    def energy(self):
        return h5py.File(self.nxsfilepath, 'r')[utils.ENERGY_PATH][0] * constants.e

    @property
    def exposure(self):
        parts = self.command.split(" ")
        try:
            exposure = float(parts[-1])
        except ValueError:
            exposure = float(parts[-2])
        return exposure

    @property
    def out_path(self):
        return os.path.join(os.path.dirname(__file__),
                            utils.OUT_PATH[self.mode].format(self.scan_num))

    def filename(self, tag, ext):
        return utils.FILENAME[self.mode].format(tag, self.scan_num, ext)

    def _create_outfile(self, tag, ext='h5'):
        os.makedirs(self.out_path, exist_ok=True)
        return h5py.File(os.path.join(self.out_path, self.filename(tag, ext)), 'w')

    def _save_parameters(self, outfile):
        arggroup = outfile.create_group('arguments')
        arggroup.create_dataset('experiment', data=self.prefix)
        arggroup.create_dataset('scan mode', data=self.__class__.__name__)
        arggroup.create_dataset('scan number', data=self.scan_num)
        arggroup.create_dataset('raw path', data=self.path)
        arggroup.create_dataset('command', data=self.command)
        arggroup.create_dataset('energy', data=self.energy)
        arggroup.create_dataset('exposure', data=self.exposure)

    def save_raw(self):
        """
        Save raw data
        """
        outfile = self._create_outfile(tag='raw')
        self._save_parameters(outfile)
        self._save_data(outfile)
        outfile.close()

def open_scan(prefix, scan_num, good_frames=None):
    """
    Automatically detect scan type and return scan object

    prefix - experiment prefix ('b12_1', 'b12_2', 'opal', 'imaging', 'alignment')
    scan_num - scan number
    """
    path = os.path.join(os.path.join(utils.RAW_PATH,
                                     utils.PREFIXES[prefix],
                                     utils.MEAS_PATH['scan'].format(scan_num)))
    command = utils.scan_command(path + '.nxs')
    if command.startswith(utils.COMMANDS['scan1d']):
        return Scan1D(prefix, scan_num, good_frames)
    elif command.startswith(utils.COMMANDS['scan2d']):
        return Scan2D(prefix, scan_num, good_frames)
    else:
        raise ValueError('Unknown scan type')

class Frame(Measurement):
    """
    Single Eiger 4M frame class

    prefix - experiment prefix ('b12_1', 'b12_2', 'opal', 'imaging', 'alignment')
    scan_num - scan number
    mode - measurement mode ('scan', 'frame')
    """
    size = (1,)
    mode = None

    def __init__(self, prefix, scan_num, mode='frame'):
        self.mode = mode
        super(Frame, self).__init__(prefix, scan_num)

    @property
    def data_filename(self):
        return utils.DATA_FILENAME[self.mode].format(self.scan_num, 1)

    def _init_raw(self):
        raw_file = h5py.File(os.path.join(self.data_path, self.data_filename), 'r')
        return raw_file[utils.DATA_PATH][:].sum(axis=0, dtype=np.uint64)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.data, compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')

class ABCScan(Measurement, metaclass=ABCMeta):
    """
    Abstract scan class

    prefix - experiment prefix ('b12_1', 'b12_2', 'opal', 'imaging', 'alignment')
    scan_num - scan number
    good_frames - a mask of frames to extract
    """
    mode = 'scan'

    def __init__(self, prefix, scan_num, good_frames=None):
        super(ABCScan, self).__init__(prefix, scan_num)
        self.coordinates = self._init_coord()
        if good_frames is None:
            self.good_frames = np.arange(0, self.raw_data.shape[0])
        else:
            self.good_frames = good_frames

    @abstractmethod
    def _init_coord(self):
        pass

    @abstractmethod
    def data_chunk(self, paths):
        pass

    @property
    def attributes(self):
        nums = []
        for num in self.command.split(" "):
            try:
                nums.append(float(num))
            except ValueError:
                continue
        return np.array(nums[:-1])

    def _init_raw(self):
        paths_list = [os.path.join(self.data_path, filename)
                      for filename in os.listdir(self.data_path)
                      if not filename.endswith('master.h5')]
        paths = np.sort(np.array(paths_list, dtype=object))
        thread_num = min(paths.size, utils.CPU_COUNT)
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for paths_chunk in np.array_split(paths, thread_num):
                futures.append(executor.submit(self.data_chunk, paths_chunk))
        return np.concatenate([future.result() for future in futures if future.result().any()])

class Scan1D(ABCScan, metaclass=ABCMeta):
    """
    1-dimensional scan class

    prefix - experiment prefix ('b12_1', 'b12_2', 'opal', 'imaging', 'alignment')
    scan_num - scan number
    good_frames - a mask of frames to extract
    """
    @property
    def size(self):
        return (self.coordinates['fs_coordinates'].size,)

    def _init_coord(self):
        return {'fs_coordinates': np.linspace(self.attributes[0],
                                              self.attributes[1],
                                              int(self.attributes[2]) + 1)}

    def data_chunk(self, paths):
        """
        Return data from files of the given paths
        """
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try:
                    data_list.append(datafile[utils.DATA_PATH][:].sum(axis=0, dtype=np.uint64))
                except KeyError:
                    continue
        if not data_list:
            return np.array([])
        else:
            return np.stack(data_list, axis=0)

    def corrected_data(self, mask=None):
        """
        Perform CBC data correction

        mask - bad pixels mask
        """
        return CorrectedData(self.data, mask)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.raw_data, compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')
        for key in self.coordinates:
            datagroup.create_dataset(key, data=self.coordinates[key])

    def save_corrected(self, mask=None):
        """
        Save the raw CBC data, correct the data, detect streaks,

        mask - bad pixel mask
        """
        outfile = self._create_outfile(tag='corrected')
        self._save_parameters(outfile)
        self._save_data(outfile)
        cordata = self.corrected_data(mask)
        cordata.save(outfile)
        outfile.close()

    def save_streaks(self, exp_set, d_tau=1.8, d_n=1.4, mask=None):
        """
        Save the raw CBC data, correct the data, detect streaks,
        and save the data to an HDF5 file

        exp_set - FrameSetup class object
        d_tau - tangent detection error
        d_n - radial detection error
        mask - bad pixel mask
        """
        out_file = self._create_outfile(tag='streaks')
        self._save_parameters(out_file)
        self._save_data(out_file)
        cor_data = self.corrected_data(mask)
        cor_data.save(out_file)
        norm_data = cor_data.normalize_data()
        det_scan = norm_data.detect(exp_set, d_tau, d_n)
        det_scan.save(out_file)
        out_file.close()

class Scan2D(Scan1D):
    """
    2-dimensional scan class

    prefix - experiment prefix ('b12_1', 'b12_2', 'opal', 'imaging', 'alignment')
    scan_num - scan number
    good_frames - a mask of frames to extract
    """
    @property
    def size(self):
        return (self.coordinates['fs_coordinates'].size,
                self.coordinates['ss_coordinates'].size)

    def _init_coord(self):
        return {'fs_coordinates': np.linspace(self.attributes[3],
                                              self.attributes[4],
                                              int(self.attributes[5]) + 1),
                'ss_coordinates': np.linspace(self.attributes[0],
                                              self.attributes[1],
                                              int(self.attributes[2]) + 1)}

class ScanST(ABCScan):
    """
    Speckle tracking scan class

    prefix - experiment prefix ('b12_1', 'b12_2', 'opal', 'imaging', 'alignment')
    scan_num - scan number
    ff_num - flatfield scan number
    good_frames - a mask of frames to extract
    flip_axes - flag to flip detector axes
    """
    pixel_vector = np.array([7.5e-5, 7.5e-5, 0])
    unit_vector_fs = np.array([0, -1, 0])
    unit_vector_ss = np.array([-1, 0, 0])

    def __init__(self, prefix, scan_num, ff_num, good_frames=None, flip_axes=False):
        super(ScanST, self).__init__(prefix, scan_num, good_frames)
        self.flip_axes = flip_axes
        self.flatfield = Frame(self.prefix, ff_num, 'scan').data

    def _init_coord(self):
        return {'fs_coordinates': np.linspace(self.attributes[3],
                                              self.attributes[4],
                                              int(self.attributes[5]) + 1),
                'ss_coordinates': np.linspace(self.attributes[0],
                                              self.attributes[1],
                                              int(self.attributes[2]) + 1)}

    @property
    def size(self):
        return (self.coordinates['fs_coordinates'].size,
                self.coordinates['ss_coordinates'].size)

    @property
    def detector_distance(self):
        return utils.DET_DIST[self.prefix]

    @property
    def x_pixel_size(self):
        return self.pixel_vector[0]

    @property
    def y_pixel_size(self):
        return self.pixel_vector[1]

    @property
    def wavelength(self):
        return constants.c * constants.h / self.energy

    def basis_vectors(self):
        """
        Return detector fast and slow axis basis vectors
        """
        _vec_fs = np.tile(self.pixel_vector * self.unit_vector_fs, (self.size[0] * self.size[1], 1))
        _vec_ss = np.tile(self.pixel_vector * self.unit_vector_ss, (self.size[0] * self.size[1], 1))
        if self.flip_axes:
            return np.stack((_vec_ss, _vec_fs), axis=1)
        else:
            return np.stack((_vec_fs, _vec_ss), axis=1)

    def data_chunk(self, paths):
        """
        Return data from files of the given paths
        """
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try:
                    data_list.append(datafile[utils.DATA_PATH][:])
                except KeyError:
                    continue
        return None if not data_list else np.concatenate(data_list, axis=0)

    def translation(self):
        """
        Translate detector coordinates
        """
        _x_pos = np.tile(self.coordinates['fs_coordinates'] * 1e-6, self.size[1])
        _y_pos = np.repeat(self.coordinates['ss_coordinates'] * 1e-6, self.size[0])
        _z_pos = np.zeros(self.size[0] * self.size[1])
        return np.stack((_x_pos, _y_pos, _z_pos), axis=1)

    def _save_data(self, outfile):
        outfile.create_dataset('frame_selector/good_frames', data=self.good_frames)
        outfile.create_dataset('mask_maker/mask', data=self.mask)
        outfile.create_dataset('make_whitefield/whitefield', data=self.flatfield)
        outfile.create_dataset('entry_1/data_1/data', data=self.data)

    def save_st(self):
        """
        Save scan data for speckle tracking
        """
        outfile = self._create_outfile(tag='st', ext='cxi')
        detector_1 = outfile.create_group('entry_1/instrument_1/detector_1')
        detector_1.create_dataset('basis_vectors', data=self.basis_vectors())
        detector_1.create_dataset('distance', data=self.detector_distance)
        detector_1.create_dataset('x_pixel_size', data=self.x_pixel_size)
        detector_1.create_dataset('y_pixel_size', data=self.y_pixel_size)
        source_1 = outfile.create_group('entry_1/instrument_1/source_1')
        source_1.create_dataset('energy', data=self.energy)
        source_1.create_dataset('wavelength', data=self.wavelength)
        outfile.create_dataset('entry_1/sample_3/geometry/translation', data=self.translation())
        self._save_data(outfile)

class CorrectedData(object):
    """
    CBC data correction class. Calculate background, content and streaks mask.

    data - raw data
    mask - bad pixels mask
    """
    bgd_kernel = (11, 1)
    line_detector = LineSegmentDetector(scale=0.6, sigma_scale=0.4)

    def __init__(self, data, mask=None):
        self.data = data
        if mask is None:
            self.bad_mask = np.median((self.data > np.percentile(self.data, 99)).astype(np.uint8),
                                      axis=0)
        else:
            self.bad_mask = mask
        self._init_background()
        self._mask_data()

    def _init_background(self):
        idxs = np.where(self.bad_mask == 0)
        data = self.data[:, idxs[0], idxs[1]]
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for data_chunk in np.array_split(data, utils.CPU_COUNT, axis=1):
                futures.append(executor.submit(CorrectedData._background_worker, data_chunk))
        self.background = np.copy(self.data)
        self.cor_data = np.zeros(self.data.shape, dtype=np.int64)
        filt_data = np.concatenate([future.result() for future in futures], axis=1)
        self.background[:, idxs[0], idxs[1]] = filt_data
        self.cor_data[:, idxs[0], idxs[1]] = data - filt_data

    def _mask_data(self):
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for frame_data in self.cor_data:
                futures.append(executor.submit(CorrectedData._mask_worker, frame_data))
        self.streaks_mask = np.stack([future.result() for future in futures], axis=0)

    @classmethod
    def _background_worker(cls, data):
        return median_filter(data, size=cls.bgd_kernel)

    @classmethod
    def _mask_worker(cls, frame_data):
        streaks = cls.line_detector.det_frame_raw(median_filter(frame_data, 3))
        streaks_mask = utils.draw_lines_aa(lines=streaks.astype(np.int64),
                                         w=1,
                                         shape_x=frame_data.shape[0],
                                         shape_y=frame_data.shape[1])
        streaks_mask = binary_dilation(streaks_mask, iterations=3)
        streaks_mask = binary_fill_holes(streaks_mask)
        return streaks_mask

    def normalize_data(self):
        """
        Normalize CBC data for streaks detection
        """
        return NormalizedData(self.cor_data, self.streaks_mask)

    def save(self, out_file):
        """
        Save corrected data to an out_file

        out_file - h5py file object
        """
        correct_group = out_file.create_group('corrected_data')
        correct_group.create_dataset('corrected_data', data=self.cor_data, compression='gzip')
        correct_group.create_dataset('streaks_mask', data=self.streaks_mask, compression='gzip')
        correct_group.create_dataset('bad_mask', data=self.bad_mask, compression='gzip')
        correct_group.create_dataset('background', data=self.background, compression='gzip')

class NormalizedData(object):
    """
    CBC data normalization for streaks detection.

    cor_data - corrected data
    streaks_mask - streaks mask
    """
    threshold = 3.2
    line_detector = LineSegmentDetector(scale=0.6)

    def __init__(self, cor_data, streaks_mask):
        self.cor_data, self.mask = cor_data, streaks_mask
        self._label_streaks()
        self._norm_data()

    def _label_streaks(self):
        self.labels, lbl_num = label(self.mask)
        self.n_labels = np.arange(1, lbl_num + 1)

    def _label_func(self, val, pos):
        if val.max() > val.mean() + self.threshold * val.std():
            self.norm_data[pos] = self.norm_data[pos] / val.mean()
        else:
            self.norm_data[pos] = 0

    def _norm_data(self):
        self.norm_data = (self.cor_data * self.mask).ravel()
        labeled_comprehension(self.cor_data, self.labels, self.n_labels, self._label_func,
                              float, None, True)
        self.norm_data = self.norm_data.reshape(self.cor_data.shape)

    def detect(self, exp_set, d_tau=1.8, d_n=1.4):
        """
        Detect detect streaks in diffraction data

        exp_set - FrameSetup class object
        d_tau - tangent detection error [pixels]
        d_n - radial detection error [pixels]
        """
        return self.line_detector.det_scan(self.norm_data, exp_set, d_tau=d_tau, d_n=d_n)
