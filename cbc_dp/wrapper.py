"""
wrapper.py - interface module with all the main classes
"""
import os
import concurrent.futures
from abc import ABCMeta, abstractmethod, abstractproperty
from functools import partial
from scipy.ndimage.filters import median_filter
from scipy import constants
import numpy as np
import h5py
from . import utils
from .data import LineSegmentDetector

class Measurement(metaclass=ABCMeta):
    mask = utils.HOTMASK

    def __init__(self, prefix, scan_num):
        self.prefix, self.scan_num = scan_num, prefix
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
        return utils.energy(self.nxsfilepath) * constants.e

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
        utils.make_output_dir(self.out_path)
        return h5py.File(os.path.join(self.out_path, self.filename(tag, ext)), 'w')

    def _save_parameters(self, outfile):
        arggroup = outfile.create_group('arguments')
        arggroup.create_dataset('experiment',data=self.prefix)
        arggroup.create_dataset('scan mode', data=self.__class__.__name__)
        arggroup.create_dataset('scan number', data=self.scan_num)
        arggroup.create_dataset('raw path', data=self.path)
        arggroup.create_dataset('command', data=self.command)
        arggroup.create_dataset('energy', data=self.energy)
        arggroup.create_dataset('exposure', data=self.exposure)

    def save_raw(self):
        outfile = self._create_outfile(tag='raw')
        self._save_parameters(outfile)
        self._save_data(outfile)
        outfile.close()

def OpenScan(prefix, scan_num, good_frames=None):
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
    size = (1,)
    mode = None

    def __init__(self, prefix, scan_num, mode='frame'):
        super(Frame, self).__init__(prefix, scan_num)
        self.mode = mode

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
        return np.concatenate([future.result() for future in futures])

class Scan1D(ABCScan, metaclass=ABCMeta):
    @property
    def size(self):
        return (self.coordinates['fs_coordinates'].size,)

    def _init_coord(self):
        return {'fs_coordinates': np.linspace(self.attributes[0],
                                              self.attributes[1],
                                              int(self.attributes[2]) + 1)}

    def data_chunk(self, paths):
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try:
                    data_list.append(datafile[utils.DATA_PATH][:].sum(axis=0, dtype=np.uint64))
                except KeyError:
                    continue
        return None if not data_list else np.stack(data_list, axis=0)

    def corrected_data(self, ffnum):
        flatfield = Frame(self.prefix, ffnum, 'scan').data
        return CorrectedData(self.data, flatfield, self.good_frames)

    def _save_data(self, outfile):
        datagroup = outfile.create_group('data')
        datagroup.create_dataset('data', data=self.raw_data, compression='gzip')
        datagroup.create_dataset('mask', data=self.mask, compression='gzip')
        for key in self.coordinates:
            datagroup.create_dataset(key, data=self.coordinates[key])

    def save_corrected(self, ffnum):
        outfile = self._create_outfile(tag='corrected')
        self._save_parameters(outfile)
        self._save_data(outfile)
        cordata = self.corrected_data(ffnum)
        cordata.save(outfile)
        outfile.close()

    def save_streaks(self, ffnum, zero, drtau, drn):
        outfile = self._create_outfile(tag='corrected')
        self._save_parameters(outfile)
        self._save_data(outfile)
        cordata = self.corrected_data(ffnum)
        cordata.save(outfile)
        streaks = LineSegmentDetector().det_scan(cordata.strks_data, zero, drtau, drn)
        streaks.save(self.raw_data, outfile)
        outfile.close()

class Scan2D(Scan2D):
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
        _vec_fs = np.tile(self.pixel_vector * self.unit_vector_fs, (self.size[0] * self.size[1], 1))
        _vec_ss = np.tile(self.pixel_vector * self.unit_vector_ss, (self.size[0] * self.size[1], 1))
        if self.flip_axes:
            return np.stack((_vec_ss, _vec_fs), axis=1)
        else:
            return np.stack((_vec_fs, _vec_ss), axis=1)

    def data_chunk(self, paths):
        data_list = []
        for path in paths:
            with h5py.File(path, 'r') as datafile:
                try:
                    data_list.append(datafile[utils.DATA_PATH][:])
                except KeyError:
                    continue
        return None if not data_list else np.concatenate(data_list, axis=0)

    def translation(self):
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
    bgd_filter = partial(median_filter, size=(30, 1))
    strks_filter = partial(median_filter, size=(1, 3, 3))
    THRESHOLD = 10

    def __init__(self, data, flatfield, good_frames):
        self.data, self.flatfield = data[good_frames], flatfield
        self._init_sub()
        self._init_bgd()
        self._init_strks()

    def _init_sub(self):
        self.sub_data = (self.data - self.flatfield[np.newaxis, :]).astype(np.int64)
        self.bad_mask = (self.data > np.percentile(self.data, 99, axis=(1, 2))).astype(np.uint8)
        self.bad_mask[self.sub_data < 0] = 1

    def _init_bgd(self):
        idxs = np.where(self.bad_mask == 0)
        data = self.sub_data[:, idxs[0], idxs[1]]
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for data_chunk in np.array_split(data, utils.CPU_COUNT, axis=1):
                futures.append(executor.submit(self.bgd_filter, data_chunk))
        self.background = np.copy(self.sub_data)
        filt_data = np.concatenate([future.result() for future in futures], axis=1)
        self.background[:, idxs[0], idxs[1]] = filt_data

    def _init_strks(self):
        sub = (self.sub_data - self.background).astype(np.int64)
        self.strks_data = np.where(sub - self.background > self.THRESHOLD, self.sub_data, 0)
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for data_chunk in np.array_split(self.strks_data, utils.CPU_COUNT):
                futures.append(executor.submit(self.strks_filter, data_chunk))
        filt_data = np.concatenate([future.result() for future in futures])
        self.strksdata = filt_data

    def save(self, outfile):
        correct_group = outfile.create_group('corrected_data')
        correct_group.create_dataset('flatfield', data=self.flatfield, compression='gzip')
        correct_group.create_dataset('corrected_data', data=self.sub_data, compression='gzip')
        correct_group.create_dataset('bad_mask', data=self.bad_mask, compression='gzip')
        correct_group.create_dataset('background', data=self.background, compression='gzip')
        correct_group.create_dataset('streaks_data', data=self.strks_data, compression='gzip')
    