"""
utils.py - Uitility constants and functions module
"""
import os
import configparser
import numpy as np
import h5py
from cv2 import cvtColor, COLOR_BGR2GRAY

RAW_PATH = "/asap3/petra3/gpfs/p06/2019/data/11006252/raw"
PROJECT_PATH = os.path.abspath('.')
PREFIXES = {'alignment': '0001_alignment',
            'opal': '0001_opal',
            'b12_1': '0002_b12_1',
            'b12_2': '0002_b12_2',
            'imaging': '0003_imaging1'}
HOTMASK = np.load(os.path.join(PROJECT_PATH, "cbc_dp/utils/P06_mask.npy"))
MEAS_PATH = {'scan': "scan_{0:05d}", "frame": "count_{0:05d}"}
EIGER_PATH = "eiger4m_01"
NXS_PATH = "/scan/program_name"
COMMAND_PATH = "scan_command"
DATA_PATH = "entry/data/data"
ENERGY_PATH = "scan/data/energy"
OUT_PATH = {'scan': os.path.join(PROJECT_PATH, "exp_results/scan_{scan_num:05d}"),
            'frame': os.path.join(PROJECT_PATH, "exp_results/count_{scan_num:05d}")}
FILENAME = {'scan': "scan_{tag:s}_{scan_num:05d}.{ext:s}",
            'frame': "count_{scan_num:05d}.{ext:s}"}
DATA_FILENAME = {'scan': 'scan_{tag:05d}_data_{scan_num:06d}.h5',
                 'frame': 'count_data_{scan_num:06d}.h5'}
COMMANDS = {'single_frame': ('cnt', 'ct'),
            'scan1d': ('dscan', 'ascan'),
            'scan2d': ('dmesh', 'cmesh')}
ORIGINS = {107: np.array([1480, 1155]), 135: np.array([1470, 1710])}
DET_DIST = {'alignment': 0.9, 'imaging': 1.46}
STRUCT = np.array([[0, 0, 1, 0, 0],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [0, 0, 1, 0, 0]], dtype=np.uint8)

class HKLList():
    """
    HKL list class

    hkl_arr - hkl indices and counts
    is_arr - intensity and sigma
    """
    hkl_str = '{0[0]:-5d}{0[1]:-4d}{0[2]:-4d}{1[0]:-12.2f}{2:-12.2f}{1[1]:-11.2f}{0[3]:-9d}\n'
    hkl_header = 'Symmetry: mmm\n\th\tk\tl\t\t\tI\t\tphase\tsigma(I)\tnmeas\n'
    hkl_footer = 'End of reflections'

    def __init__(self, hkl_arr, is_arr):
        self.hkl_arr, self.is_arr = hkl_arr, is_arr
        self.phase = np.zeros(hkl_arr.shape[0])

    def save_txt(self, out_path):
        """
        Save to a text file
        """
        with open(out_path, 'w') as out_file:
            out_file.write(self.hkl_header)
            for hkl_val, is_val, ph_val in zip(self.hkl_arr, self.is_arr, self.phase):
                out_file.write(self.hkl_str.format(hkl_val, is_val, np.angle(ph_val)))
            out_file.write(self.hkl_footer)

def make_path(path, idx=0):
    """
    Return a nonexistant path to write a file
    """
    name, ext = os.path.splitext(path)
    new_path = name + "_{:02d}".format(idx) + ext
    if os.path.isfile(new_path):
        return make_path(path, idx + 1)
    else:
        return new_path

def scan_command(nxsfilepath):
    command = h5py.File(nxsfilepath, 'r')[NXS_PATH].attrs[COMMAND_PATH]
    if isinstance(command, np.ndarray):
        command = str(command)[2:-2]
    return command

def arraytoimg(array):
    """
    Convert numpy array to OpenCV image

    array - numpy array
    """
    img = np.tile((array / array.max() * 255).astype(np.uint8)[..., np.newaxis], (1, 1, 3))
    return cvtColor(img, COLOR_BGR2GRAY)

def rec_basis(basis):
    """
    Return orientation matrix based on unit cell primitive vectors matrix

    basis - unit cell primitive vectors matrix
    """
    a_rec = np.cross(basis[1], basis[2]) / (np.cross(basis[1], basis[2]).dot(basis[0]))
    b_rec = np.cross(basis[2], basis[0]) / (np.cross(basis[2], basis[0]).dot(basis[1]))
    c_rec = np.cross(basis[0], basis[1]) / (np.cross(basis[0], basis[1]).dot(basis[2]))
    return np.stack((a_rec, b_rec, c_rec))

def proj(vec, axis):
    """
    Return the vector projection onto an axis
    """
    return ((vec * axis).sum(axis=-1) / (axis**2).sum(axis=-1))[..., None] * axis

def gramm_schmidt(or_mat):
    """
    Return orthogonalized orientation matrix using Gramm Schmidt orthogonalization
    """
    b_vec = or_mat[1] - proj(or_mat[1], or_mat[0])
    c_vec = or_mat[2] - proj(or_mat[2], b_vec) - proj(or_mat[2], or_mat[0])
    return np.stack((or_mat[0], b_vec, c_vec))

def find_reduced(vectors, basis):
    """
    Find reduced vector to basis set

    vectors - array of vectors of shape (N, 3)
    basis - basis set of shape (M, 3)
    """
    prod = vectors.dot(basis.T)
    mask = 2 * np.abs(prod) < (basis * basis).sum(axis=1)
    return np.where(mask.all(axis=1))

class INIParser():
    """
    INI files parser class
    """
    section = None
    data_dict = {}

    @staticmethod
    def get_int_array(string):
        """
        Integer numpy.ndarray converter
        """
        return np.array([int(coord) for coord in string.strip('[]').split()])

    @staticmethod
    def get_float_array(string):
        """
        Float numpy.ndarray converter
        """
        return np.array([float(coord) for coord in string.strip('[]').split()])

    @classmethod
    def ini_parser(cls):
        """
        Return config parser
        """
        return configparser.ConfigParser(converters={'intarr': cls.get_int_array,
                                                     'floatarr': cls.get_float_array})

    @classmethod
    def read_ini(cls, ini_file):
        """
        Read ini file
        """
        if not os.path.isfile(ini_file):
            raise ValueError("File {:s} doesn't exist".format(ini_file))
        ini_parser = cls.ini_parser()
        ini_parser.read(ini_file)
        return ini_parser

    def __getattr__(self, attr):
        if attr in self.data_dict:
            return self.data_dict[attr]

    def save_ini(self, filename):
        """
        Save experiment settings to an ini file
        """
        ini_parser = self.ini_parser()
        ini_parser[self.section] = self.data_dict
        with open(filename, 'w') as ini_file:
            ini_parser.write(ini_file)
