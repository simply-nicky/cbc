"""
utils.py - Uitility constants and functions module
"""
from math import sqrt, cos, sin
import os
import errno
from multiprocessing import cpu_count
import numpy as np
import numba as nb
import h5py
import cv2

CPU_COUNT = cpu_count()
RAW_PATH = "/asap3/petra3/gpfs/p06/2019/data/11006252/raw"
PREFIXES = {'alignment': '0001_alignment',
            'opal': '0001_opal',
            'b12_1': '0002_b12_1',
            'b12_2': '0002_b12_2',
            'imaging': '0003_imaging1'}
HOTMASK = np.load(os.path.join(os.path.dirname(__file__), "P06_mask.npy"))
MEAS_PATH = {'scan': "scan_{0:05d}", "frame": "count_{0:05d}"}
EIGER_PATH = "eiger4m_01"
NXS_PATH = "/scan/program_name"
COMMAND_PATH = "scan_command"
DATA_PATH = "entry/data/data"
ENERGY_PATH = "scan/data/energy"
OUT_PATH = {'scan': "../exp_results/scan_{0:05d}", 'frame': "../exp_results/count_{0:05d}"}
FILENAME = {'scan': "scan_{0:s}_{1:05d}.{2:s}", 'frame': "count_{0:s}_{1:05d}.{2:s}"}
DATA_FILENAME = {'scan': 'scan_{0:05d}_data_{1:06d}.h5', 'frame': 'count_{0:05d}_data_{1:06d}.h5'}
COMMANDS = {'single_frame': ('cnt', 'ct'),
            'scan1d': ('dscan', 'ascan'),
            'scan2d': ('dmesh', 'cmesh')}
MASKS = {107: np.load(os.path.join(os.path.dirname(__file__), '107_mask.npy')),
         135: np.load(os.path.join(os.path.dirname(__file__), '135_mask.npy'))}
ORIGINS = {107: np.array([1480, 1155]), 135: np.array([1470, 1710])}
DET_DIST = {'alignment': 0.9, 'imaging': 1.46}

def make_output_dir(path):
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise OSError(error.errno, error.strerror, error.filename)

def scan_command(nxsfilepath):
    command = h5py.File(nxsfilepath, 'r')[NXS_PATH].attrs[COMMAND_PATH]
    if isinstance(command, np.ndarray):
        command = str(command)[2:-2]
    return command

def energy(nxsfilepath):
    nxsfile = h5py.File(nxsfilepath, 'r')
    return nxsfile[ENERGY_PATH][0]

def get_attributes(command):
    nums = []
    for part in command.split(" "):
        try:
            nums.append(float(part))
        except ValueError:
            continue
    return tuple(nums[:-1])

def coordinates(command):
    nums = get_attributes(command)
    coord_dict = {'fast_crds': np.linspace(nums[0], nums[1], int(nums[2]) + 1)}
    return coord_dict

def coordinates2d(command):
    nums = get_attributes(command)
    fast_crds = np.linspace(nums[3], nums[4], int(nums[5]) + 1, endpoint=True)
    slow_crds = np.linspace(nums[0], nums[1], int(nums[2]) + 1, endpoint=True)
    return fast_crds, fast_crds.size, slow_crds, slow_crds.size

def arraytoimg(array):
    img = np.tile((array / array.max() * 255).astype(np.uint8)[..., np.newaxis], (1, 1, 3))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    _a = cos(theta / 2.0)
    _b, _c, _d = -axis * sin(theta / 2.0)
    return np.array([[_a * _a + _b * _b - _c * _c - _d * _d,
                      2 * (_b * _c + _a * _d),
                      2 * (_b * _d - _a * _c)],
                     [2 * (_b * _c - _a * _d),
                      _a * _a + _c * _c - _b * _b - _d * _d,
                      2 * (_c * _d + _a * _b)],
                     [2 * (_b * _d + _a * _c),
                      2 * (_c * _d - _a * _b),
                      _a * _a + _d * _d - _b * _b - _c * _c]])

@nb.njit(nb.types.UniTuple(nb.float64[:], 3)(nb.float64[:, :],
                                             nb.float64[:],
                                             nb.float64[:],
                                             nb.float64[:]), fastmath=True)
def rotate(m, xs, ys, zs):
    new_xs = np.empty(xs.shape, dtype=np.float64)
    new_ys = np.empty(ys.shape, dtype=np.float64)
    new_zs = np.empty(zs.shape, dtype=np.float64)
    for i in range(xs.size):
        new_xs[i] = m[0, 0] * xs[i] + m[0, 1] * ys[i] + m[0, 2] * zs[i]
        new_ys[i] = m[1, 0] * xs[i] + m[1, 1] * ys[i] + m[1, 2] * zs[i]
        new_zs[i] = m[2, 0] * xs[i] + m[2, 1] * ys[i] + m[2, 2] * zs[i]
    return new_xs, new_ys, new_zs
