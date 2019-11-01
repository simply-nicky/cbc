"""
utils.py - Uitility constants and functions module
"""
from math import sqrt, cos, sin
import os
from multiprocessing import cpu_count
import numpy as np
import numba as nb
import h5py
from cv2 import cvtColor, COLOR_BGR2GRAY

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
ORIGINS = {107: np.array([1480, 1155]), 135: np.array([1470, 1710])}
DET_DIST = {'alignment': 0.9, 'imaging': 1.46}

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

def rotation_matrix(axis, theta):
    """
    Return roatational matrix around axis to theta angle

    axis - rotational axis
    theta - angle of rotation
    """
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
    