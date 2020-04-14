"""
utils.py - Uitility constants and functions module
"""
import os
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
OUT_PATH = {'scan': os.path.join(PROJECT_PATH, "exp_results/scan_{0:05d}"),
            'frame': os.path.join(PROJECT_PATH, "exp_results/count_{0:05d}")}
FILENAME = {'scan': "scan_{0:s}_{1:05d}.{2:s}", 'frame': "count_{0:s}_{1:05d}.{2:s}"}
DATA_FILENAME = {'scan': 'scan_{0:05d}_data_{1:06d}.h5', 'frame': 'count_{0:05d}_data_{1:06d}.h5'}
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
    