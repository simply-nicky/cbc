import pickle
import numpy as np
from scipy import constants
from cbc_dp import QIndexTF, qindex_point
from cbc_dp.utils import rotation_matrix

wavelength = constants.h * constants.c / 17000 / constants.e * 1e3
pupil_size = np.radians([0.74, 1.6])
b12_sizes = np.array([16.18, 21.23, 24.33]) * 1e-7
b12_rec_sizes = b12_sizes**-1 * wavelength
rec_basis_full = np.array([[-0.00795996, 0.04611703, 0.00019548],
                           [-0.03183985, -0.0060065, 0.00078193],
                           [-0.00236648, -0.00173342, 0.02502171]])
b12_scan = pickle.load(open('exp_results/b12_scan.p', 'rb'))
rot_axis = np.array([0, 1, 0])

frame_idx = 0
rot_m = rotation_matrix(rot_axis, np.radians(-frame_idx))
qs_frame = b12_scan[frame_idx].kout_streaks(0)
rec_basis = rec_basis_full.dot(rot_m.T)

q_tf = QIndexTF(data=qs_frame, num_ap=pupil_size)

q_tf(qindex_point(rec_basis))