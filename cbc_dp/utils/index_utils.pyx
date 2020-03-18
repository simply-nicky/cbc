"""
index_utils.pyx - indexing and modelling utility functions written in Cython
"""
from libc.math cimport sqrt, sin, cos, tan, atan, atan2, acos, floor, ceil
import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint64_t uint_t
ctypedef cnp.uint8_t uint8_t

cdef void rotate_matrix(float_t[:, ::1] mat, float_t[:, ::1] output,
                        float_t alpha, float_t betta, float_t theta) nogil:
    """
    Rotate 3x3 matrix mat around axis by angle theta and write it to output matrix

    mat - matrix to rotate
    output - rotated matrix
    alpha, betta - spherical angles of axis of rotation
    theta - angle of rotation
    """
    cdef:
        float_t a = cos(theta / 2), b = -sin(alpha) * cos(betta) * sin(theta / 2)
        float_t c = -sin(alpha) * sin(betta) * sin(theta / 2), d = -cos(alpha) * sin(theta / 2)
        float_t rot_mat[3][3]
    rot_mat[0][0] = a * a + b * b - c * c - d * d
    rot_mat[0][1] = 2 * (b * c + a * d)
    rot_mat[0][2] = 2 * (b * d - a * c)
    rot_mat[1][0] = 2 * (b * c - a * d)
    rot_mat[1][1] = a * a + c * c - b * b - d * d
    rot_mat[1][2] = 2 * (c * d + a * b)
    rot_mat[2][0] = 2 * (b * d + a * c)
    rot_mat[2][1] = 2 * (c * d - a * b)
    rot_mat[2][2] = a * a + d * d - b * b - c * c
    output[0, 0] = rot_mat[0][0] * mat[0, 0] + rot_mat[0][1] * mat[0, 1] + rot_mat[0][2] * mat[0, 2]
    output[0, 1] = rot_mat[1][0] * mat[0, 0] + rot_mat[1][1] * mat[0, 1] + rot_mat[1][2] * mat[0, 2]
    output[0, 2] = rot_mat[2][0] * mat[0, 0] + rot_mat[2][1] * mat[0, 1] + rot_mat[2][2] * mat[0, 2]
    output[1, 0] = rot_mat[0][0] * mat[1, 0] + rot_mat[0][1] * mat[1, 1] + rot_mat[0][2] * mat[1, 2]
    output[1, 1] = rot_mat[1][0] * mat[1, 0] + rot_mat[1][1] * mat[1, 1] + rot_mat[1][2] * mat[1, 2]
    output[1, 2] = rot_mat[2][0] * mat[1, 0] + rot_mat[2][1] * mat[1, 1] + rot_mat[2][2] * mat[1, 2]
    output[2, 0] = rot_mat[0][0] * mat[2, 0] + rot_mat[0][1] * mat[2, 1] + rot_mat[0][2] * mat[2, 2]
    output[2, 1] = rot_mat[1][0] * mat[2, 0] + rot_mat[1][1] * mat[2, 1] + rot_mat[1][2] * mat[2, 2]
    output[2, 2] = rot_mat[2][0] * mat[2, 0] + rot_mat[2][1] * mat[2, 1] + rot_mat[2][2] * mat[2, 2]

cdef void inverse_matrix(float_t[:, ::1] mat, float_t[:, ::1] output) nogil:
    """
    Inverse 3x3 matrix mat and write it to output matrix

    mat - matrix to inverse
    output - inversed matrix
    """
    cdef:
        float_t det = (mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]) -
                       mat[0, 1] * (mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0]) +
                       mat[0, 2] * (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0]))
    output[0, 0] = (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]) / det
    output[1, 0] = -(mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0]) / det
    output[2, 0] = (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0]) / det
    output[0, 1] = -(mat[0, 1] * mat[2, 2] - mat[0, 2] * mat[2, 1]) / det
    output[1, 1] = (mat[0, 0] * mat[2, 2] - mat[0, 2] * mat[2, 0]) / det
    output[2, 1] = -(mat[0, 0] * mat[2, 1] - mat[0, 1] * mat[2, 0]) / det
    output[0, 2] = (mat[0, 1] * mat[1, 2] - mat[0, 2] * mat[1, 1]) / det
    output[1, 2] = -(mat[0, 0] * mat[1, 2] - mat[0, 2] * mat[1, 0]) / det
    output[2, 2] = (mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]) / det

def init_source(float_t[:, ::1] rec_vec):
    """
    Return reciprocal vector angles and source line origin points for an BallLattice class object

    rec_vec - array of reciprocal vectors
    """
    cdef:
        int a = rec_vec.shape[0], i
        float_t source_th
        float_t[::1] rec_abs = np.empty(a, dtype=np.float64)
        float_t[::1] rec_th = np.empty(a, dtype=np.float64)
        float_t[::1] rec_phi = np.empty(a, dtype=np.float64)
        float_t[:, ::1] source = np.empty((a, 3), dtype=np.float64)
    for i in range(a):
        rec_abs[i] = sqrt(rec_vec[i, 0]**2 + rec_vec[i, 1]**2 + rec_vec[i, 2]**2)
        rec_th[i] = acos(-rec_vec[i, 2] / rec_abs[i])
        rec_phi[i] = atan2(rec_vec[i, 1], rec_vec[i, 0])
        source_th = rec_th[i] - acos(rec_abs[i] / 2)
        source[i, 0] = -sin(source_th) * cos(rec_phi[i])
        source[i, 1] = -sin(source_th) * sin(rec_phi[i])
        source[i, 2] =  cos(source_th)
    return np.asarray(rec_abs), np.asarray(rec_th), np.asarray(rec_phi), np.asarray(source)

def model_source_lines(float_t[:, ::1] source, float_t[:, ::1] rec_vec, float_t na_x, float_t na_y):
    """
    Return source lines coordinates for a diffraction streaks model

    source - source line origins
    rec_vec - reciprocal vectors
    na_x, na_y - numerical apertires in x- and y-axes
    """
    cdef:
        int a = rec_vec.shape[0], ii = 0, jj, i, k
        uint8_t[::1] mask = np.zeros(a, dtype=np.uint8)
        float_t source_prd, coeff1, coeff2, alpha, betta, gamma, delta, sol_1, sol_2, prod_1, prod_2
        float_t[::1] bounds = np.array([na_x, -na_x, na_y, -na_y], dtype=np.float64)
        float_t[:, :, ::1] source_lines = np.empty((a, 2, 3), dtype=np.float64)
    for i in range(a):
        source_prd = source[i, 0] * rec_vec[i, 0] + source[i, 1] * rec_vec[i, 1] + source[i, 2] * rec_vec[i, 2]
        jj = 0
        for k in range(4):
            coeff1 = source_prd - bounds[k] * rec_vec[i, k // 2]
            coeff2 = rec_vec[i, (3 - k) // 2]
            alpha = coeff2**2 + rec_vec[i, 2]**2
            betta = coeff2 * coeff1
            gamma = coeff1**2 - rec_vec[i, 2]**2 * (1 - bounds[k]**2)
            delta = betta**2 - alpha * gamma
            sol_1 = (betta + sqrt(delta)) / alpha
            prod_1 = (sol_1 * rec_vec[i, (3 - k) // 2] +
                      bounds[k] * rec_vec[i, k // 2] +
                      sqrt(1 - bounds[k]**2 - sol_1**2) * rec_vec[i, 2]) - source_prd
            sol_2 = (betta - sqrt(delta)) / alpha
            prod_2 = (sol_2 * rec_vec[i, (3 - k) // 2] +
                      bounds[k] * rec_vec[i, k // 2] +
                      sqrt(1 - bounds[k]**2 - sol_2**2) * rec_vec[i, 2]) - source_prd
            if abs(prod_1) < 1e-11 and abs(sol_1) < abs(bounds[3 - k]):
                source_lines[ii, jj, k // 2] = bounds[k]
                source_lines[ii, jj, (3 - k) // 2] = sol_1
                source_lines[ii, jj, 2] = sqrt(1 - bounds[k]**2 - sol_1**2)
                jj += 1
            if delta > 0 and abs(prod_2) < 1e-11 and abs(sol_2) < abs(bounds[3 - k]):
                source_lines[ii, jj, k // 2] = bounds[k]
                source_lines[ii, jj, (3 - k) // 2] = sol_2
                source_lines[ii, jj, 2] = sqrt(1 - bounds[k]**2 - sol_2**2)
                jj += 1
            if jj == 2:
                mask[i] = 1; ii += 1
                break
    return np.asarray(source_lines[:ii]), np.asarray(mask).astype(bool)

def kout_frame(float_t[:, :, ::1] lines, float_t x0, float_t y0, float_t z0):
    """
    Return outcoming wavevectors of a pattern
    
    lines - detected diffraction streaks lines at the detector [mm]
    x0, y0, z0 - sample position relative to the detector [mm]
    """
    cdef:
        int_t a = lines.shape[0], i, j
        float_t[:, :, ::1] kout = np.empty((a, 2, 3), dtype=np.float64)
        float_t x, y, phi, theta
    for i in range(a):
        for j in range(2):
            dx = lines[i, j, 0] - x0
            dy = lines[i, j, 1] - y0
            phi = atan2(dy, dx)
            theta = atan(sqrt(dx**2 + dy**2) / z0)
            kout[i, j, 0] = sin(theta) * cos(phi)
            kout[i, j, 1] = sin(theta) * sin(phi)
            kout[i, j, 2] = cos(theta)
    return np.asarray(kout)

def kout_scan(float_t[:, :, ::1] lines, int_t[::1] frame_idxs, float_t[::1] x0,
              float_t[::1] y0, float_t[::1] z0):
    """
    Return outcoming wavevectors of a scan
    
    lines - detected diffraction streaks lines at the detector [mm]
    frame_idxs - frame indices
    x0, y0, z0 - sample position relative to the detector [mm]
    """
    cdef:
        int_t a = lines.shape[0], i, idx
        float_t dx, dy, phi, theta
        float_t[:, :, ::1] kout = np.empty((a, 2, 3), dtype=np.float64)
    for i in range(a):
        idx = frame_idxs[i]
        for j in range(2):
            dx = lines[i, j, 0] - x0[idx]
            dy = lines[i, j, 1] - y0[idx]
            phi = atan2(dy, dx)
            theta = atan(sqrt(dx**2 + dy**2) / z0[idx])
            kout[i, j, 0] = sin(theta) * cos(phi)
            kout[i, j, 1] = sin(theta) * sin(phi)
            kout[i, j, 2] = cos(theta)
    return np.asarray(kout)

def voting_vectors_f(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t na_x, float_t na_y):
    """
    Return reciprocal lattice voting points of a pattern

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    na_x, na_y - numerical apertires in x- and y-axes
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig, h_ind, k_ind, l_ind, ind, h_size, k_size, l_size
        float_t max_na = max(na_x, na_y)**2 / 2, rec_x, rec_y, rec_z, rec_abs, source_th, source_phi
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        float_t[:, :, ::1] vot_vec
    inverse_matrix(rec_basis, inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + max_na * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + max_na * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + max_na * inv_basis[2, 2])))
    vot_vec = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.float64)
    for i in range(a):
        h_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 0] +
                      kout_exp[i, 1] * inv_basis[1, 0] +
                      (kout_exp[i, 2] - 1) * inv_basis[2, 0]))
        k_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 1] +
                      kout_exp[i, 1] * inv_basis[1, 1] +
                      (kout_exp[i, 2] - 1) * inv_basis[2, 1]))
        l_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 2] +
                      kout_exp[i, 1] * inv_basis[1, 2] +
                      (kout_exp[i, 2] - 1) * inv_basis[2, 2]))
        for ii in range(2 * h_size):
            for jj in range(2 * k_size):
                for kk in range(2 * l_size):
                    h_ind = h_orig + ii - h_size + 1
                    k_ind = k_orig + jj - k_size + 1
                    l_ind = l_orig + kk - l_size + 1
                    ind = 4 * k_size * l_size * ii + 2 * l_size * jj + kk
                    vot_vec[i, ind, 0] = h_ind * rec_basis[0, 0] + k_ind * rec_basis[1, 0] + l_ind * rec_basis[2, 0]
                    vot_vec[i, ind, 1] = h_ind * rec_basis[0, 1] + k_ind * rec_basis[1, 1] + l_ind * rec_basis[2, 1]
                    vot_vec[i, ind, 2] = h_ind * rec_basis[0, 2] + k_ind * rec_basis[1, 2] + l_ind * rec_basis[2, 2]
    return np.array(vot_vec)

def voting_idxs_f(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t na_x, float_t na_y):
    """
    Return voting points hkl indices of a pattern

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    na_x, na_y - numerical apertires in x- and y-axes
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig, h_ind, k_ind, l_ind, ind, h_size, k_size, l_size
        float_t max_na = max(na_x, na_y)**2 / 2, rec_x, rec_y, rec_z, rec_abs, source_th, source_phi
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        int_t[:, :, ::1] vot_idxs
    inverse_matrix(rec_basis, inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + max_na * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + max_na * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + max_na * inv_basis[2, 2])))
    vot_idxs = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.int64)
    for i in range(a):
        h_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 0] +
                      kout_exp[i, 1] * inv_basis[1, 0] +
                      (kout_exp[i, 2] - 1) * inv_basis[2, 0]))
        k_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 1] +
                      kout_exp[i, 1] * inv_basis[1, 1] +
                      (kout_exp[i, 2] - 1) * inv_basis[2, 1]))
        l_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 2] +
                      kout_exp[i, 1] * inv_basis[1, 2] +
                      (kout_exp[i, 2] - 1) * inv_basis[2, 2]))
        for ii in range(2 * h_size):
            for jj in range(2 * k_size):
                for kk in range(2 * l_size):
                    ind = 4 * k_size * l_size * ii + 2 * l_size * jj + kk
                    vot_idxs[i, ind, 0] = h_orig + ii - h_size + 1
                    vot_idxs[i, ind, 1] = k_orig + jj - k_size + 1
                    vot_idxs[i, ind, 2] = l_orig + kk - l_size + 1
    return np.array(vot_idxs)

def voting_vectors_s(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t[::1] thetas,
                     int_t[::1] frame_idxs, float_t alpha, float_t betta, float_t na_x, float_t na_y):
    """
    Return reciprocal lattice voting points of a scan

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    thetas - tilt angles
    frame_idxs - frame indices
    alpha, betta - spherical angles of axis of rotation
    na_x, na_y - numerical apertires in x- and y-axes
    """
    cdef:
        int_t a = kout_exp.shape[0], idx = frame_idxs[0], i, ii, jj, kk
        int_t h_orig, k_orig, l_orig, h_ind, k_ind, l_ind, ind, h_size, k_size, l_size
        float_t max_na = max(na_x, na_y)**2 / 2, rec_x, rec_y, rec_z, rec_abs, source_th, source_phi
        float_t[:, ::1] frame_basis = np.empty((3, 3), dtype=np.float64)
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        float_t[:, :, :] vot_vec
    rotate_matrix(rec_basis, frame_basis, alpha, betta, thetas[idx])
    inverse_matrix(frame_basis, inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + max_na * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + max_na * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + max_na * inv_basis[2, 2])))
    vot_vec = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.float64)
    for i in range(a):
        if idx != frame_idxs[i]:
            idx = frame_idxs[i]
            rotate_matrix(rec_basis, frame_basis, alpha, betta, thetas[idx])
            inverse_matrix(frame_basis, inv_basis)
        h_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 0] +
                           kout_exp[i, 1] * inv_basis[1, 0] +
                           (kout_exp[i, 2] - 1) * inv_basis[2, 0]))
        k_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 1] +
                           kout_exp[i, 1] * inv_basis[1, 1] +
                           (kout_exp[i, 2] - 1) * inv_basis[2, 1]))
        l_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 2] +
                           kout_exp[i, 1] * inv_basis[1, 2] +
                           (kout_exp[i, 2] - 1) * inv_basis[2, 2]))
        for ii in range(2 * h_size):
            for jj in range(2 * k_size):
                for kk in range(2 * l_size):
                    ind = 4 * k_size * l_size * ii + 2 * l_size * jj + kk
                    h_ind = h_orig + ii - h_size + 1
                    k_ind = k_orig + jj - k_size + 1
                    l_ind = l_orig + kk - l_size + 1
                    vot_vec[i, ind, 0] = h_ind * frame_basis[0, 0] + k_ind * frame_basis[1, 0] + l_ind * frame_basis[2, 0]
                    vot_vec[i, ind, 1] = h_ind * frame_basis[0, 1] + k_ind * frame_basis[1, 1] + l_ind * frame_basis[2, 1]
                    vot_vec[i, ind, 2] = h_ind * frame_basis[0, 2] + k_ind * frame_basis[1, 2] + l_ind * frame_basis[2, 2]
    return np.asarray(vot_vec)

def voting_idxs_s(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t[::1] thetas,
                int_t[::1] frame_idxs, float_t alpha, float_t betta, float_t na_x, float_t na_y):
    """
    Return voting points hkl indices of a scan

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    thetas - tilt angles
    frame_idxs - frame indices
    alpha, betta - spherical angles of axis of rotation
    na_x, na_y - numerical apertires in x- and y-axes
    """
    cdef:
        int_t a = kout_exp.shape[0], idx = frame_idxs[0], i, ii, jj, kk
        int_t h_orig, k_orig, l_orig, h_ind, k_ind, l_ind, ind, h_size, k_size, l_size
        float_t  max_na = max(na_x, na_y)**2 / 2, rec_x, rec_y, rec_z, rec_abs, source_th, source_phi
        float_t[:, ::1] frame_basis = np.empty((3, 3), dtype=np.float64)
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        int_t[:, :, ::1] vot_idxs
    rotate_matrix(rec_basis, frame_basis, alpha, betta, thetas[idx])
    inverse_matrix(frame_basis, inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + max_na * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + max_na * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + max_na * inv_basis[2, 2])))
    vot_idxs = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.int64)
    for i in range(a):
        if idx != frame_idxs[i]:
            idx = frame_idxs[i]
            rotate_matrix(rec_basis, frame_basis, alpha, betta, thetas[idx])
            inverse_matrix(frame_basis, inv_basis)
        h_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 0] +
                           kout_exp[i, 1] * inv_basis[1, 0] +
                           (kout_exp[i, 2] - 1) * inv_basis[2, 0]))
        k_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 1] +
                           kout_exp[i, 1] * inv_basis[1, 1] +
                           (kout_exp[i, 2] - 1) * inv_basis[2, 1]))
        l_orig = int(floor(kout_exp[i, 0] * inv_basis[0, 2] +
                           kout_exp[i, 1] * inv_basis[1, 2] +
                           (kout_exp[i, 2] - 1) * inv_basis[2, 2]))
        for ii in range(2 * h_size):
            for jj in range(2 * k_size):
                for kk in range(2 * l_size):
                    ind = 4 * k_size * l_size * ii + 2 * l_size * jj + kk
                    vot_idxs[i, ind, 0] = h_orig + ii - h_size + 1
                    vot_idxs[i, ind, 1] = k_orig + jj - k_size + 1
                    vot_idxs[i, ind, 2] = l_orig + kk - l_size + 1
    return np.array(vot_idxs)

def fitness(float_t[:, :, ::1] vot_vec, float_t[:, :, ::1] kout_exp, float_t na_x, float_t na_y, float_t pen_coeff):
    """
    Return target function fitness value for a given point with the penalty added

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    num_ap - convergent beam numerical aperture
    pen_coeff - penalty coefficient
    """
    cdef:
        int a = vot_vec.shape[0], b = vot_vec.shape[1], i, j
        float_t rec_abs, source_th, source_phi, source_x, source_y
        float_t tau_x, tau_y, dk_x, dk_y, fit_x, fit_y, dist_x, dist_y
        float_t fitness = 0.0, min_fit, pt_fit
    for i in range(a):
        tau_x = kout_exp[i, 1, 0] - kout_exp[i, 0, 0]
        tau_y = kout_exp[i, 1, 1] - kout_exp[i, 0, 1]
        rec_abs = sqrt(vot_vec[i, 0, 0]**2 + vot_vec[i, 0, 1]**2 + vot_vec[i, 0, 2]**2)
        if rec_abs != 0:
            source_th = acos(-vot_vec[i, 0, 2] / rec_abs) - acos(rec_abs / 2)
            source_phi = atan2(vot_vec[i, 0, 1], vot_vec[i, 0, 0])
            source_x = vot_vec[i, 0, 0] - sin(source_th) * cos(source_phi)
            source_y = vot_vec[i, 0, 1] - sin(source_th) * sin(source_phi)
            dk_x = source_x - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2
            dk_y = source_y - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2
            fit_x = (dk_x * tau_y**2 - dk_y * tau_y * tau_x) / (tau_x**2 + tau_y**2)
            fit_y = (dk_y * tau_y**2 - dk_x * tau_x * tau_y) / (tau_x**2 + tau_y**2)
            min_fit = sqrt(fit_x**2 + fit_y**2)
        else:
            min_fit = 0
        dist_x = abs(vot_vec[i, 0, 0] - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2)
        if dist_x > na_x:
            min_fit += pen_coeff * (dist_x - na_x)
        dist_y = abs(vot_vec[i, 0, 1] - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2)
        if dist_y > na_y:
            min_fit += pen_coeff * (dist_y - na_y)
        for j in range(1, b):
            rec_abs = sqrt(vot_vec[i, j, 0]**2 + vot_vec[i, j, 1]**2 + vot_vec[i, j, 2]**2)
            if rec_abs != 0:
                source_th = acos(-vot_vec[i, j, 2] / rec_abs) - acos(rec_abs / 2)
                source_phi = atan2(vot_vec[i, j, 1], vot_vec[i, j, 0])
                source_x = vot_vec[i, j, 0] - sin(source_th) * cos(source_phi)
                source_y = vot_vec[i, j, 1] - sin(source_th) * sin(source_phi)
                dk_x = source_x - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2
                dk_y = source_y - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2
                fit_x = (dk_x * tau_y**2 - dk_y * tau_y * tau_x) / (tau_x**2 + tau_y**2)
                fit_y = (dk_y * tau_y**2 - dk_x * tau_x * tau_y) / (tau_x**2 + tau_y**2)
                pt_fit = sqrt(fit_x**2 + fit_y**2)
            else:
                pt_fit = 0
            dist_x = abs(vot_vec[i, j, 0] - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2)
            if dist_x > na_x:
                pt_fit += pen_coeff * (dist_x - na_x)
            dist_y = abs(vot_vec[i, j, 1] - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2)
            if dist_y > na_y:
                pt_fit += pen_coeff * (dist_y - na_y)
            if pt_fit < min_fit:
                min_fit = pt_fit
        fitness += min_fit
    return fitness / a

def fitness_idxs(float_t[:, :, ::1] vot_vec, float_t[:, :, ::1] kout_exp, float_t na_x, float_t na_y, float_t pen_coeff):
    """
    Return indices of the best lattice vectors based on fitness values with the penalty added

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    num_ap - convergent beam numerical aperture
    pen_coeff - penalty coefficient
    """
    cdef:
        int a = vot_vec.shape[0], b = vot_vec.shape[1], i, j
        float_t rec_abs, source_th, source_phi, source_x, source_y
        float_t tau_x, tau_y, dk_x, dk_y, fit_x, fit_y, dist_x, dist_y
        float_t min_fit, pt_fit
        int_t[::1] idxs = np.empty(a, dtype=np.int64)
    for i in range(a):
        tau_x = kout_exp[i, 1, 0] - kout_exp[i, 0, 0]
        tau_y = kout_exp[i, 1, 1] - kout_exp[i, 0, 1]
        rec_abs = sqrt(vot_vec[i, 0, 0]**2 + vot_vec[i, 0, 1]**2 + vot_vec[i, 0, 2]**2)
        if rec_abs != 0:
            source_th = acos(-vot_vec[i, 0, 2] / rec_abs) - acos(rec_abs / 2)
            source_phi = atan2(vot_vec[i, 0, 1], vot_vec[i, 0, 0])
            source_x = vot_vec[i, 0, 0] - sin(source_th) * cos(source_phi)
            source_y = vot_vec[i, 0, 1] - sin(source_th) * sin(source_phi)
            dk_x = source_x - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2
            dk_y = source_y - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2
            fit_x = (dk_x * tau_y**2 - dk_y * tau_y * tau_x) / (tau_x**2 + tau_y**2)
            fit_y = (dk_y * tau_y**2 - dk_x * tau_x * tau_y) / (tau_x**2 + tau_y**2)
            min_fit = sqrt(fit_x**2 + fit_y**2)
        else:
            min_fit = 0
        dist_x = abs(vot_vec[i, 0, 0] - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2)
        if dist_x > na_x:
            min_fit += pen_coeff * (dist_x - na_x)
        dist_y = abs(vot_vec[i, 0, 1] - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2)
        if dist_y > na_y:
            min_fit += pen_coeff * (dist_y - na_y)
        idxs[i] = 0
        for j in range(1, b):
            rec_abs = sqrt(vot_vec[i, j, 0]**2 + vot_vec[i, j, 1]**2 + vot_vec[i, j, 2]**2)
            if rec_abs != 0:
                source_th = acos(-vot_vec[i, j, 2] / rec_abs) - acos(rec_abs / 2)
                source_phi = atan2(vot_vec[i, j, 1], vot_vec[i, j, 0])
                source_x = vot_vec[i, j, 0] - sin(source_th) * cos(source_phi)
                source_y = vot_vec[i, j, 1] - sin(source_th) * sin(source_phi)
                dk_x = source_x - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2
                dk_y = source_y - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2
                fit_x = (dk_x * tau_y**2 - dk_y * tau_y * tau_x) / (tau_x**2 + tau_y**2)
                fit_y = (dk_y * tau_y**2 - dk_x * tau_x * tau_y) / (tau_x**2 + tau_y**2)
                pt_fit = sqrt(fit_x**2 + fit_y**2)
            else:
                pt_fit = 0
            dist_x = abs(vot_vec[i, j, 0] - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2)
            if dist_x > na_x:
                pt_fit += pen_coeff * (dist_x - na_x)
            dist_y = abs(vot_vec[i, j, 1] - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2)
            if dist_y > na_y:
                pt_fit += pen_coeff * (dist_y - na_y)
            if pt_fit < min_fit:
                min_fit = pt_fit; idxs[i] = j
    return (np.arange(a), np.asarray(idxs))