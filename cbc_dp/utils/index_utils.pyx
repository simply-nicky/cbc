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

def model_source_lines(float_t[:, ::1] source, float_t[:, ::1] rec_vec, float_t[:, ::1] kin):
    """
    Return source lines coordinates for a diffraction streaks model
    source - source line origins
    rec_vec - reciprocal vectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    """
    cdef:
        int a = rec_vec.shape[0], ii = 0, jj, i, k
        float_t source_prd, coeff1, coeff2, alpha, betta, gamma, delta, sol_1, sol_2, prod_1, prod_2
        uint8_t[::1] mask = np.zeros(a, dtype=np.uint8)
        float_t[:, :, ::1] source_lines = np.empty((a, 2, 3), dtype=np.float64)
    for i in range(a):
        source_prd = source[i, 0] * rec_vec[i, 0] + source[i, 1] * rec_vec[i, 1] + source[i, 2] * rec_vec[i, 2]
        jj = 0
        for k in range(4):
            coeff1 = source_prd - kin[k % 2, k // 2] * rec_vec[i, k // 2]
            coeff2 = rec_vec[i, (3 - k) // 2]
            alpha = coeff2**2 + rec_vec[i, 2]**2
            betta = coeff2 * coeff1
            gamma = coeff1**2 - rec_vec[i, 2]**2 * (1 - kin[k % 2, k // 2]**2)
            delta = betta**2 - alpha * gamma
            sol_1 = (betta + sqrt(delta)) / alpha
            prod_1 = (sol_1 * rec_vec[i, (3 - k) // 2] +
                      kin[k % 2, k // 2] * rec_vec[i, k // 2] +
                      sqrt(1 - kin[k % 2, k // 2]**2 - sol_1**2) * rec_vec[i, 2]) - source_prd
            sol_2 = (betta - sqrt(delta)) / alpha
            prod_2 = (sol_2 * rec_vec[i, (3 - k) // 2] +
                      kin[k % 2, k // 2] * rec_vec[i, k // 2] +
                      sqrt(1 - kin[k % 2, k // 2]**2 - sol_2**2) * rec_vec[i, 2]) - source_prd
            if abs(prod_1) < 1e-11 and sol_1 < kin[1, (3 - k) // 2] and sol_1 > kin[0, (3 - k) // 2]:
                source_lines[ii, jj, k // 2] = kin[k % 2, k // 2]
                source_lines[ii, jj, (3 - k) // 2] = sol_1
                source_lines[ii, jj, 2] = sqrt(1 - kin[k % 2, k // 2]**2 - sol_1**2)
                jj += 1
            if delta > 0 and abs(prod_2) < 1e-11 and sol_2 < kin[1, (3 - k) // 2] and sol_2 > kin[0, (3 - k) // 2]:
                source_lines[ii, jj, k // 2] = kin[k % 2, k // 2]
                source_lines[ii, jj, (3 - k) // 2] = sol_2
                source_lines[ii, jj, 2] = sqrt(1 - kin[k % 2, k // 2]**2 - sol_2**2)
                jj += 1
            if jj == 2:
                mask[i] = 1; ii += 1
                break
    return np.asarray(source_lines[:ii]), np.asarray(mask).astype(bool)

cdef void kout_streak(float_t[:, ::1] streak, float_t[::1] pt0, float_t[:, ::1] kout) nogil:
    """
    Return outcoming wavevectors of a streak

    streak - detected diffraction streak at the detector [mm]
    pt = [x0, y0, z0] - sample position relative to the detector [mm]
    """
    cdef:
        int_t j
        float_t dx, dy, phi, theta
    for j in range(2):
        dx = streak[j, 0] - pt0[0]
        dy = streak[j, 1] - pt0[1]
        phi = atan2(dy, dx)
        theta = atan(sqrt(dx**2 + dy**2) / pt0[2])
        kout[j, 0] = sin(theta) * cos(phi)
        kout[j, 1] = sin(theta) * sin(phi)
        kout[j, 2] = cos(theta)

def kout_frame(float_t[:, :, ::1] streaks, float_t[::1] pt0):
    """
    Return outcoming wavevectors of a pattern
    
    streaks - detected diffraction streaks at the detector [mm]
    pt0 = [x0, y0, z0] - sample position relative to the detector [mm]
    """
    cdef:
        int_t a = streaks.shape[0], i
        float_t[:, :, ::1] kout = np.empty((a, 2, 3), dtype=np.float64)
    for i in range(a):
        kout_streak(streaks[i], pt0, kout[i])
    return np.asarray(kout)

def vot_vec_frame(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t[:, ::1] kin):
    """
    Return reciprocal lattice voting points of a frame

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig
        int_t h_ind, k_ind, l_ind, ind, h_size, k_size, l_size
        float_t na_x = (kin[1, 0] - kin[0, 0]) / 2, na_y = (kin[1, 1] - kin[0, 1]) / 2
        float_t na_z = max(na_x, na_y)**2 / 2
        float_t kin_x = (kin[1, 0] + kin[0, 0]) / 2, kin_y = (kin[1, 1] + kin[0, 1]) / 2
        float_t kin_z = sqrt(1 - kin_x**2 - kin_y**2)
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        float_t[:, :, ::1] vot_vec
    inverse_matrix(rec_basis, inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + na_z * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + na_z * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + na_z * inv_basis[2, 2])))
    vot_vec = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.float64)
    for i in range(a):
        h_orig = int(floor((kout_exp[i, 0] - kin_x) * inv_basis[0, 0] +
                           (kout_exp[i, 1] - kin_y) * inv_basis[1, 0] +
                           (kout_exp[i, 2] - kin_z) * inv_basis[2, 0]))
        k_orig = int(floor((kout_exp[i, 0] - kin_x) * inv_basis[0, 1] +
                           (kout_exp[i, 1] - kin_y) * inv_basis[1, 1] +
                           (kout_exp[i, 2] - kin_z) * inv_basis[2, 1]))
        l_orig = int(floor((kout_exp[i, 0] - kin_x) * inv_basis[0, 2] +
                           (kout_exp[i, 1] - kin_y) * inv_basis[1, 2] +
                           (kout_exp[i, 2] - kin_z) * inv_basis[2, 2]))
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

def vot_idxs_frame(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t[:, ::1] kin):
    """
    Return voting points hkl indices of a frame

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    na_x, na_y - x- and y-coordinates of the incoming beam numerical aperture 
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig
        int_t h_ind, k_ind, l_ind, ind, h_size, k_size, l_size
        float_t na_x = (kin[1, 0] - kin[0, 0]) / 2, na_y = (kin[1, 1] - kin[0, 1]) / 2
        float_t na_z = max(na_x, na_y)**2 / 2
        float_t kin_x = (kin[1, 0] + kin[0, 0]) / 2, kin_y = (kin[1, 1] + kin[0, 1]) / 2
        float_t kin_z = sqrt(1 - kin_x**2 - kin_y**2)
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        int_t[:, :, ::1] vot_idxs
    inverse_matrix(rec_basis, inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + na_z * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + na_z * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + na_z * inv_basis[2, 2])))
    vot_idxs = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.int64)
    for i in range(a):
        h_orig = int(floor((kout_exp[i, 0] - kin_x) * inv_basis[0, 0] +
                           (kout_exp[i, 1] - kin_y) * inv_basis[1, 0] +
                           (kout_exp[i, 2] - kin_z) * inv_basis[2, 0]))
        k_orig = int(floor((kout_exp[i, 0] - kin_x) * inv_basis[0, 1] +
                           (kout_exp[i, 1] - kin_y) * inv_basis[1, 1] +
                           (kout_exp[i, 2] - kin_z) * inv_basis[2, 1]))
        l_orig = int(floor((kout_exp[i, 0] - kin_x) * inv_basis[0, 2] +
                           (kout_exp[i, 1] - kin_y) * inv_basis[1, 2] +
                           (kout_exp[i, 2] - kin_z) * inv_basis[2, 2]))
        for ii in range(2 * h_size):
            for jj in range(2 * k_size):
                for kk in range(2 * l_size):
                    ind = 4 * k_size * l_size * ii + 2 * l_size * jj + kk
                    vot_idxs[i, ind, 0] = h_orig + ii - h_size + 1
                    vot_idxs[i, ind, 1] = k_orig + jj - k_size + 1
                    vot_idxs[i, ind, 2] = l_orig + kk - l_size + 1
    return np.array(vot_idxs)

cdef float_t frame_distance(float_t x, float_t y, float_t[:, ::1] frame) nogil:
    """
    Return distance to a frame for a point lying outside the frame
    """
    cdef:
        float_t dx = 0, dy = 0
    if x > frame[1, 0]:
        dx += (x - frame[1, 0])
    if y > frame[1, 1]:
        dy += (y - frame[1, 1])
    if x < frame[0, 0]:
        dx += (frame[0, 0] - x)
    if y < frame[0, 1]:
        dy += (frame[0, 1] - y)
    return sqrt(dx**2 + dy**2)

cdef float_t fit_streak(float_t[::1] vot_vec, float_t[:, ::1] kout_exp, float_t[:, ::1] kin,
                        float_t tau_x, float_t tau_y, float_t pen_coeff):
    """
    Return fitness value for a streak

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    [tau_x, tau_y] - diffraction streak orientation vector
    pen_coeff - penalty coefficient
    """
    cdef:
        float_t rec_abs, source_th, source_phi, source_x, source_y
        float_t dk_x, dk_y, fit_x, fit_y, fit, d0, d1
    rec_abs = sqrt(vot_vec[0]**2 + vot_vec[1]**2 + vot_vec[2]**2)
    if rec_abs != 0:
        source_th = acos(-vot_vec[2] / rec_abs) - acos(rec_abs / 2)
        source_phi = atan2(vot_vec[1], vot_vec[0])
        source_x = vot_vec[0] - sin(source_th) * cos(source_phi)
        source_y = vot_vec[1] - sin(source_th) * sin(source_phi)
        dk_x = source_x - (kout_exp[0, 0] + kout_exp[1, 0]) / 2
        dk_y = source_y - (kout_exp[0, 1] + kout_exp[1, 1]) / 2
        fit_x = (dk_x * tau_y**2 - dk_y * tau_y * tau_x) / (tau_x**2 + tau_y**2)
        fit_y = (dk_y * tau_y**2 - dk_x * tau_x * tau_y) / (tau_x**2 + tau_y**2)
        fit = sqrt(fit_x**2 + fit_y**2)
    else:
        fit = 0

    d0 = frame_distance(kout_exp[0, 0] - vot_vec[0], kout_exp[0, 1] - vot_vec[1], kin)
    d1 = frame_distance(kout_exp[1, 0] - vot_vec[0], kout_exp[1, 1] - vot_vec[1], kin)
    fit += pen_coeff * (d0 + d1)
    return fit

def fitness(float_t[:, :, ::1] vot_vec, float_t[:, :, ::1] kout_exp, float_t[:, ::1] kin, float_t pen_coeff):
    """
    Return fitness value for a frame

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    pen_coeff - penalty coefficient
    """
    cdef:
        int a = vot_vec.shape[0], b = vot_vec.shape[1], i, j
        float_t fitness = 0.0, min_fit, pt_fit, tau_x, tau_y
    for i in range(a):
        tau_x = kout_exp[i, 1, 0] - kout_exp[i, 0, 0]
        tau_y = kout_exp[i, 1, 1] - kout_exp[i, 0, 1]
        min_fit = fit_streak(vot_vec[i, 0], kout_exp[i], kin, tau_x, tau_y, pen_coeff)
        for j in range(1, b):
            pt_fit = fit_streak(vot_vec[i, j], kout_exp[i], kin, tau_x, tau_y, pen_coeff)
            if pt_fit < min_fit:
                min_fit = pt_fit
        fitness += min_fit
    return fitness / a

def fitness_idxs(float_t[:, :, ::1] vot_vec, float_t[:, :, ::1] kout_exp, float_t[:, ::1] kin, float_t pen_coeff):
    """
    Return indices of the best lattice vectors of a frame

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    pen_coeff - penalty coefficient
    """
    cdef:
        int a = vot_vec.shape[0], b = vot_vec.shape[1], i, j
        float_t min_fit, pt_fit, tau_x, tau_y
        int_t[::1] idxs = np.empty(a, dtype=np.int64)
    for i in range(a):
        tau_x = kout_exp[i, 1, 0] - kout_exp[i, 0, 0]
        tau_y = kout_exp[i, 1, 1] - kout_exp[i, 0, 1]
        min_fit = fit_streak(vot_vec[i, 0], kout_exp[i], kin, tau_x, tau_y, pen_coeff)
        idxs[i] = 0
        for j in range(1, b):
            pt_fit = fit_streak(vot_vec[i, j], kout_exp[i], kin, tau_x, tau_y, pen_coeff)
            if pt_fit < min_fit:
                min_fit = pt_fit; idxs[i] = j
    return (np.arange(a), np.asarray(idxs))

def reduce_streaks(float_t[:, :, ::1] kout_exp, int_t[:, ::1] hkl_idxs, float_t[:, ::1] rec_basis,
                   float_t na_x, float_t na_y, float_t na_ext_x, float_t na_ext_y, float_t pen_coeff):
    """
    Exclude multiple streaks in a frame with the same hkl index

    kout_exp - experimental outcoming wavevectors
    hkl_idxs - the hkl indices of the diffraction reflections
    rec_basis - reciprocal lattice basis vectors
    na_x, na_y - x- and y-coordinates of the incoming beam numerical aperture
    na_ext_x, na_ext_y - upper bounds of the incoming beam numerical aperture
    pen_coeff - penalty coefficient
    """
    cdef:
        int_t a = kout_exp.shape[0], ii = 0, i, j
        float_t tau_x, tau_y, q_x, q_y, q_z, q_abs, source_x, source_y, source_th, source_phi
        float_t dk_x, dk_y, fit_x, fit_y, fit, kin_x, kin_y, new_fit
        int_t[::1] idxs_arr = np.empty(a, dtype=np.int64)
        uint8_t[::1] mask = np.zeros(a, dtype=np.uint8)
    for i in range(a):
        if not mask[i]:
            mask[i] = 1
            tau_x = kout_exp[i, 1, 0] - kout_exp[i, 0, 0]
            tau_y = kout_exp[i, 1, 1] - kout_exp[i, 0, 1]
            q_x = hkl_idxs[i, 0] * rec_basis[0, 0] + hkl_idxs[i, 1] * rec_basis[1, 0] + hkl_idxs[i, 2] * rec_basis[2, 0]
            q_y = hkl_idxs[i, 0] * rec_basis[0, 1] + hkl_idxs[i, 1] * rec_basis[1, 1] + hkl_idxs[i, 2] * rec_basis[2, 1]
            q_z = hkl_idxs[i, 0] * rec_basis[0, 2] + hkl_idxs[i, 1] * rec_basis[1, 2] + hkl_idxs[i, 2] * rec_basis[2, 2]
            kin_x = max(abs(q_x - kout_exp[i, 0, 0]), abs(q_x - kout_exp[i, 1, 0]))
            kin_y = max(abs(q_y - kout_exp[i, 0, 1]), abs(q_y - kout_exp[i, 1, 1]))
            if kin_x < na_ext_x and kin_y < na_ext_y:
                idxs_arr[ii] = i
                q_abs = sqrt(q_x**2 + q_y**2 + q_z**2)
                source_th = acos(-q_z / q_abs) - acos(q_abs / 2)
                source_phi = atan2(q_y, q_x)
                source_x = q_x - sin(source_th) * cos(source_phi)
                source_y = q_y - sin(source_th) * sin(source_phi)
                dk_x = source_x - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2
                dk_y = source_y - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2
                fit_x = (dk_x * tau_y**2 - dk_y * tau_y * tau_x) / (tau_x**2 + tau_y**2)
                fit_y = (dk_y * tau_y**2 - dk_x * tau_x * tau_y) / (tau_x**2 + tau_y**2)
                fit = sqrt(fit_x**2 + fit_y**2)
                if kin_x > na_x:
                    fit += pen_coeff * (kin_x - na_x)
                if kin_y > na_y:
                    fit += pen_coeff * (kin_y - na_y)
                for j in range(a):
                    if j != i and hkl_idxs[j, 0] == hkl_idxs[i, 0] and hkl_idxs[j, 1] == hkl_idxs[i, 1] and hkl_idxs[j, 2] == hkl_idxs[i, 2]:
                        mask[j] = 1
                        kin_x = max(abs(q_x - kout_exp[j, 0, 0]), abs(q_x - kout_exp[j, 1, 0]))
                        kin_y = max(abs(q_y - kout_exp[j, 0, 1]), abs(q_y - kout_exp[j, 1, 1]))
                        if kin_x < na_ext_x and kin_y < na_ext_y:
                            tau_x = kout_exp[j, 1, 0] - kout_exp[j, 0, 0]
                            tau_y = kout_exp[j, 1, 1] - kout_exp[j, 0, 1]
                            dk_x = source_x - (kout_exp[i, 0, 0] + kout_exp[i, 1, 0]) / 2
                            dk_y = source_y - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2
                            fit_x = (dk_x * tau_y**2 - dk_y * tau_y * tau_x) / (tau_x**2 + tau_y**2)
                            fit_y = (dk_y * tau_y**2 - dk_x * tau_x * tau_y) / (tau_x**2 + tau_y**2)
                            new_fit = sqrt(fit_x**2 + fit_y**2)
                            if kin_x > na_x:
                                new_fit += pen_coeff * (kin_x - na_x)
                            if kin_y > na_y:
                                new_fit += pen_coeff * (kin_y - na_y)
                            if new_fit < fit:
                                idxs_arr[ii] = j; fit = new_fit
                ii += 1
    return np.asarray(idxs_arr[:ii])