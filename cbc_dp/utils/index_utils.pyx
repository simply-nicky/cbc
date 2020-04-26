"""
index_utils.pyx - indexing and modelling utility functions written in Cython
"""
from libc.math cimport sqrt, sin, cos, tan, atan, atan2, acos, floor, ceil, pi
from scipy.special.cython_special cimport binom
import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint64_t uint_t
ctypedef cnp.uint8_t uint8_t

def scan_ps(int_t pop_size, int_t scan_size):
    cdef:
        int_t k = scan_size
        float_t ps1 = binom(k, scan_size - 1), ps2 = binom(k + 1, scan_size - 1)
    while abs(ps1 - pop_size) > abs(ps2 - pop_size):
        k += 1
        ps1 = ps2; ps2 = binom(k + 1, scan_size - 1)
    return int(ps1)

cdef void eul_ang_c(float_t[::1] output, float_t[:, ::1] rot_mat) nogil:
    output[1] = acos(rot_mat[2, 2])
    if output[1] < 1e-8:
        output[0] = atan2(-rot_mat[1, 0], rot_mat[0, 0])
        output[2] = 0
    elif pi - output[1] < (1e-8 + 1e-5 * pi):
        output[0] = atan2(rot_mat[1, 0], rot_mat[0, 0])
        output[2] = 0
    else:
        output[0] = atan2(rot_mat[2, 0], -rot_mat[2, 1])
        output[2] = atan2(rot_mat[0, 2], rot_mat[1, 2])

def euler_angles(float_t[:, ::1] rot_mat):
    """
    Return euler angles with Bunge convention from a rotation matrix
    """
    cdef float_t[::1] eul_ang = np.empty(3, dtype=np.float64)
    eul_ang_c(eul_ang, rot_mat)
    return np.asarray(eul_ang)

cdef void euler_mat_c(float_t[:, ::1] output, float_t phi1, float_t Phi, float_t phi2) nogil:
    """
    Write an Euler rotation matrix with Bunge convention to output matrix

    phi1, Phi, phi2 - Euler angles
    """
    output[0, 0] = cos(phi1) * cos(phi2) - sin(phi1) * sin(phi2) * cos(Phi)
    output[0, 1] = sin(phi1) * cos(phi2) + cos(phi1) * sin(phi2) * cos(Phi)
    output[0, 2] = sin(phi2) * sin(Phi)
    output[1, 0] = -cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(Phi)
    output[1, 1] = -sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(Phi)
    output[1, 2] = cos(phi2) * sin(Phi)
    output[2, 0] = sin(phi1) * sin(Phi)
    output[2, 1] = -cos(phi1) * sin(Phi)
    output[2, 2] = cos(Phi)

def euler_matrix(float_t phi1, float_t Phi, float_t phi2):
    """
    Return an Euler matrix with Bunge convention

    phi1, Phi, phi2 - Euler angles
    """
    cdef:
        float_t[:, ::1] rot_mat = np.empty((3, 3), dtype=np.float64)
    euler_mat_c(rot_mat, phi1, Phi, phi2)
    return np.asarray(rot_mat)

cdef void rot_mat_c(float_t[:, ::1] output, float_t alpha, float_t betta, float_t theta) nogil:
    """
    Write a rotation matrix to output matrix

    alpha, betta - spherical angles of axis of rotation
    theta - angle of rotation
    """
    cdef:
        float_t a = cos(theta / 2), b = -sin(alpha) * cos(betta) * sin(theta / 2)
        float_t c = -sin(alpha) * sin(betta) * sin(theta / 2), d = -cos(alpha) * sin(theta / 2)
    output[0, 0] = a * a + b * b - c * c - d * d
    output[0, 1] = 2 * (b * c + a * d)
    output[0, 2] = 2 * (b * d - a * c)
    output[1, 0] = 2 * (b * c - a * d)
    output[1, 1] = a * a + c * c - b * b - d * d
    output[1, 2] = 2 * (c * d + a * b)
    output[2, 0] = 2 * (b * d + a * c)
    output[2, 1] = 2 * (c * d - a * b)
    output[2, 2] = a * a + d * d - b * b - c * c

def scan_eul_ang(float_t[::1] axis, float_t[::1] thetas):
    """
    Return Euler angles of a tilt scan

    axis - axis of rotation
    thetas - angles of rotation
    """
    cdef:
        int_t a = thetas.shape[0], i
        float_t length = sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
        float_t alpha = acos(axis[2] / length), betta = atan2(axis[1], axis[0])
        float_t[:, ::1] rot_mat = np.empty((3, 3), dtype=np.float64)
        float_t[:, ::1] eul_ang = np.empty((a, 3), dtype=np.float64)
    for i in range(a):
        rot_mat_c(rot_mat, alpha, betta, thetas[i])
        eul_ang_c(eul_ang[i], rot_mat)
    return np.asarray(eul_ang)

def rotation_matrix(float_t[::1] axis, float_t theta):
    """
    Return a rotation matrix

    axis - rotational axis
    theta - angle of rotation
    """
    cdef:
        float_t[:, ::1] rot_mat = np.empty((3, 3), dtype=np.float64)
        float_t length = sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
        float_t alpha = acos(axis[2] / length), betta = atan2(axis[1], axis[0])
    rot_mat_c(rot_mat, alpha, betta, theta)
    return np.asarray(rot_mat)

cdef void rotate_vector(float_t[::1] vec, float_t[:, ::1] rot_mat, float_t[::1] output) nogil:
    """
    Rotate a 3d vector by the rotation matrix and write it to output vector
    """
    output[0] = rot_mat[0, 0] * vec[0] + rot_mat[0, 1] * vec[1] + rot_mat[0, 2] * vec[2]
    output[1] = rot_mat[1, 0] * vec[0] + rot_mat[1, 1] * vec[1] + rot_mat[1, 2] * vec[2]
    output[2] = rot_mat[2, 0] * vec[0] + rot_mat[2, 1] * vec[1] + rot_mat[2, 2] * vec[2]
    
cdef void rotate_matrix(float_t[:, ::1] mat, float_t[:, ::1] rot_mat, float_t[:, ::1] output) nogil:
    """
    Rotate a 3x3 matrix by the rotation matrix and write it to output vector
    """
    rotate_vector(mat[0], rot_mat, output[0])
    rotate_vector(mat[1], rot_mat, output[1])
    rotate_vector(mat[2], rot_mat, output[2])

def rcbi_rb(float_t[:, ::1] or_mat, float_t[::1] rb_sizes, float_t[::1] eul_ang):
    """
    Return reciprocal lattice basis vectors

    or_mat - orientation matrix
    rb_sizes - reciprocal lattice vector lengths
    eul_ang - orientation matrix Euler angles
    """
    cdef:
        float_t[:, ::1] rb = np.empty((3, 3), dtype=np.float64)
        float_t[:, ::1] rot_mat = np.empty((3, 3), dtype=np.float64)
    euler_mat_c(rot_mat, eul_ang[0], eul_ang[1], eul_ang[2])
    rotate_matrix(or_mat, rot_mat, rb)
    rb[0, 0] *= rb_sizes[0]; rb[0, 1] *= rb_sizes[0]; rb[0, 2] *= rb_sizes[0]
    rb[1, 0] *= rb_sizes[1]; rb[1, 1] *= rb_sizes[1]; rb[1, 2] *= rb_sizes[1]
    rb[2, 0] *= rb_sizes[2]; rb[2, 1] *= rb_sizes[2]; rb[2, 2] *= rb_sizes[2]
    return np.asarray(rb)

def fcbi_rb(float_t[:, ::1] or_mat, float_t[::1] rb_sizes, float_t[::1] eul_ang):
    """
    Return reciprocal lattice basis vectors

    rb_sizes - reciprocal lattice vector lengths
    eul_ang - basis vectors Euler angles
    """
    cdef:
        float_t[:, ::1] rb = np.empty((3, 3), dtype=np.float64)
        float_t[:, ::1] rot_mat = np.empty((3, 3), dtype=np.float64)
    euler_mat_c(rot_mat, eul_ang[0], eul_ang[1], eul_ang[2])
    rotate_vector(or_mat[0], rot_mat, rb[0])
    euler_mat_c(rot_mat, eul_ang[3], eul_ang[4], eul_ang[5])
    rotate_vector(or_mat[1], rot_mat, rb[1])
    euler_mat_c(rot_mat, eul_ang[6], eul_ang[7], eul_ang[8])
    rotate_vector(or_mat[2], rot_mat, rb[2])
    rb[0, 0] *= rb_sizes[0]; rb[0, 1] *= rb_sizes[0]; rb[0, 2] *= rb_sizes[0]
    rb[1, 0] *= rb_sizes[1]; rb[1, 1] *= rb_sizes[1]; rb[1, 2] *= rb_sizes[1]
    rb[2, 0] *= rb_sizes[2]; rb[2, 1] *= rb_sizes[2]; rb[2, 2] *= rb_sizes[2]
    return np.asarray(rb)

def scan_rb(float_t[::1] rb_mat, float_t[::1] eul_ang, int_t[::1] frames):
    """
    Return reciprocal lattice basis vectors

    rb_mat - reciprocal lattice basis vectors matrix
    eul_ang - Euler angles
    frames - frame indices
    """
    cdef:
        int_t a = frames.shape[0], i
        float_t[:, ::1] rot_mat = np.empty((3, 3), dtype=np.float64)
        float_t[:, ::1] rb0 = np.empty((3, 3), dtype=np.float64)
        float_t[:, :, ::1] rec_basis = np.empty((a, 3, 3), dtype=np.float64)
    rb0[0] = rb_mat[:3]; rb0[1] = rb_mat[3:6]; rb0[2] = rb_mat[6:9]
    for i in range(a):
        euler_mat_c(rot_mat, eul_ang[3 * i], eul_ang[3 * i + 1], eul_ang[3 * i + 2])
        rotate_matrix(rb0, rot_mat, rec_basis[i])
    return np.asarray(rec_basis)

cdef void inverse_mat_c(float_t[:, ::1] mat, float_t[:, ::1] output) nogil:
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

def inverse_matrix(float_t[:, ::1] mat):
    """
    Inverse 3x3 matrix

    mat - matrix to inverse
    """
    cdef float_t[:, ::1] inv_mat = np.empty((3, 3), dtype=np.float64)
    inverse_mat_c(mat, inv_mat)
    return np.asarray(inv_mat)

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

cdef void kin_pupil(float_t[:, ::1] pupil, float_t[::1] f_pos, float_t[:, ::1] output) nogil:
    """
    Return kin bounds

    pupil - pupil bounds at the detector plane [mm]
    f_pos - focus position relative to the detector [mm]
    """
    cdef:
        int_t j
        float_t dx, dy, phi, theta
    for j in range(2):
        dx = pupil[j, 0] - f_pos[0]
        dy = pupil[j, 1] - f_pos[1]
        phi = atan2(dy, dx)
        theta = atan(sqrt(dx**2 + dy**2) / f_pos[2])
        output[j, 0] = sin(theta) * cos(phi)
        output[j, 1] = sin(theta) * sin(phi)

def kin_frame(float_t[:, ::1] pupil, float_t[::1] f_pos):
    """
    Return kin bounds for a frame

    pupil - pupil bounds at the detector plane [mm]
    f_pos - focus position relative to the detector [mm]
    """
    cdef float_t[:, ::1] kin = np.empty((2, 2), dtype=np.float64)
    kin_pupil(pupil, f_pos, kin)
    return np.asarray(kin)

def kin_scan(float_t[:, :, ::1] pupil, float_t[::1] f_pos):
    """
    Return kin bounds for a scan

    pupil - pupil bounds at the detector plane [mm]
    f_pos - focus position relative to the detector [mm]
    """
    cdef:
        int_t a = pupil.shape[0], i
        float_t[:, :, ::1] kin = np.empty((a, 2, 2), dtype=np.float64)
    for i in range(a):
        kin_pupil(pupil[i], f_pos, kin[i])
    return np.asarray(kin)

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

def kout_scan(float_t[:, :, ::1] streaks, float_t[::1] pts0, int_t[::1] idxs):
    """
    Return outcoming wavevectors of a pattern
    
    streaks - detected diffraction streaks at the detector [mm]
    pts0 - sample positions relative to the detector [mm]
    frame_idxs - streaks' frame indices
    """
    cdef:
        int_t a = idxs.shape[0] - 1, aa = streaks.shape[0], i, ii
        float_t[:, :, ::1] kout = np.empty((aa, 2, 3), dtype=np.float64)
    for i in range(a):
        for ii in range(idxs[i], idxs[i + 1]):
            kout_streak(streaks[ii], pts0[3 * i:3 * (i + 1)], kout[ii])
    return np.asarray(kout)

cdef void vot_vec_c(float_t[:, :, ::1] output, float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis,
                    float_t[:, ::1] inv_basis, float_t[:, ::1] kin, int_t h_size, int_t k_size, int_t l_size) nogil:
    """
    Write voting vectors to an output array
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig, h_ind, k_ind, l_ind
        float_t kin_x = (kin[1, 0] + kin[0, 0]) / 2, kin_y = (kin[1, 1] + kin[0, 1]) / 2
        float_t kin_z = sqrt(1 - kin_x**2 - kin_y**2)
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
        for ii in range(2 * l_size):
            for jj in range(2 * k_size):
                for kk in range(2 * l_size):
                    h_ind = h_orig + ii - h_size + 1
                    k_ind = k_orig + jj - k_size + 1
                    l_ind = l_orig + kk - l_size + 1
                    ind = 4 * k_size * l_size * ii + 2 * l_size * jj + kk
                    output[i, ind, 0] = h_ind * rec_basis[0, 0] + k_ind * rec_basis[1, 0] + l_ind * rec_basis[2, 0]
                    output[i, ind, 1] = h_ind * rec_basis[0, 1] + k_ind * rec_basis[1, 1] + l_ind * rec_basis[2, 1]
                    output[i, ind, 2] = h_ind * rec_basis[0, 2] + k_ind * rec_basis[1, 2] + l_ind * rec_basis[2, 2]

cdef void vot_idxs_c(int_t[:, :, ::1] output, float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis,
                     float_t[:, ::1] inv_basis, float_t[:, ::1] kin, int_t h_size, int_t k_size, int_t l_size) nogil:
    """
    Write voting vectors' hkl indices to an output array
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig
        float_t kin_x = (kin[1, 0] + kin[0, 0]) / 2, kin_y = (kin[1, 1] + kin[0, 1]) / 2
        float_t kin_z = sqrt(1 - kin_x**2 - kin_y**2)
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
        for ii in range(2 * l_size):
            for jj in range(2 * k_size):
                for kk in range(2 * l_size):
                    ind = 4 * k_size * l_size * ii + 2 * l_size * jj + kk
                    output[i, ind, 0] = h_orig + ii - h_size + 1
                    output[i, ind, 1] = k_orig + jj - k_size + 1
                    output[i, ind, 2] = l_orig + kk - l_size + 1

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
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        float_t[:, :, ::1] vot_vec
    inverse_mat_c(rec_basis, inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + na_z * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + na_z * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + na_z * inv_basis[2, 2])))
    vot_vec = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.float64)
    vot_vec_c(vot_vec, kout_exp, rec_basis, inv_basis, kin, h_size, k_size, l_size)
    return np.asarray(vot_vec)

def vot_idxs_frame(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t[:, ::1] kin):
    """
    Return voting points hkl indices of a frame

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig
        int_t h_ind, k_ind, l_ind, ind, h_size, k_size, l_size
        float_t na_x = (kin[1, 0] - kin[0, 0]) / 2, na_y = (kin[1, 1] - kin[0, 1]) / 2
        float_t na_z = max(na_x, na_y)**2 / 2
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        int_t[:, :, ::1] vot_idxs
    inverse_mat_c(rec_basis, inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + na_z * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + na_z * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + na_z * inv_basis[2, 2])))
    vot_idxs = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.int64)
    vot_idxs_c(vot_idxs, kout_exp, rec_basis, inv_basis, kin, h_size, k_size, l_size)
    return np.asarray(vot_idxs)

def vot_vec_scan(float_t[:, ::1] kout_exp, float_t[:, :, ::1] rec_basis,
                 float_t[:, :, ::1] kin, int_t[::1] idxs):
    """
    Return voting points of a scan

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    idxs - streaks unique indices
    """
    cdef:
        int aa = kout_exp.shape[0], a = rec_basis.shape[0], i, h_size, k_size, l_size
        float_t na_x = (kin[0, 1, 0] - kin[0, 0, 0]) / 2, na_y = (kin[0, 1, 1] - kin[0, 0, 1]) / 2
        float_t na_z = max(na_x, na_y)**2 / 2
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        float_t[:, :, ::1] vot_vec
    inverse_mat_c(rec_basis[0], inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + na_z * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + na_z * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + na_z * inv_basis[2, 2])))
    vot_vec = np.empty((aa, 8 * h_size * k_size * l_size, 3), dtype=np.float64)
    vot_vec_c(vot_vec[idxs[0]:idxs[1]], kout_exp[idxs[0]:idxs[1]], rec_basis[0],
              inv_basis, kin[0], h_size, k_size, l_size)
    for i in range(1, a):
        inverse_mat_c(rec_basis[i], inv_basis)
        vot_vec_c(vot_vec[idxs[i]:idxs[i + 1]], kout_exp[idxs[i]:idxs[i + 1]],
                  rec_basis[i], inv_basis, kin[i], h_size, k_size, l_size)
    return np.asarray(vot_vec)

def vot_idxs_scan(float_t[:, ::1] kout_exp, float_t[:, :, ::1] rec_basis,
                  float_t[:, :, ::1] kin, int_t[::1] idxs):
    """
    Return voting points hkl indices of a scan

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    idxs - streaks unique indices
    """
    cdef:
        int aa = kout_exp.shape[0], a = rec_basis.shape[0], i, h_size, k_size, l_size
        float_t na_x = (kin[0, 1, 0] - kin[0, 0, 0]) / 2, na_y = (kin[0, 1, 1] - kin[0, 0, 1]) / 2
        float_t na_z = max(na_x, na_y)**2 / 2
        float_t[:, ::1] inv_basis = np.empty((3, 3), dtype=np.float64)
        int_t[:, :, ::1] vot_idxs
    inverse_mat_c(rec_basis[0], inv_basis)
    h_size = int(ceil(abs(na_x * inv_basis[0, 0] + na_y * inv_basis[1, 0] + na_z * inv_basis[2, 0])))
    k_size = int(ceil(abs(na_x * inv_basis[0, 1] + na_y * inv_basis[1, 1] + na_z * inv_basis[2, 1])))
    l_size = int(ceil(abs(na_x * inv_basis[0, 2] + na_y * inv_basis[1, 2] + na_z * inv_basis[2, 2])))
    vot_idxs = np.empty((aa, 8 * h_size * k_size * l_size, 3), dtype=np.int64)
    vot_idxs_c(vot_idxs[idxs[0]:idxs[1]], kout_exp[idxs[0]:idxs[1]], rec_basis[0],
               inv_basis, kin[0], h_size, k_size, l_size)
    for i in range(1, a):
        inverse_mat_c(rec_basis[i], inv_basis)
        vot_idxs_c(vot_idxs[idxs[i]:idxs[i + 1]], kout_exp[idxs[i]:idxs[i + 1]],
                   rec_basis[i], inv_basis, kin[i], h_size, k_size, l_size)
    return np.asarray(vot_idxs)

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

cdef float_t fit_streak(float_t[::1] vot_vec, float_t[:, ::1] kout_exp,
                        float_t[:, ::1] kin, float_t pen_coeff) nogil:
    """
    Return fitness value for a streak

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    pen_coeff - penalty coefficient
    """
    cdef:
        float_t rec_abs = sqrt(vot_vec[0]**2 + vot_vec[1]**2 + vot_vec[2]**2)
        float_t k0_x = kout_exp[0, 0] - vot_vec[0], k1_x = kout_exp[1, 0] - vot_vec[0]
        float_t k0_y = kout_exp[0, 1] - vot_vec[1], k1_y = kout_exp[1, 1] - vot_vec[1]
        float_t k0_z = kout_exp[0, 2] - vot_vec[2], k1_z = kout_exp[1, 2] - vot_vec[2]
        float_t n_x, n_y, n_z, 
        float_t src_th, src_phi, prod
        float_t fit_x, fit_y, fit, d0, d1
    if rec_abs != 0:
        src_th = acos(-vot_vec[2] / rec_abs) - acos(rec_abs / 2)
        src_phi = atan2(vot_vec[1], vot_vec[0])
        n_x = k0_y * k1_z - k0_z * k1_y
        n_y = k0_z * k1_x - k0_x * k1_z
        n_z = k0_x * k1_y - k0_y * k1_x
        prod = (k0_x * n_x + k0_y * n_y + (k0_z - 1) * n_z) / (n_x**2 + n_y**2 + n_z**2)
        fit_x = prod * n_x + sin(src_th) * cos(src_phi)
        fit_y = prod * n_y + sin(src_th) * sin(src_phi)
        fit = sqrt(fit_x**2 + fit_y**2)
    else:
        fit = 0

    d0 = frame_distance(k0_x, k0_y, kin)
    d1 = frame_distance(k1_x, k1_y, kin)
    fit += pen_coeff * (d0 + d1)
    return fit

cpdef float_t fit_frame(float_t[:, :, ::1] vot_vec, float_t[:, :, ::1] kout_exp, float_t[:, ::1] kin, float_t pen_coeff):
    """
    Return fitness value for a frame

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    pen_coeff - penalty coefficient
    """
    cdef:
        int a = vot_vec.shape[0], b = vot_vec.shape[1], i, j
        float_t fit = 0.0, min_fit, pt_fit
    with nogil:
        for i in range(a):
            min_fit = fit_streak(vot_vec[i, 0], kout_exp[i], kin, pen_coeff)
            for j in range(1, b):
                pt_fit = fit_streak(vot_vec[i, j], kout_exp[i], kin, pen_coeff)
                if pt_fit < min_fit:
                    min_fit = pt_fit
            fit += min_fit
    return fit / a

def fit_scan(float_t[:, :, ::1] vot_vec, float_t[:, :, ::1] kout_exp, float_t[:, :, ::1] kin,
              int_t[::1] idxs, float_t pen_coeff):
    """
    Return fitness value for a frame

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    idxs - streaks unique indices
    pen_coeff - penalty coefficient
    """
    cdef:
        int aa = kout_exp.shape[0], a = idxs.shape[0] - 1, i
        float_t[::1] fit = np.empty(a, dtype=np.float64)
        float_t[:, :, ::1] vv_frame
    for i in range(a):
        fit[i] = fit_frame(vot_vec[idxs[i]:idxs[i + 1]], kout_exp[idxs[i]:idxs[i + 1]],
                           kin[i], pen_coeff)
    return np.asarray(fit)

def fit_idxs(float_t[:, :, ::1] vot_vec, float_t[:, :, ::1] kout_exp, float_t[:, ::1] kin, float_t pen_coeff):
    """
    Return indices of the best lattice vectors of a frame

    vot_vec - voting reciprocal lattice vectors
    kout_exp - experimental outcoming wavevectors
    kin = [[kin_x_min, kin_y_min], [kin_x_max, kin_y_max]] - lens' pupil bounds
    pen_coeff - penalty coefficient
    """
    cdef:
        int a = vot_vec.shape[0], b = vot_vec.shape[1], i, j
        float_t min_fit, pt_fit
        int_t[::1] idxs = np.empty(a, dtype=np.int64)
    for i in range(a):
        min_fit = fit_streak(vot_vec[i, 0], kout_exp[i], kin, pen_coeff)
        idxs[i] = 0
        for j in range(1, b):
            pt_fit = fit_streak(vot_vec[i, j], kout_exp[i], kin, pen_coeff)
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
