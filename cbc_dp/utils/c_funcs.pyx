from libc.math cimport sqrt, sin, cos, tan, atan, atan2, acos
from cpython cimport array
import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint8_t uint8_t

def py_swap(float_t a, float_t b) -> None:
    cdef float_t temp = a
    a = b
    b = temp

def binary_search(float_t[::1] values, int l, int r, float_t x):
    cdef int_t m = l + (r - l) // 2
    if l <= r:
        if x == values[m]:
            return m
        elif x > values[m] and x <= values[m + 1]:
            return m + 1
        elif x < values[m]:
            return binary_search(values, l, m, x)
        else:
            return binary_search(values, m + 1, r, x)

def searchsorted(float_t[::1] values, float_t x):
    cdef int r = values.shape[0]
    if x < values[0]:
        return 0
    elif x > values[r - 1]:
        return r
    else:
        return binary_search(values, 0, r, x)

def make_grid(float_t[:, ::1] points, float_t[::1] values, int_t size):
    """
    Make grid array with shape (size, size, size) based on points array and values to fill

    points - points array of shape (N, 3)
    values - values array of shape (N,) to fill into grid
    size - grid size
    """
    cdef:
        int a = points.shape[0]
        int i, ii, jj, kk
        float_t[:, :, ::1] grid = np.zeros((size, size, size), dtype=np.float64)
        float_t[::1] x_coord = np.linspace(min(points[:, 0]), max(points[:, 0]), size)
        float_t[::1] y_coord = np.linspace(min(points[:, 1]), max(points[:, 1]), size)
        float_t[::1] z_coord = np.linspace(min(points[:, 2]), max(points[:, 2]), size)
    for i in range(a):
        ii = searchsorted(x_coord, points[i, 0])
        jj = searchsorted(y_coord, points[i, 1])
        kk = searchsorted(z_coord, points[i, 2])
        grid[ii, jj, kk] = values[i]
    return np.array(grid)

def hl_refiner(float_t[:, :, ::1] lines, float_t[:, ::1] taus, float_t d_tau, float_t d_n):
    cdef:
        int a = lines.shape[0], count = 0
        uint8_t[::1] mask = np.zeros(a, dtype=np.uint8)
        float_t[:, :, ::1] hl_lines = np.empty((a, 2, 2), dtype=np.float64)
        int i, j
        int_t[::1] min_idx = np.empty((2,), dtype=np.int64)
        int_t[::1] max_idx = np.empty((2,), dtype=np.int64)
        float_t proj_00, proj_01, proj_10, proj_11
        float_t dist_x, dist_y, tau_dist, n_dist
    for i in range(a):
        if not mask[i]:
            proj_00 = lines[i, 0, 0] * taus[i, 0] + lines[i, 0, 1] * taus[i, 1]
            proj_01 = lines[i, 1, 0] * taus[i, 0] + lines[i, 1, 1] * taus[i, 1]
            if proj_00 < proj_01:
                min_idx[0] = i; min_idx[1] = 0
                max_idx[0] = i; max_idx[1] = 1
            else:
                min_idx[0] = i; min_idx[1] = 1
                max_idx[0] = i; max_idx[1] = 0
                py_swap(proj_00, proj_01)
            for j in range(a):
                if i == j:
                    continue
                dist_x = (lines[i, 0, 0] - lines[j, 0, 0] + lines[i, 1, 0] - lines[j, 1, 0]) / 2
                dist_y = (lines[i, 0, 1] - lines[j, 0, 1] + lines[i, 1, 1] - lines[j, 1, 1]) / 2
                tau_dist = abs(dist_x * taus[i, 0] + dist_y * taus[i, 1])
                n_dist = sqrt((dist_x - tau_dist * taus[i, 0])**2 + (dist_y - tau_dist * taus[i, 1])**2)
                if tau_dist < d_tau and n_dist < d_n:
                    mask[j] = 1
                    proj_10 = lines[j, 0, 0] * taus[i, 0] + lines[j, 0, 1] * taus[i, 1]
                    proj_11 = lines[j, 1, 0] * taus[i, 0] + lines[j, 1, 1] * taus[i, 1]
                    if proj_10 < proj_00:
                        min_idx[0] = j; min_idx[1] = 0
                    if proj_10 > proj_10:
                        max_idx[0] = j; max_idx[1] = 0
                    if proj_11 < proj_00:
                        min_idx[0] = j; min_idx[1] = 1
                    if proj_11 > proj_10:
                        max_idx[0] = j; max_idx[1] = 1
            hl_lines[count, 0] = lines[min_idx[0], min_idx[1]]
            hl_lines[count, 1] = lines[max_idx[0], max_idx[1]]
            count += 1
    return np.array(hl_lines)[:count]

def lsd_refiner(float_t[:, :, ::1] lines, float_t[:, ::1] taus, float_t d_tau, float_t d_n):
    cdef:
        int a = lines.shape[0], count = 0, i, j
        uint8_t[::1] mask = np.zeros(a, dtype=np.uint8)
        float_t[:, :, ::1] lsd_lines = np.empty((a, 2, 2), dtype=np.float64)
        float_t proj_00, proj_01, proj_10, proj_11
        float_t dist_x, dist_y, tau_dist, n_dist
    for i in range(a):
        if not mask[i]:
            proj_00 = lines[i, 0, 0] * taus[i, 0] + lines[i, 0, 1] * taus[i, 1]
            proj_01 = lines[i, 1, 0] * taus[i, 0] + lines[i, 1, 1] * taus[i, 1]
            if proj_00 < proj_01:
                lsd_lines[count, 0] = lines[i, 0]
                lsd_lines[count, 1] = lines[i, 1]
            else:
                lsd_lines[count, 0] = lines[i, 1]
                lsd_lines[count, 1] = lines[i, 0]
            for j in range(a):
                if i == j:
                    continue
                dist_x = (lines[i, 0, 0] - lines[j, 0, 0] + lines[i, 1, 0] - lines[j, 1, 0]) / 2
                dist_y = (lines[i, 0, 1] - lines[j, 0, 1] + lines[i, 1, 1] - lines[j, 1, 1]) / 2
                tau_dist = abs(dist_x * taus[i, 0] + dist_y * taus[i, 1])
                n_dist = sqrt((dist_x - tau_dist * taus[i, 0])**2 + (dist_y - tau_dist * taus[i, 1])**2)
                if tau_dist < d_tau and n_dist < d_n:
                    mask[j] = 1
                    proj_10 = lines[j, 0, 0] * taus[i, 0] + lines[j, 0, 1] * taus[i, 1]
                    proj_11 = lines[j, 1, 0] * taus[i, 0] + lines[j, 1, 1] * taus[i, 1]
                    if proj_10 < proj_11:
                        lsd_lines[count, 0, 0] = (lsd_lines[count, 0, 0] + lines[j, 0, 0]) / 2
                        lsd_lines[count, 0, 1] = (lsd_lines[count, 0, 1] + lines[j, 0, 1]) / 2
                        lsd_lines[count, 1, 0] = (lsd_lines[count, 1, 0] + lines[j, 1, 0]) / 2
                        lsd_lines[count, 1, 1] = (lsd_lines[count, 1, 1] + lines[j, 1, 1]) / 2
                    else:
                        lsd_lines[count, 0, 0] = (lsd_lines[count, 0, 0] + lines[j, 1, 0]) / 2
                        lsd_lines[count, 0, 1] = (lsd_lines[count, 0, 1] + lines[j, 1, 1]) / 2
                        lsd_lines[count, 1, 0] = (lsd_lines[count, 1, 0] + lines[j, 0, 0]) / 2
                        lsd_lines[count, 1, 1] = (lsd_lines[count, 1, 1] + lines[j, 0, 1]) / 2
            count += 1
    return np.array(lsd_lines)[:count]

def source_index(float_t[:, :, ::1] rec_vec):
    """
    Return reciprocal vector angles and source line origin points for an IndexLattice class object

    rec_vec - array of reciprocal vectors
    """
    cdef:
        int a = rec_vec.shape[0], b = rec_vec.shape[1]
        int i, j
        float_t source_th
        float_t[:, ::1] rec_abs = np.empty((a, b), dtype=np.float64)
        float_t[:, ::1] rec_th = np.empty((a, b), dtype=np.float64)
        float_t[:, ::1] rec_phi = np.empty((a, b), dtype=np.float64)
        float_t[:, :, ::1] source = np.empty((a, b, 3), dtype=np.float64)
    for i in range(a):
        for j in range(b):
            rec_abs[i, j] = sqrt(rec_vec[i, j, 0]**2 + rec_vec[i, j, 1]**2 + rec_vec[i, j, 2]**2)
            rec_th[i, j] = acos(-rec_vec[i, j, 2] / rec_abs[i, j])
            rec_phi[i, j] = atan2(rec_vec[i, j, 1], rec_vec[i, j, 0])
            source_th = rec_th[i, j] - acos(rec_abs[i, j] / 2)
            source[i, j, 0] = -sin(source_th) * cos(rec_phi[i, j])
            source[i, j, 1] = -sin(source_th) * sin(rec_phi[i, j])
            source[i, j, 2] =  cos(source_th)
    return np.asarray(rec_abs), np.asarray(rec_th), np.asarray(rec_phi), np.asarray(source)

def source_ball(float_t[:, ::1] rec_vec):
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

def source_lines(float_t[:, :, ::1] source, float_t[:, :, ::1] rec_vec, float_t num_ap_x, float_t num_ap_y):
    """
    Return source lines coordinates for the convergent beam indexer

    source - source line origins
    rec_vec - reciprocal vectors
    num_ap_x, num_ap_y - numerical apertures in x- and y-axes
    """
    cdef:
        int a = rec_vec.shape[0], b = rec_vec.shape[1], i, j, k
        float_t source_prd, coeff1, coeff2, alpha, betta, gamma, delta
        float_t[::1] bounds = np.array([num_ap_x, -num_ap_x, num_ap_y, -num_ap_y], dtype=np.float64)
        float_t[:, :, :, ::1] source_lines = np.empty((a, b, 8, 2), dtype=np.float64)
    for i in range(a):
        for j in range(b):
            source_prd = (source[i, j, 0] * rec_vec[i, j, 0] +
                          source[i, j, 1] * rec_vec[i, j, 1] +
                          source[i, j, 2] * rec_vec[i, j, 2])
            for k in range(4):
                coeff1 = source_prd - bounds[k] * rec_vec[i, j, k // 2]
                coeff2 = rec_vec[i, j, (3 - k) // 2]
                alpha = coeff2**2 + rec_vec[i, j, 2]**2
                betta = coeff2 * coeff1
                gamma = coeff1**2 - rec_vec[i, j, 2]**2 * (1 - bounds[k]**2)
                delta = betta**2 - alpha * gamma
                source_lines[i, j, k, k // 2] = bounds[k]
                source_lines[i, j, k, (3 - k) // 2] = (betta + sqrt(delta)) / alpha
                source_lines[i, j, k + 4, k // 2] = bounds[k]
                source_lines[i, j, k + 4, (3 - k) // 2] = (betta - sqrt(delta)) / alpha
    return np.asarray(source_lines)

def model_source_lines(float_t[:, ::1] source, float_t[:, ::1] rec_vec, float_t num_ap_x, float_t num_ap_y):
    """
    Return source lines coordinates for a diffraction streaks model

    source - source line origins
    rec_vec - reciprocal vectors
    num_ap_x, num_ap_y - numerical apertires in x- and y-axes
    """
    cdef:
        int a = rec_vec.shape[0], ii = 0, jj, i, k
        uint8_t[::1] mask = np.zeros(a, dtype=np.uint8)
        float_t source_prd, coeff1, coeff2, alpha, betta, gamma, delta, sol_1, sol_2
        float_t[::1] bounds = np.array([num_ap_x, -num_ap_x, num_ap_y, -num_ap_y], dtype=np.float64)
        float_t[:, :, ::1] source_lines = np.empty((a, 2, 3), dtype=np.float64)
    for i in range(a):
        source_prd = (source[i, 0] * rec_vec[i, 0] + source[i, 1] * rec_vec[i, 1] + source[i, 2] * rec_vec[i, 2])
        jj = 0
        for k in range(4):
            coeff1 = source_prd - bounds[k] * rec_vec[i, k // 2]
            coeff2 = rec_vec[i, (3 - k) // 2]
            alpha = coeff2**2 + rec_vec[i, 2]**2
            betta = coeff2 * coeff1
            gamma = coeff1**2 - rec_vec[i, 2]**2 * (1 - bounds[k]**2)
            delta = betta**2 - alpha * gamma
            sol_1 = (betta + sqrt(delta)) / alpha
            sol_2 = (betta - sqrt(delta)) / alpha
            if abs(sol_1) < abs(bounds[3 - k]):
                source_lines[ii, jj, k // 2] = bounds[k]
                source_lines[ii, jj, (3 - k) // 2] = sol_1
                source_lines[ii, jj, 2] = sqrt(1 - bounds[k]**2 - sol_1**2)
                jj += 1
            if delta != 0 and abs(sol_2) < abs(bounds[3 - k]):
                source_lines[ii, jj, k // 2] = bounds[k]
                source_lines[ii, jj, (3 - k) // 2] = sol_2
                source_lines[ii, jj, 2] = sqrt(1 - bounds[k]**2 - sol_2**2)
                jj += 1
            if jj == 2:
                mask[i] = 1; ii += 1
                break
    return np.asarray(source_lines[:ii]), np.asarray(mask).astype(bool)

def kout(float_t[:, :, ::1] lines, float_t[:] point, float_t pix_size):
    """
    Return outcoming wavevectors array based on a target function point

    lines - detected diffraction streaks lines at the detector [pixels]
    point - target function point
    pix_size - detector pixel size [mm]
    """
    cdef:
        int a = lines.shape[0]
        float_t[:, :, ::1] kout = np.empty((a, 2, 3), dtype=np.float64)
        int i, j
        float_t x, y, phi, theta
    for i in range(a):
        for j in range(2):
            x = lines[i, j, 0] * pix_size - point[0]
            y = lines[i, j, 1] * pix_size - point[1]
            phi = atan2(y, x)
            theta = atan(sqrt(x**2 + y**2) / point[2])
            kout[i, j, 0] = sin(theta) * cos(phi)
            kout[i, j, 1] = sin(theta) * sin(phi)
            kout[i, j, 2] = cos(theta)
    return np.asarray(kout)

def fitness(float_t[:, :, ::1] index_vec, float_t[:, :, ::1] kout):
    """
    Return target function fitness value for a given point

    index_vec - IndexLattice class object reciprocal vectors
    kout - experimental outcoming wavevectors
    """
    cdef:
        int a = index_vec.shape[0], b = index_vec.shape[1], i, j
        float_t norm_abs_0, norm_abs_1, pt_fit, min_fit, fitness = 0.0
    for i in range(a):
        norm_abs_0 = sqrt((index_vec[i, 0, 0] - kout[i, 0, 0])**2 +
                          (index_vec[i, 0, 1] - kout[i, 0, 1])**2 +
                          (index_vec[i, 0, 2] - kout[i, 0, 2])**2)
        norm_abs_1 = sqrt((index_vec[i, 0, 0] - kout[i, 1, 0])**2 +
                          (index_vec[i, 0, 1] - kout[i, 1, 1])**2 +
                          (index_vec[i, 0, 2] - kout[i, 1, 2])**2)
        min_fit = abs(1 - norm_abs_0) + abs(1 - norm_abs_1)
        for j in range(1, b):
            norm_abs_0 = sqrt((index_vec[i, j, 0] - kout[i, 0, 0])**2 +
                              (index_vec[i, j, 1] - kout[i, 0, 1])**2 +
                              (index_vec[i, j, 2] - kout[i, 0, 2])**2)
            norm_abs_1 = sqrt((index_vec[i, j, 0] - kout[i, 1, 0])**2 +
                              (index_vec[i, j, 1] - kout[i, 1, 1])**2 +
                              (index_vec[i, j, 2] - kout[i, 1, 2])**2)
            pt_fit = abs(1 - norm_abs_0) + abs(1 - norm_abs_1)
            if pt_fit < min_fit:
                min_fit = pt_fit
        fitness += min_fit
    return fitness / a

def fitness_idxs(float_t[:, :, ::1] index_vec, float_t[:, :, ::1] kout):
    """
    Return indices of the best lattice vectors based on fitness values

    index_vec - IndexLattice class object reciprocal vectors
    kout - experimental outcoming wavevectors
    """
    cdef:
        int a = index_vec.shape[0], b = index_vec.shape[1], i, j
        float_t norm_abs_0, norm_abs_1, pt_fit, min_fit, fitness = 0.0
        int_t[::1] idxs = np.empty(a, dtype=np.int64)
    for i in range(a):
        norm_abs_0 = sqrt((index_vec[i, 0, 0] - kout[i, 0, 0])**2 +
                          (index_vec[i, 0, 1] - kout[i, 0, 1])**2 +
                          (index_vec[i, 0, 2] - kout[i, 0, 2])**2)
        norm_abs_1 = sqrt((index_vec[i, 0, 0] - kout[i, 1, 0])**2 +
                          (index_vec[i, 0, 1] - kout[i, 1, 1])**2 +
                          (index_vec[i, 0, 2] - kout[i, 1, 2])**2)
        min_fit = abs(1 - norm_abs_0) + abs(1 - norm_abs_1)
        idxs[i] = 0
        for j in range(1, b):
            norm_abs_0 = sqrt((index_vec[i, j, 0] - kout[i, 0, 0])**2 +
                              (index_vec[i, j, 1] - kout[i, 0, 1])**2 +
                              (index_vec[i, j, 2] - kout[i, 0, 2])**2)
            norm_abs_1 = sqrt((index_vec[i, j, 0] - kout[i, 1, 0])**2 +
                              (index_vec[i, j, 1] - kout[i, 1, 1])**2 +
                              (index_vec[i, j, 2] - kout[i, 1, 2])**2)
            pt_fit = abs(1 - norm_abs_0) + abs(1 - norm_abs_1)
            if pt_fit < min_fit:
                min_fit = pt_fit; idxs[i] = j
    return (np.arange(a), np.asarray(idxs))