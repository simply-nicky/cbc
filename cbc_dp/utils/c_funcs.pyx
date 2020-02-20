from libc.math cimport sqrt, sin, cos, tan, atan, atan2, acos, floor, ceil
from cpython cimport array
import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint8_t uint8_t

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

def draw_lines_aa(int_t[:, :, ::1] lines, float_t w, int_t shape_x, int_t shape_y):
    """
    Generate mask with anti-aliased lines with given thickness drawn

    lines - lines coordinates (x0, y0, x1, y1)
    w - line thickness
    shape = (shape_x, shape_y) - mask shape
    """
    cdef:
        float_t slope, thickness, val
        int_t a = lines.shape[0], R0, C0, R1 , C1, temp, i, j, k, k_max, ind, y, xx, yy
        float_t[:, ::1] mask = np.zeros((shape_x, shape_y), dtype=np.float64)
    for i in range(a):
        if abs(lines[i, 0, 0] - lines[i, 1, 0]) < abs(lines[i, 0, 1] - lines[i, 1, 1]):
            R0 = lines[i, 0, 0]; C0 = lines[i, 0, 1]; R1 = lines[i, 1, 0]; C1 = lines[i, 1, 1]
            if C0 > C1:
                temp = R0; R0 = R1; R1 = temp
                temp = C0; C0 = C1; C1 = temp
            slope = (float(R1) - float(R0)) / (float(C1) - float(C0))
            thickness = w * sqrt(1 + abs(slope)) / 2
            k_max = int(ceil(thickness / 2)) + 1
            for j in range(C0, C1 + 1):
                y = int(floor(j * slope + (C1 * R0 - C0 * R1) / (C1 - C0)))
                for k in range(-k_max, k_max + 1):
                    xx = j
                    if y + k >= 0 and y + k < shape_y:
                        yy = y + k
                        val = min(yy + 1 + w / 2 - y, -yy + 1 + w / 2 + y)
                        if val > 0:
                            mask[xx, yy] = min(val, 1.)
        else:
            R0 = lines[i, 0, 1]; C0 = lines[i, 0, 0]; R1 = lines[i, 1, 1]; C1 = lines[i, 1, 0]
            if C0 > C1:
                temp = R0; R0 = R1; R1 = temp
                temp = C0; C0 = C1; C1 = temp
            slope = (float(R1) - float(R0)) / (float(C1) - float(C0))
            thickness = w * sqrt(1 + abs(slope)) / 2
            k_max = int(ceil(thickness / 2)) + 1
            for j in range(C0, C1 + 1):
                y = int(floor(j * slope + (C1 * R0 - C0 * R1) / (C1 - C0)))
                for k in range(-k_max, k_max + 1):
                    yy = j
                    if y + k >= 0 and y + k < shape_x:
                        xx = y + k
                        val = min(xx + 1 + w / 2 - y, -xx + 1 + w / 2 + y)
                        if val > 0:
                            mask[xx, yy] = min(val, 1.)
    return np.asarray(mask)

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

def hl_refiner(float_t[:, :, ::1] lines, float_t w):
    cdef:
        int a = lines.shape[0], ii = 0, i, j
        uint8_t[::1] mask = np.zeros(a, dtype=np.uint8)
        float_t[:, :, ::1] hl = np.empty((a, 2, 2), dtype=np.float64)
        float_t p_00, p_01, p_10, p_11, temp
        float_t dist_0, tau_x, tau_y, dist_x, dist_y, dist_t, dist_n
    for i in range(a):
        if not mask[i]:
            hl[ii] = lines[i]
            dist_0 = sqrt((hl[ii, 1, 0] - hl[ii, 0, 0])**2 + (hl[ii, 1, 1] - hl[ii, 0, 1])**2)
            tau_x = (hl[ii, 1, 0] - hl[ii, 0, 0]) / dist_0
            tau_y = (hl[ii, 1, 1] - hl[ii, 0, 1]) / dist_0
            p_00 = hl[ii, 0, 0] * tau_x + hl[ii, 0, 1] * tau_y
            p_01 = hl[ii, 1, 0] * tau_x + hl[ii, 1, 1] * tau_y
            for j in range(a):
                if not mask[j] and j != i:
                    dist_x = (lines[i, 0, 0] - lines[j, 0, 0] + lines[i, 1, 0] - lines[j, 1, 0]) / 2
                    dist_y = (lines[i, 0, 1] - lines[j, 0, 1] + lines[i, 1, 1] - lines[j, 1, 1]) / 2
                    dist_t = dist_x * tau_x + dist_y * tau_y
                    dist_n = sqrt((dist_x - dist_t * tau_x)**2 + (dist_y - dist_t * tau_y)**2)
                    if abs(dist_t) < dist_0 / 2 and dist_n < w:
                        mask[j] = 1
                        p_10 = lines[j, 0, 0] * tau_x + lines[j, 0, 1] * tau_y
                        p_11 = lines[j, 1, 0] * tau_x + lines[j, 1, 1] * tau_y
                        if p_11 < p_10:
                            temp = lines[j, 1, 0]; lines[j, 1, 0] = lines[j, 0, 0]; lines[j, 0, 0] = temp
                            temp = lines[j, 1, 1]; lines[j, 1, 1] = lines[j, 0, 1]; lines[j, 0, 1] = temp
                            temp = p_11; p_11 = p_10; p_10 = temp
                        if p_10 < p_00:
                            hl[ii, 0] = lines[j, 0]
                        if p_11 > p_01:
                            hl[ii, 1] = lines[j, 1]
                        dist_0 = sqrt((hl[ii, 1, 0] - hl[ii, 0, 0])**2 + (hl[ii, 1, 1] - hl[ii, 0, 1])**2)
                        tau_x = (hl[ii, 1, 0] - hl[ii, 0, 0]) / dist_0
                        tau_y = (hl[ii, 1, 1] - hl[ii, 0, 1]) / dist_0
                        p_00 = hl[ii, 0, 0] * tau_x + hl[ii, 0, 1] * tau_y
                        p_01 = hl[ii, 1, 0] * tau_x + hl[ii, 1, 1] * tau_y
            ii += 1
    return np.array(hl)[:ii]

def lsd_refiner(float_t[:, :, ::1] lines, float_t w):
    cdef:
        int a = lines.shape[0], ii = 0, i, j
        uint8_t[::1] mask = np.zeros(a, dtype=np.uint8)
        float_t[:, :, ::1] ll = np.empty((a, 2, 2), dtype=np.float64)
        float_t dist_0, tau_x, tau_y, dist_x, dist_y, dist_t, dist_n, prod, dist_1
    for i in range(a):
        if not mask[i]:
            ll[ii] = lines[i]
            dist_0 = sqrt((ll[ii, 1, 0] - ll[ii, 0, 0])**2 + (ll[ii, 1, 1] - ll[ii, 0, 1])**2)
            tau_x = (ll[ii, 1, 0] - ll[ii, 0, 0]) / dist_0
            tau_y = (ll[ii, 1, 1] - ll[ii, 0, 1]) / dist_0
            for j in range(a):
                if not mask[j] and j != i:
                    dist_x = (ll[ii, 0, 0] - lines[j, 0, 0] + ll[ii, 1, 0] - lines[j, 1, 0]) / 2
                    dist_y = (ll[ii, 0, 1] - lines[j, 0, 1] + ll[ii, 1, 1] - lines[j, 1, 1]) / 2
                    dist_t = dist_x * tau_x + dist_y * tau_y
                    dist_n = sqrt((dist_x - dist_t * tau_x)**2 + (dist_y - dist_t * tau_y)**2)
                    if abs(dist_t) < dist_0 / 2 and dist_n < w:
                        mask[j] = 1
                        prod = tau_x * (lines[j, 1, 0] - lines[j, 0, 0]) + tau_y * (lines[j, 1, 1] - lines[j, 0, 1])
                        dist_1 = sqrt((lines[j, 1, 0] - lines[j, 0, 0])**2 + (lines[j, 1, 1] - lines[j, 0, 1])**2)
                        if prod >= 0:
                            ll[ii, 0, 0] += dist_1 / (dist_0 + dist_1) * (lines[j, 0, 0] - ll[ii, 0, 0])
                            ll[ii, 0, 1] += dist_1 / (dist_0 + dist_1) * (lines[j, 0, 1] - ll[ii, 0, 1])
                            ll[ii, 1, 0] += dist_1 / (dist_0 + dist_1) * (lines[j, 1, 0] - ll[ii, 1, 0])
                            ll[ii, 1, 1] += dist_1 / (dist_0 + dist_1) * (lines[j, 1, 1] - ll[ii, 1, 1])
                        else:
                            ll[ii, 0, 0] += dist_1 / (dist_0 + dist_1) * (lines[j, 1, 0] - ll[ii, 0, 0])
                            ll[ii, 0, 1] += dist_1 / (dist_0 + dist_1) * (lines[j, 1, 1] - ll[ii, 0, 1])
                            ll[ii, 1, 0] += dist_1 / (dist_0 + dist_1) * (lines[j, 0, 0] - ll[ii, 1, 0])
                            ll[ii, 1, 1] += dist_1 / (dist_0 + dist_1) * (lines[j, 0, 1] - ll[ii, 1, 1])
                        dist_0 = sqrt((ll[ii, 1, 0] - ll[ii, 0, 0])**2 + (ll[ii, 1, 1] - ll[ii, 0, 1])**2)
                        tau_x = (ll[ii, 1, 0] - ll[ii, 0, 0]) / dist_0
                        tau_y = (ll[ii, 1, 1] - ll[ii, 0, 1]) / dist_0
            ii += 1
    return np.array(ll)[:ii]

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

def kout(float_t[:, :, ::1] lines, float_t[::1] det_pos, float_t[::1] vec):
    """
    Return outcoming wavevectors array based on a target function point
    
    lines - detected diffraction streaks lines at the detector [mm]
    det_pos = [det_x, det_y, det_z] - relative to the sample detector position
    vec - target function vector
    """
    cdef:
        int a = lines.shape[0]
        float_t[:, :, ::1] kout = np.empty((a, 2, 3), dtype=np.float64)
        int i, j
        float_t x, y, phi, theta
    for i in range(a):
        for j in range(2):
            x = lines[i, j, 0] - det_pos[0] * (1 + vec[0])
            y = lines[i, j, 1] - det_pos[1] * (1 + vec[1])
            phi = atan2(y, x)
            theta = atan(sqrt(x**2 + y**2) / det_pos[2] / (1 + vec[2]))
            kout[i, j, 0] = sin(theta) * cos(phi)
            kout[i, j, 1] = sin(theta) * sin(phi)
            kout[i, j, 2] = cos(theta)
    return np.asarray(kout)

def voting_vectors(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t num_ap_x, float_t num_ap_y):
    """
    Return reciprocal lattice voting points for the given experiment outcoming wavevectors kout_exp

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    num_ap_x, num_ap_y - numerical apertires in x- and y-axes
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig, h_ind, k_ind, l_ind, ind
        float_t rec_x, rec_y, rec_z, rec_abs, source_th, source_phi
        float_t[:, ::1] inv_basis = np.linalg.inv(rec_basis)
        float_t max_na = max(num_ap_x, num_ap_y)**2 / 2
        int_t h_size = int(ceil(abs(num_ap_x * inv_basis[0, 0] + num_ap_y * inv_basis[1, 0] + max_na * inv_basis[2, 0])))
        int_t k_size = int(ceil(abs(num_ap_x * inv_basis[0, 1] + num_ap_y * inv_basis[1, 1] + max_na * inv_basis[2, 1])))
        int_t l_size = int(ceil(abs(num_ap_x * inv_basis[0, 2] + num_ap_y * inv_basis[1, 2] + max_na * inv_basis[2, 2])))
        float_t[:, :, ::1] vot_vec = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.float64)
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

def voting_idxs(float_t[:, ::1] kout_exp, float_t[:, ::1] rec_basis, float_t num_ap_x, float_t num_ap_y):
    """
    Return voting points hkl indices for the given experiment outcoming wavevectors kout_exp

    kout_exp - experimental outcoming wavevectors
    rec_basis - reciprocal lattice basis vectors
    num_ap_x, num_ap_y - numerical apertires in x- and y-axes
    """
    cdef:
        int_t a = kout_exp.shape[0], i, ii, jj, kk, h_orig, k_orig, l_orig, h_ind, k_ind, l_ind, ind
        float_t rec_x, rec_y, rec_z, rec_abs, source_th, source_phi
        float_t[:, ::1] inv_basis = np.linalg.inv(rec_basis)
        float_t max_na = max(num_ap_x, num_ap_y)**2 / 2
        int_t h_size = int(ceil(abs(num_ap_x * inv_basis[0, 0] + num_ap_y * inv_basis[1, 0] + max_na * inv_basis[2, 0])))
        int_t k_size = int(ceil(abs(num_ap_x * inv_basis[0, 1] + num_ap_y * inv_basis[1, 1] + max_na * inv_basis[2, 1])))
        int_t l_size = int(ceil(abs(num_ap_x * inv_basis[0, 2] + num_ap_y * inv_basis[1, 2] + max_na * inv_basis[2, 2])))
        int_t[:, :, ::1] vot_idxs = np.empty((a, 8 * h_size * k_size * l_size, 3), dtype=np.int64)
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

def fitness(float_t[:, :, ::1] vot_vec,
            float_t[:, :, ::1] kout_exp,
            float_t num_ap_x,
            float_t num_ap_y,
            float_t pen_coeff):
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
        if dist_x > num_ap_x:
            min_fit += pen_coeff * (dist_x - num_ap_x)
        dist_y = abs(vot_vec[i, 0, 1] - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2)
        if dist_y > num_ap_y:
            min_fit += pen_coeff * (dist_y - num_ap_y)
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
            if dist_x > num_ap_x:
                pt_fit += pen_coeff * (dist_x - num_ap_x)
            dist_y = abs(vot_vec[i, j, 1] - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2)
            if dist_y > num_ap_y:
                pt_fit += pen_coeff * (dist_y - num_ap_y)
            if pt_fit < min_fit:
                min_fit = pt_fit
        fitness += min_fit
    return fitness / a

def fitness_idxs(float_t[:, :, ::1] vot_vec,
                 float_t[:, :, ::1] kout_exp,
                 float_t num_ap_x,
                 float_t num_ap_y,
                 float_t pen_coeff):
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
        if dist_x > num_ap_x:
            min_fit += pen_coeff * (dist_x - num_ap_x)
        dist_y = abs(vot_vec[i, 0, 1] - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2)
        if dist_y > num_ap_y:
            min_fit += pen_coeff * (dist_y - num_ap_y)
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
            if dist_x > num_ap_x:
                pt_fit += pen_coeff * (dist_x - num_ap_x)
            dist_y = abs(vot_vec[i, j, 1] - (kout_exp[i, 0, 1] + kout_exp[i, 1, 1]) / 2)
            if dist_y > num_ap_y:
                pt_fit += pen_coeff * (dist_y - num_ap_y)
            if pt_fit < min_fit:
                min_fit = pt_fit; idxs[i] = j
    return (np.arange(a), np.asarray(idxs))