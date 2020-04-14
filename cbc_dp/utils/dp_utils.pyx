"""
dp_utils.pyx - Data processing utility functions written in Cython
"""
from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t
ctypedef cnp.uint64_t uint_t
ctypedef cnp.uint8_t uint8_t

def chunkify(int_t num, int_t chunk):
    cdef:
        int_t size = num / chunk, rem = num % chunk
        int_t[::1] chunks = np.empty(size + (rem != 0), dtype=np.int64)
    for i in range(size):
        chunks[i] = chunk
    if rem:
        chunks[size] = rem
    return np.asarray(chunks)

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

cdef uint_t wirthselect_uint(uint_t[::1] array, int k) nogil:
    """
    Nikolaus Wirth's selection algorithm to find the kth smallest element
    """
    cdef:
        int_t l = 0, m = array.shape[0] - 1, i, j
        uint_t x, tmp 
    while l < m: 
        x = array[k] 
        i = l; j = m 
        while 1: 
            while array[i] < x: i += 1 
            while x < array[j]: j -= 1 
            if i <= j: 
                tmp = array[i]; array[i] = array[j]; array[j] = tmp
                i += 1; j -= 1 
            if i > j: break 
        if j < k: l = i 
        if k < i: m = j 
    return array[k]

cdef int_t wirthselect_int(int_t[::1] array, int k) nogil:
    """
    Nikolaus Wirth's selection algorithm to find the kth smallest element
    """
    cdef:
        int_t l = 0, m = array.shape[0] - 1, i, j
        int_t x, tmp 
    while l < m: 
        x = array[k] 
        i = l; j = m 
        while 1: 
            while array[i] < x: i += 1 
            while x < array[j]: j -= 1 
            if i <= j: 
                tmp = array[i]; array[i] = array[j]; array[j] = tmp
                i += 1; j -= 1 
            if i > j: break 
        if j < k: l = i 
        if k < i: m = j 
    return array[k]

cdef uint8_t[:, ::1] background_mask(uint8_t[:, ::1] mask, uint8_t[:, ::1] structure):
    """
    Generate background mask for a streak mask based on morphological erosion with a binary structure
    """
    cdef:
        int_t a = mask.shape[0], b = mask.shape[1]
        int_t ksize = structure.shape[0], i, j, ii, jj
        uint8_t bgd
        uint8_t[:, ::1] bgd_mask = np.zeros((a, b), dtype=np.uint8)
    for i in range(a):
        for j in range(b):
            if mask[i, j]:
                bgd = 1
                for ii in range(ksize):
                    for jj in range(ksize):
                        if structure[ii, jj]:
                            bgd *= mask[i - ksize//2 + ii, j - ksize//2 + jj]
                bgd_mask[i, j] = mask[i, j] - bgd
    return bgd_mask

def background_filter(uint_t[:, :, ::1] data, uint_t[:, ::1] bgd, uint8_t[:, ::1] bad_mask,
                      int_t k_max, float_t sigma):
    """
    Generate refined background with experimentally measured one using median filtering

    data - diffraction data
    bgd - experimentally measured background
    bad_mask - bad pixels mask
    k_max - median filtering kernel size
    sigma - filtering threshold
    """
    cdef:
        int_t a = data.shape[0], b = data.shape[1], c = data.shape[2]
        int_t count, i, j, k, l, m, n, k0, k1
        uint_t[:, :, ::1] res = np.empty((a, b, c), dtype=np.uint64)
        uint_t[::1] array = np.empty(k_max, dtype=np.uint64)
    for i in range(b):
        for j in range(c):
            if bad_mask[i, j]:
                for k in range(a):
                    res[k, i, j] = data[k, i, j]
            else:
                for k in range(a):
                    k0 = k - k_max // 2
                    k1 = k + k_max // 2
                    count = 0
                    for l in range(k0, k1):
                        m = l % (a - 1)
                        n = l // (a - 1) % 2
                        l = n * (a - 1) + (1 - 2 * n) * m
                        if data[l, i, j] < (1 + sigma) * bgd[i, j] and data[l, i, j] > (1 - sigma) * bgd[i, j]:
                            array[count] = data[l, i, j]; count += 1
                    if count:
                        res[k, i, j] = wirthselect_uint(array[:count], count // 2)
                    else:
                        res[k, i, j] = bgd[i, j]
    return np.asarray(res)

cdef void streak_mask(float_t[:, ::1] streak, uint8_t[:, ::1] structure, int_t width, uint8_t[:, ::1] mask):
    """
    Generate a streak mask with the given line width and binary dilated with the given structure

    streak - streak's coordinates (x0, y0, x1, y1)
    structure - binary structure
    width - line thickness
    mask - output mask
    """
    cdef:
        float_t slope, thickness, R0, C0, R1 , C1, temp
        int_t r_max = structure.shape[0], c_max = structure.shape[1], i, j, k, y, xx, yy, r, c
        int_t shape_x = mask.shape[0], shape_y = mask.shape[1], k_max = width // 2
    if abs(streak[0, 0] - streak[1, 0]) < abs(streak[0, 1] - streak[1, 1]):
        R0 = streak[0, 0]; C0 = streak[0, 1]; R1 = streak[1, 0]; C1 = streak[1, 1]
        if C0 > C1:
            temp = R0; R0 = R1; R1 = temp
            temp = C0; C0 = C1; C1 = temp
        slope = (R1 - R0) / (C1 - C0)
        for j in range(int(C0), int(C1) + 1):
            y = int(j * slope + (C1 * R0 - C0 * R1) / (C1 - C0))
            for k in range(-k_max, k_max + 1):
                for r in range(r_max):
                    for c in range(c_max):
                        xx = j + r - r_max // 2
                        yy = y + k + c - c_max // 2
                        if yy >= 0 and yy < shape_y and xx >= 0 and xx < shape_x and structure[r, c]:
                            mask[xx, yy] = 1
    else:
        R0 = streak[0, 1]; C0 = streak[0, 0]; R1 = streak[1, 1]; C1 = streak[1, 0]
        if C0 > C1:
            temp = R0; R0 = R1; R1 = temp
            temp = C0; C0 = C1; C1 = temp
        slope = (R1 - R0) / (C1 - C0)
        for j in range(int(C0), int(C1) + 1):
            y = int(j * slope + (C1 * R0 - C0 * R1) / (C1 - C0))
            for k in range(-k_max, k_max + 1):
                for r in range(r_max):
                    for c in range(c_max):
                        xx = y + k + c - c_max // 2
                        yy = j + r - r_max // 2
                        if yy >= 0 and yy < shape_y and xx >= 0 and xx < shape_x and structure[r, c]:
                            mask[xx, yy] = 1

def streaks_mask(float_t[:, :, ::1] lines, uint8_t[:, ::1] structure, int_t width, int_t shape_x, int_t shape_y):
    """
    Generate a streaks mask with the given line width and binary dilated with the given structure

    lines - lines coordinates (x0, y0, x1, y1)
    structure - binary structure
    width - line thickness
    shape = (shape_x, shape_y) - mask shape
    """
    cdef:
        int_t a = lines.shape[0], i
        uint8_t[:, ::1] mask = np.zeros((shape_x, shape_y), dtype=np.uint8)
    for i in range(a):
        streak_mask(lines[i], structure, width, mask)
    return np.asarray(mask)

def normalize_frame(int_t[:, ::1] data, int[:, ::1] labels, uint8_t[:, ::1] structure, int_t lbl_num, float_t threshold):
    """
    Normalize a diffraction pattern for better line detection

    data - diffraction data
    labels - mask with all the diffraction streaks labeled
    lbl_num - number of labels
    threshold - detection threshold
    """
    cdef:
        uint8_t found
        int_t a = data.shape[0], b = data.shape[1], cnt, bgd_cnt, i, j, max_val, bgd_median
        float_t mean, var, delta
        uint8_t[:, ::1] mask, bgd_mask
        int_t[::1] bgd_arr = np.empty(a * b, dtype=np.int64)
        float_t[:, ::1] norm_data = np.zeros((a, b), dtype=np.float64)
    for lbl in range(1, 1 + lbl_num):
        mask = np.zeros((a, b), dtype=np.uint8)
        for i in range(a):
            for j in range(b):
                if labels[i, j] == lbl:
                    mask[i, j] = 1
        bgd_mask = background_mask(mask, structure)
        found = 1; mean = 0; var = 0; cnt = 0; bgd_cnt = 0
        for i in range(a):
            for j in range(b):
                if mask[i, j]:
                    cnt += 1
                    delta = float(data[i, j]) - mean
                    mean += delta / cnt
                    var += (data[i, j] - mean) * delta
                    if bgd_mask[i, j]:
                        bgd_arr[bgd_cnt] = data[i, j]; bgd_cnt += 1
                    if found:
                        max_val = data[i, j]; found = 0
                    elif max_val < data[i, j]:
                        max_val = data[i, j]
        bgd_median = wirthselect_int(bgd_arr[:bgd_cnt], bgd_cnt // 2)
        var = var / cnt
        if max_val - bgd_median > threshold * sqrt(var):
            for i in range(a):
                for j in range(b):
                    if mask[i, j] and not bgd_mask[i, j]:
                        norm_data[i, j] = (float(data[i, j]) - float(bgd_median)) / (float(max_val) - float(bgd_median))
    return np.asarray(norm_data)

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
    """
    Group and refine HL detected lines

    lines - HL detected lines [pixels]
    w - line grouping zone width [pixels]
    """
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
    """
    Group and refine LSD detected lines

    lines - LSD detected lines [pixels]
    w - line grouping zone width [pixels]
    """
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

def i_sigma_frame(float_t[:, :, ::1] streaks, float_t[:, :, ::1] source_streaks, int_t[:, ::1] cor_data,
                  uint_t[:, ::1] background, uint8_t[:, ::1] structure, int_t width):
    """
    Return diffraction reflection's intensity and Poisson noise of a frame

    streaks - detected diffraction streaks at the detector plane
    source_streaks - source streaks at the detector plane
    cor_data - background subtracted diffraction pattern image
    background - background image
    structure - binary structure for binary dilation
    width - diffraction streaks width
    """
    cdef:
        int_t aa = streaks.shape[0], a = cor_data.shape[0], b = cor_data.shape[1]
        int_t cnt = 0, s_cnt = 0, ii, i, j
        float_t bgd_mean, bgd_var, s_mean, s_var, I, bgd_sigma, delta
        uint8_t[:, ::1] mask = np.empty((a, b), dtype=np.uint8)
        uint8_t[:, ::1] s_mask = np.empty((a, b), dtype=np.uint8)
        float_t[:, ::1] i_sigma = np.empty((aa, 2), dtype=np.float64)
    for ii in range(aa):
        mask[...] = 0; s_mask[...] = 0
        streak_mask(streaks[ii], structure, width, mask)
        streak_mask(source_streaks[ii], structure, width, s_mask)
        cnt = 0; s_cnt = 0
        bgd_mean = 0; bgd_var = 0; I = 0; s_mean = 0; s_var = 0
        for i in range(a):
            for j in range(b):
                if mask[i, j]:
                    cnt += 1
                    delta = (background[i, j] - bgd_mean)
                    bgd_mean += delta / cnt
                    bgd_var += (background[i, j] - bgd_mean) * delta
                    I += cor_data[i, j]
                if s_mask[i, j]:
                    s_cnt += 1
                    delta = (background[i, j] - s_mean)
                    s_mean += delta / s_cnt
                    s_var += (background[i, j] - s_mean) * delta
        bgd_sigma = max(bgd_mean * cnt, bgd_var)
        i_sigma[ii, 0] = I / (s_mean * s_cnt)
        i_sigma[ii, 1] = sqrt(I + bgd_sigma + s_var)
    return np.asarray(i_sigma)