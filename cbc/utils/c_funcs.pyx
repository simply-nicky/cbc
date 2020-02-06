import numpy as np
cimport numpy as cnp
from scipy.special.cython_special cimport j0
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from cpython.mem cimport PyMem_Malloc,  PyMem_Free
from libc.math cimport sqrt, cos, sin, exp, pi
from scipy import LowLevelCallable

ctypedef cnp.complex128_t complex_t
ctypedef cnp.float64_t float_t
ctypedef cnp.int64_t int_t

cdef double rectlens_re(double xx, void* user_data):
    cdef:
        double x = (<double*> user_data)[0], z = (<double*> user_data)[1]
        double k = (<double*> user_data)[2], focus = (<double*> user_data)[3]
    return cos(k * xx**2 / 2 * (1 / focus - 1 / z) + k / z * x * xx)

cdef double rectlens_im(double xx, void* user_data):
    cdef:
        double x = (<double*> user_data)[0], z = (<double*> user_data)[1]
        double k = (<double*> user_data)[2], focus = (<double*> user_data)[3]
    return -sin(k * xx**2 / 2 * (1 / focus - 1 / z) + k / z * x * xx)

cdef double circlens_re(double rr, void* user_data):
    cdef:
        double r = (<double*> user_data)[0], z = (<double*> user_data)[1]
        double k = (<double*> user_data)[2], focus = (<double*> user_data)[3]
    return cos(k * rr**2 / 2 * (1 / focus - 1 / z)) * j0(k * r * rr / z) * 2 * pi * rr

cdef double circlens_im(double rr, void* user_data):
    cdef:
        double r = (<double*> user_data)[0], z = (<double*> user_data)[1]
        double k = (<double*> user_data)[2], focus = (<double*> user_data)[3]
    return -sin(k * rr**2 / 2 * (1 / focus - 1 / z)) * j0(k * r * rr / z) * 2 * pi * rr

cdef object pack_data(double x, double z, double k, double focus):
    cdef double* data_ptr = <double*> PyMem_Malloc(4 * sizeof(double))
    data_ptr[0] = x; data_ptr[1] = z; data_ptr[2] = k; data_ptr[3] = focus
    return PyCapsule_New(<void*> data_ptr, NULL, free_data)

cdef void free_data(capsule):
    PyMem_Free(PyCapsule_GetPointer(capsule, NULL))

def llc_rectlens_re(double x, double z, double k, double focus):
    func_capsule = PyCapsule_New(<void*> rectlens_re, "double (double, void *)", NULL)
    data_capsule = pack_data(x, z, k, focus)
    return LowLevelCallable(func_capsule, data_capsule)

def llc_rectlens_im(double x, double z, double k, double focus):
    func_capsule = PyCapsule_New(<void*> rectlens_im, "double (double, void *)", NULL)
    data_capsule = pack_data(x, z, k, focus)
    return LowLevelCallable(func_capsule, data_capsule)

def llc_circlens_re(double r, double z, double k, double focus):
    func_capsule = PyCapsule_New(<void*> circlens_re, "double (double, void *)", NULL)
    data_capsule = pack_data(r, z, k, focus)
    return LowLevelCallable(func_capsule, data_capsule)

def llc_circlens_im(double r, double z, double k, double focus):
    func_capsule = PyCapsule_New(<void*> circlens_im, "double (double, void *)", NULL)
    data_capsule = pack_data(r, z, k, focus)
    return LowLevelCallable(func_capsule, data_capsule)

def rotation_matrix(float_t[::1] axis, float_t theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.

    axis - rotation axis
    theta - rotation angle
    """
    cdef:
        float_t axis_norm = sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2), a = cos(theta / 2.0)
        float_t b = -axis[0] / axis_norm * sin(theta / 2.0)
        float_t c = -axis[1] / axis_norm * sin(theta / 2.0)
        float_t d = -axis[2] / axis_norm * sin(theta / 2.0)
        float_t[:, ::1] m = np.empty((3, 3), dtype=np.float64)
    m[0, 0] = a * a + b * b - c * c - d * d
    m[0, 1] = 2 * (b * c + a * d)
    m[0, 2] = 2 * (b * d - a * c)
    m[1, 0] = 2 * (b * c - a * d)
    m[1, 1] = a * a + c * c - b * b - d * d
    m[1, 2] = 2 * (c * d + a * b)
    m[2, 0] = 2 * (b * d + a * c)
    m[2, 1] = 2 * (c * d - a * b)
    m[2, 2] = a * a + d * d - b * b - c * c
    return np.asarray(m)

def phase_inc(float_t[:, :, ::1] kins, float_t[:, ::1] xs, float_t[:, ::1] ys, float_t[:, ::1] zs, float_t wl):
    cdef:
        int a = xs.shape[0], b = xs.shape[1], i, j
        float_t ph
        complex_t[:, ::1] phase = np.empty((a, b), dtype=np.complex128)
    for i in range(a):
        for j in range(b):
            ph = kins[i, j, 0] * xs[i, j] + kins[i, j, 1] * ys[i, j] + kins[i, j, 2] * zs[i, j]
            phase[i, j] = cos(2 * pi / wl * ph) - sin(2 * pi / wl * ph) * 1j
    return np.asarray(phase)

def diff_calc(float_t[:, ::1] kouts, float_t[:, :, ::1] kins, float_t[:, ::1] xs, float_t[:, ::1] ys,
              float_t[:, ::1] zs, float_t[:, ::1] asf_coeffs, complex_t[:, ::1] us, float_t wl):
    cdef:
        int a = kouts.shape[0], b = xs.shape[0], c = xs.shape[1], i, j, k
        complex_t[::1] diff = np.empty(a, dtype=np.complex128)
        complex_t res
        float_t q, ph
    for i in range(a):
        res = 0.0
        for j in range(b):
            for k in range(c):
                q = ((kouts[i, 0] - kins[j, k, 0])**2 + (kouts[i, 1] - kins[j, k, 1])**2 + (kouts[i, 2] - kins[j, k, 2])**2) / 4e14 / wl**2
                ph = kouts[i, 0] * xs[j, k] + kouts[i, 1] * ys[j, k] + kouts[i, 2] * zs[j, k]
                res += (us[j, k] * (asf_coeffs[k, 0] * exp(-asf_coeffs[k, 1] * q) + asf_coeffs[k, 2]) *
                        exp(-asf_coeffs[k, 3] * q) * (cos(2 * pi / wl * ph) + sin(2 * pi / wl * ph) * 1j))
        diff[i] = res
    return np.asarray(diff)