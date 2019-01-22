"""
File: test.py (Python 2.X and 3.X)

A test example of using cbc package for plotting and saving results.
"""

import cbc, h5py
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

if __name__ == "__main__":   
    waist = 2e-5
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    detNx, detNy = 16, 16
    pix_size = 55e-3
    det_dist = 54

    start = timer()
    diff = cbc.diff(kout_args=cbc.kout_args(detNx=detNx, detNy=detNy))
    dres = diff.diff_pool()
    dres.write()
    print(timer() - start)

    # f = h5py.File("diff_18-01-2019_17_31.hdf5", 'r')
    # data = []
    # results = f[list(f)[1]]
    # for key in results.keys():
    #     data.append(results[key][:])
    # f.close()
    # plt.contourf(data[1], data[2], np.abs(data[0]))
    # plt.show()