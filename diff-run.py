"""
File: diff-run.py (Python 2.X and 3.X)

An example of using cbc package for calculating and saving diffraction pattern.
"""

import cbc, h5py
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from functools import partial

if __name__ == "__main__":   
    waist = 2e-5
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    detNx, detNy = 32, 32
    pix_size = 55e-3
    det_dist = 54

    start = timer()
    diff = cbc.diff(setup_args=cbc.setup_args(relpath='results'), kout_args=cbc.kout_args(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size), lat_args=cbc.lat_args(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz), waist=waist, wavelength=wavelength)
    diffres = diff.diff_pool()
    print(timer() - start)
    start = timer()
    diffres.write()
    print(timer() - start)