"""
File: diff-run.py (Python 2.X and 3.X)

An example of using cbc package for calculating and saving diffraction pattern.
"""
import cbc, logging, datetime, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    waist = 2e-5
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    detNx, detNy = 1024, 1024
    pix_size = 55e-3 / 4
    det_dist = 54

    axis = np.random.rand(3)
    theta = 2 * np.pi * np.random.random()

    logpath = os.path.join('logs', str(datetime.date.today()) + '.log')
    diff = cbc.diff(setup_args=cbc.setup_args(handler=logging.FileHandler(logpath), relpath='results/'), kout_args=cbc.kout_args(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size), lat_args=cbc.lat_args(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz), waist=waist, wavelength=wavelength)
    diff.rotate_lat(axis, theta)
    start = timer()
    diffres1 = diff.diff_noinfr(knum=100)
    diffres2 = diff.diff_pool()
    print('Estimated time: %f' % (timer() - start))
    diffres1.write()
    diffres2.write()