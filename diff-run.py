"""
File: diff-run.py (Python 2.X and 3.X)

An example of using cbc package for calculating and saving diffraction pattern.
"""
import cbc, logging, datetime, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    waist = 4e-6
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    detNx, detNy = 512, 512
    pix_size = 55e-3 / 4
    det_dist = 54

    axis = np.random.rand(3)
    theta = 2 * np.pi * np.random.random()

    logpath = os.path.join('logs', str(datetime.date.today()) + '.log')
    beam = cbc.GausBeam(waist, wavelength)
    diff = cbc.Diff(beam=beam, setup_args=cbc.SetupArgs(handler=logging.FileHandler(logpath), relpath='results/'),
                    det_args=cbc.DetArgs(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size), lat_args=cbc.LatArgs(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz))
    diff.rotate_lat(axis, theta)
    start = timer()
    diffres = diff.henry().pool()
    print('Estimated time: %fs' % (timer() - start))
    diffres.write()