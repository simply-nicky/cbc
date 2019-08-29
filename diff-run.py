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
    a, b, c = np.array([7.9e-6, 0, 0]), np.array([0, 7.9e-6, 0]), np.array([0, 0, 3.8e-6])
    Na, Nb, Nc = 20, 20, 20

    detNx, detNy = 512, 512
    pixsize = 55e-3 / 4
    detdist = 54

    axis = np.random.rand(3)
    theta = 2 * np.pi * np.random.random()

    logpath = cbc.utils.get_logpath()
    beam = cbc.GausBeam(waist, wavelength)
    diff = cbc.Diff(beam=beam, setup=cbc.Setup(handler=logging.FileHandler(logpath)),
                    detector=cbc.Detector(detdist=detdist, Nx=detNx, Ny=detNy, pixsize=pixsize),
                    lattice=cbc.CubicLattice(a=a, b=b, c=c, Na=Na, Nb=Nb, Nc=Nc))
    diff.rotate_lat(axis, theta)
    start = timer()
    diffres = diff.calculate().pool()
    diffres.write()
    print('Estimated time: %fs' % (timer() - start))