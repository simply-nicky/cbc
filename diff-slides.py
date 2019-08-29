"""
File: diff-slides.py (Python 2.X and 3.X)

An example of using cbc package for calculating and saving diffraction pattern series.
"""

import cbc, logging, os, datetime
import numpy as np

if __name__ == "__main__":
    waist = 4e-6
    wavelength = 1.5e-7
    a, b, c = np.array([7.9e-6, 0, 0]), np.array([0, 7.9e-6, 0]), np.array([0, 0, 3.8e-6])
    Na, Nb, Nc = 20, 20, 20

    detNx, detNy = 512, 512
    pixsize = 55e-3
    detdist = 54

    zmax = max(np.sqrt(a.dot(a)) * Na, np.sqrt(b.dot(b)) * Nb, np.sqrt(c.dot(c)) * Nc) * np.pi * waist / wavelength
    pts = np.linspace(0, zmax, num=11, endpoint=True)
    beam = cbc.GausBeam(waist, wavelength)

    for counter, pt in enumerate(pts):
        logpath = cbc.utils.get_logpath()
        relpath = os.path.join('results/zseries', str(counter))
        setup = cbc.Setup(handler=logging.FileHandler(logpath), relpath=relpath)
        lattice = cbc.CubicLattice(a=a, b=b, c=c, Na=Na, Nb=Nb, Nc=Nc)
        detector = cbc.Detector(detdist=detdist, Nx=detNx, Ny=detNy, pixsize=pixsize)
        diff = cbc.Diff(beam=beam, setup=setup, lattice=lattice, detector=detector)
        diff.move_lat(pt)
        diffres = diff.calculate().pool()
        diffres.write()