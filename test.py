import cbc, logging, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    wavelength = 1.14e-7
    thdiv = 0.015
    waist = wavelength / np.pi / thdiv
    Na, Nb, Nc = 100, 100, 100
    r = 5e-4
    detNx, detNy = 2000, 2000
    pixsize = 88.6e-3
    detdist = 250

    astar = np.array([0.00551908483885947, -0.00294352907953398, 0.0109864094612009])
    bstar = np.array([-0.0112435046699143, 0.000431835526544485, 0.00576393741858660])
    cstar = np.array([-0.00357471961041716, -0.0255767535096894, -0.00505686021507011])
    aa = np.cross(bstar, cstar) / (np.cross(bstar, cstar).dot(astar)) * 1e-7
    bb = np.cross(cstar, astar) / (np.cross(cstar, astar).dot(bstar)) * 1e-7
    cc = np.cross(astar, bstar) / (np.cross(astar, bstar).dot(cstar)) * 1e-7
    dz = r / thdiv

    logpath = cbc.utils.get_logpath()
    beam = cbc.GausBeam(waist, wavelength)
    diff = cbc.Diff(beam=beam, setup=cbc.Setup(handler=logging.FileHandler(logpath)),
                    detector=cbc.Detector(detdist=detdist, Nx=detNx, Ny=detNy, pixsize=pixsize),
                    lattice=cbc.BallLattice(a=aa, b=bb, c=cc, r=r))

    diff.move_lat([0, 0, dz])
    
    start = timer()
    diffres = diff.calculate().pool()
    diffres.write()
    print('Estimated time: %fs' % (timer() - start))