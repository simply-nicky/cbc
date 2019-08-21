import cbc, logging, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    wavelength = 1.14e-7
    # f, ap, defoc = 2, 2e-2, 1e-4
    thdiv = 0.015
    waist = wavelength / np.pi / thdiv
    Nx, Ny, Nz = 100, 100, 100
    r = 2e-4
    detNx, detNy = 2000, 2000
    pix_size = 88.6e-3
    det_dist = 250
    dz = Nx * np.sqrt(aa.dot(aa)) / thdiv

    astar = np.array([0.00551908483885947, -0.00294352907953398, 0.0109864094612009])
    bstar = np.array([-0.0112435046699143, 0.000431835526544485, 0.00576393741858660])
    cstar = np.array([-0.00357471961041716, -0.0255767535096894, -0.00505686021507011])
    aa = np.cross(bstar, cstar) / (np.cross(bstar, cstar).dot(astar)) * 1e-7
    bb = np.cross(cstar, astar) / (np.cross(cstar, astar).dot(bstar)) * 1e-7
    cc = np.cross(astar, bstar) / (np.cross(astar, bstar).dot(cstar)) * 1e-7

    logpath = cbc.utils.get_logpath()
    beam = cbc.GausBeam(waist, wavelength)
    diff = cbc.Diff(beam=beam, setup=cbc.Setup(handler=logging.FileHandler(logpath)),
                    detector=cbc.Detector(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size),
                    lattice=cbc.CubicLattice(a=aa, b=bb, c=cc, Nx=Nx, Ny=Ny, Nz=Nz))

    diff.move_lat([0, 0, dz])
    
    start = timer()
    diffres = diff.calculate().pool()
    diffres.write()
    print('Estimated time: %fs' % (timer() - start))