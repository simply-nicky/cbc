import cbc, logging, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    wavelength = 1.14e-7
    # f, ap, defoc = 2, 2e-2, 1e-4
    waist = wavelength / np.pi / 0.015
    Nx, Ny, Nz = 200, 200, 200
    detNx, detNy = 512, 512
    pix_size = 55e-3
    det_dist = 20

    astar = np.array([0.0109864094612009, -0.00551908483885947, 0.00294352907953398])
    bstar = np.array([0.00576393741858660, 0.0112435046699143, -0.000431835526544485])
    cstar = np.array([-0.00505686021507011, 0.00357471961041716, 0.0255767535096894])
    aa = np.cross(bstar, cstar) / (np.cross(bstar, cstar).dot(astar)) * 1e-7
    bb = np.cross(cstar, astar) / (np.cross(cstar, astar).dot(bstar)) * 1e-7
    cc = np.cross(astar, bstar) / (np.cross(astar, bstar).dot(cstar)) * 1e-7

    # th_div = 2 * ap / f
    th_div = wavelength / np.pi / waist
    dz = Nx * np.sqrt(aa.dot(aa)) / th_div

    # axis = np.random.rand(3)
    # theta = 2 * np.pi * np.random.random()

    logpath = cbc.utils.get_logpath()
    beam = cbc.GausBeam(waist, wavelength)
    diff = cbc.Diff(beam=beam, setup_args=cbc.SetupArgs(handler=logging.FileHandler(logpath)),
                    det_args=cbc.DetArgs(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size),
                    # cell_args=cbc.CellArgs.importpdb('4et8.pdb'),
                    # cell_args=cbc.CellArgs(XS=np.array([0, 0.5*a]), YS=np.zeros(2), ZS=np.zeros(2), bs=np.zeros(2), elems=['Au', 'Ag']),
                    lat_args=cbc.LatArgs(a=aa, b=bb, c=cc, Nx=Nx, Ny=Ny, Nz=Nz))

    # diff.rotate_lat(axis, theta)
    diff.move_lat([0, 0, dz])
    
    start = timer()
    diffres = diff.henry().pool()
    diffres.write()
    print('Estimated time: %fs' % (timer() - start))