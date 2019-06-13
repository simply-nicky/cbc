import cbc, logging, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    wavelength = 1.5e-7
    f, ap, defoc = 2, 2e-2, 1e-4
    waist = 4.5e-6
    a, b, c = 7.9e-6, 7.9e-6, 3.8e-6
    Nx, Ny, Nz = 20, 20, 1
    detNx, detNy = 512, 512
    pix_size = 55e-3 / 2
    det_dist = 30

    knum = 20000
    zr = np.pi * waist**2 / wavelength
    th_lens = 2 * ap / f
    dz = Nx * a / th_lens

    axis = np.random.rand(3)
    theta = 2 * np.pi * np.random.random()

    logpath = cbc.utils.get_logpath()
    beam = cbc.CircBeam(f, ap, wavelength)
    diff = cbc.Diff(beam=beam, setup_args=cbc.SetupArgs(handler=logging.FileHandler(logpath)),
                    det_args=cbc.DetArgs(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size),
                    cell_args=cbc.CellArgs.importpdb('4et8.pdb'),
                    # cell_args=cbc.CellArgs(XS=np.array([0, 0.5*a]), YS=np.zeros(2), ZS=np.zeros(2), bs=np.zeros(2), elems=['Au', 'Ag']),
                    lat_args=cbc.LatArgs(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz))

    # diff.rotate_lat(axis, theta)
    diff.move_lat([0.5001 * a, 0.5001 * b, dz])
    
    # start = timer()
    diffres = diff.henry().pool()
    # diffres2 = diff.conv(knum).pool()
    # diffres3 = diff.nocoh(knum).pool()
    # print('Estimated time: %fs' % (timer() - start))
    diffres.write()
    # diffres2.write()
    # diffres3.write()
    # diffres.plot()
    # diffres2.plot()
    # diffres3.plot()