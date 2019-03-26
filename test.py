import cbc, logging, datetime, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    waist, f, ap = 12e-7, 2.0, 9e-2
    wavelength = 1.5e-7
    a, b, c = 2e-6, 2e-6, 3e-6
    Nx, Ny, Nz = 60, 60, 1

    detNx, detNy = 512, 512
    pix_size = 55e-3 / 4
    det_dist = 30
    knum = 1000
    zr = np.pi * waist**2 / wavelength

    axis = np.random.rand(3)
    theta = 2 * np.pi * np.random.random()

    logpath = os.path.join('logs', str(datetime.date.today()) + '.log')
    beam = cbc.CircBeam(f, ap, wavelength)
    diff = cbc.Diff(beam=beam, setup_args=cbc.SetupArgs(handler=logging.FileHandler(logpath), relpath='results/'),
                    det_args=cbc.DetArgs(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size), lat_args=cbc.LatArgs(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz))

    # diff.rotate_lat(axis, theta)
    diff.move_lat([0.501 * a, 0.501 * b, 1e-4])
    
    start = timer()
    diffres = diff.henry().pool()
    # diffres2 = diff.conv(knum).pool()
    # diffres3 = diff.nocoh(knum).pool()
    # print('Estimated time: %fs' % (timer() - start))
    # diffres.write()
    # diffres2.write()
    # diffres3.write()
    diffres.plot()
    # diffres2.plot()
    # diffres3.plot()