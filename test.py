import cbc, logging, datetime, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    waist = 4e-6
    wavelength = 1.5e-7
    a, b, c = 2e-6, 2.5e-6, 1e-5
    Nx, Ny, Nz = 10, 10, 4

    detNx, detNy = 128, 128
    pix_size = 55e-3
    det_dist = 30
    knum = 5000
    zr = np.pi * waist**2 / wavelength

    axis = np.random.rand(3)
    theta = 2 * np.pi * np.random.random()

    logpath = os.path.join('logs', str(datetime.date.today()) + '.log')
    diff = cbc.diff(setup_args=cbc.setup_args(handler=logging.FileHandler(logpath), relpath='results/'), kout_args=cbc.kout_args(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size), lat_args=cbc.lat_args(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz), waist=waist, wavelength=wavelength)

    # diff.rotate_lat(axis, theta)
    diff.move_lat(2 * zr)
    
    start = timer()
    diffres = diff.henry().pool()
    # diffres.write()
    print('Estimated time: %fs' % (timer() - start))
    diffres.plot()
    # diffres2.plot()