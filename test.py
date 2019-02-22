import cbc, logging, datetime, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    waist = 1e-5
    wavelength = 1.5e-7
    a, b, c = 2e-6, 2.5e-6, 3e-5
    Nx, Ny, Nz = 20, 20, 1

    detNx, detNy = 512, 512
    pix_size = 55e-3 / 2
    det_dist = 30
    knum = 100
    zr = np.pi * waist**2 / wavelength

    axis = np.random.rand(3)
    theta = 2 * np.pi * np.random.random()

    logpath = os.path.join('logs', str(datetime.date.today()) + '.log')
    diff = cbc.diff(setup_args=cbc.setup_args(handler=logging.FileHandler(logpath), relpath='results/'), kout_args=cbc.kout_args(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size), lat_args=cbc.lat_args(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz), waist=waist, wavelength=wavelength)

    # diff.rotate_lat(axis, theta)
    diff.move_lat(zr)
    
    start = timer()
    diffres = diff.conv(1000).pool()
    diffres.plot()