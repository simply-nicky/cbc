"""
File: diff-run.py (Python 2.X and 3.X)

An example of using cbc package for calculating and saving diffraction pattern.
"""
import cbc, logging, datetime, os
from timeit import default_timer as timer


if __name__ == "__main__":   
    waist = 2e-5
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    detNx, detNy = 1024, 1024
    pix_size = 55e-3 / 2
    det_dist = 54

    logpath = os.path.join('logs', str(datetime.date.today()) + '.log')
    start = timer()
    diff = cbc.diff(setup_args=cbc.setup_args(handler = logging.FileHandler(logpath), relpath = 'new_results/'), kout_args=cbc.kout_args(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size), lat_args=cbc.lat_args(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz), waist=waist, wavelength=wavelength)
    diff.move_lat()
    diffres = diff.diff_pool()
    diffres.write()
    print('Estimated time: %f' % (timer() - start))