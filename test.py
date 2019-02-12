import cbc, logging, datetime, os
from timeit import default_timer as timer
import numpy as np

if __name__ == "__main__":
    waist = 4e-6
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    detNx, detNy = 512, 512
    pix_size = 55e-3 / 4
    det_dist = 54
    knum = 1000

    # axis = np.random.rand(3)
    # theta = 2 * np.pi * np.random.random()

    logpath = os.path.join('logs', str(datetime.date.today()) + '.log')
    diff = cbc.diff(setup_args=cbc.setup_args(handler=logging.FileHandler(logpath), relpath='results/'), kout_args=cbc.kout_args(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size), lat_args=cbc.lat_args(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz), waist=waist, wavelength=wavelength)
    
    kins1, _kdx = cbc.kins_grid(2 * diff.thdiv, int(np.sqrt(knum)))
    kins2 = cbc.normal(0, diff.thdiv, knum)
    kins3 = cbc.uniform(knum)
    kins4 = cbc.kins(diff.lat_pts, diff.waist, diff.wavelength)
    np.random.shuffle(kins4)
    kins4 = kins4[0:knum, 0:2]

    # diff.rotate_lat(axis, theta)
    
    start = timer()
    diffress = [diff.diff_noinfr(kins) for kins in (kins1, kins2, kins3, kins4)] 
    print('Estimated time: %f' % (timer() - start))
    for diffres in diffress:
        diffres.write()
