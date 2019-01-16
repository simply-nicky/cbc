from src import main as cbc
import numpy as np
import multiprocessing as mp
import concurrent.futures
from timeit import default_timer as timer
from functools import partial


if __name__ == "__main__":   
    waist = 2e-5
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    DetNx, DetNy = 4, 4
    pix_size = 55e-3
    det_dist = 54
    sigma = pix_size**2/det_dist**2
    kouts = cbc.kout(det_dist, DetNx, DetNy, pix_size)

    # kxs, kys, kzs = cbc.kout_grid(det_dist, DetNx, DetNy, pix_size)
    # start = timer()
    # diffs = cbc.diff_grid(kxs, kys, kzs, lat_pts, asf, waist, pix_size**2 / det_dist**2, wavelength)
    # print(timer() - start)

    thdiv = wavelength / np.pi / waist
    asf = cbc.asf_advanced('src/asf_hw_Au.txt', 'src/asf_q_fit_Au_2.txt', wavelength)
    lat_orig = [0, 0, Nx * c / thdiv]
    lat_pts = cbc.lattice(a, b, c, Nx, Ny, Nz, lat_orig)
    us = np.array([cbc.gaussian(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    kins = np.array([cbc.kin(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    
    worker = partial(cbc.diff_work, lat_pts=lat_pts, kins=kins, us=us, asf_hw='src/asf_hw_Au.txt', asf_fit='src/asf_q_fit_Au_2.txt', wavelength=wavelength, sigma=sigma)
    
    # pool = mp.Pool(processes = mp.cpu_count())
    # start = timer()
    # result = pool.map(worker, kouts)
    # print(timer() - start)

    start = timer()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for kout, diff in zip(kouts, executor.map(worker, kouts)):
            print(kouts.index(kout), ":", diff)
    
    