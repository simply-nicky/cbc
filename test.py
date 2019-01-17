from src import main as cbc
import numpy as np
import concurrent.futures
from timeit import default_timer as timer
from functools import partial
import matplotlib.pyplot as plt

if __name__ == "__main__":   
    waist = 2e-5
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    DetNx, DetNy = 16, 16
    pix_size = 55e-3
    det_dist = 54
    sigma = pix_size**2/det_dist**2
    kouts = cbc.kouts(det_dist, DetNx, DetNy, pix_size)

    thdiv = wavelength / np.pi / waist
    lat_orig = [0, 0, Nx * c / thdiv]
    lat_pts = cbc.lattice(a, b, c, Nx, Ny, Nz, lat_orig)
    us = np.array([cbc.gaussian(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    kins = np.array([cbc.kin(*pt, waist=waist, wavelength=wavelength) for pt in lat_pts])
    
    worker = partial(cbc.diff_work, lat_pts=lat_pts, kins=kins, us=us, asf_hw='src/asf_hw_Au.txt', asf_fit='src/asf_q_fit_Au_2.txt', sigma=sigma, wavelength=wavelength)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        diffs = [diff for diff in executor.map(worker, kouts)]
    
    grid, diff_grid = cbc.make_grid(kouts, diffs)

    plt.contourf(*grid, np.abs(diff_grid))
    plt.show() 