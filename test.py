from src import main as cbc
import numpy as np
from timeit import default_timer as timer
from random import shuffle

waist = 2e-5
wavelength = 1.5e-7
thdiv = wavelength / np.pi / waist

a, b, c = 2e-5, 2.5e-5, 3e-5
Nx, Ny, Nz = 20, 20, 20
lat_orig = [0, 0, Nz * c / thdiv]
lat_pts = cbc.lattice(a, b, c, Nx, Ny, Nz, lat_orig)

DetNx, DetNy = 512, 512
pix_size = 55e-3
det_dist = 54
kouts = cbc.kout(det_dist, DetNx, DetNy, pix_size)
shuffle(kouts)

asf = cbc.asf_fit_parser('src/asf_q_fit_Au.txt')

diffs = cbc.lattice_diff_gen(kouts, lattice_pts=lat_pts, asf=asf, waist=waist, wavelength=wavelength)
start = timer()
difflist = [next(diffs) for _ in range(int(1e0))]
print(timer() - start)
print(difflist[0])