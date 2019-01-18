import cbc
import numpy as np
from timeit import default_timer as timer

if __name__ == "__main__":   
    waist = 2e-5
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    detNx, detNy = 32, 32
    pix_size = 55e-3
    det_dist = 54

    diff = cbc.diff(kout_args=cbc.kout_args(detNx=detNx, detNy=detNy))

    start = timer()
    diffres = diff.diff_pool()
    print(timer() - start)
    diffres.write()
    diffres.plot()