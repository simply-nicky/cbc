"""
File: diff-slides.py (Python 2.X and 3.X)

An example of using cbc package for calculating and saving diffraction pattern series.
"""

import cbc, logging, os, datetime
import numpy as np

if __name__ == "__main__":
    waist = 4e-6
    wavelength = 1.5e-7
    a, b, c = 2e-5, 2.5e-5, 3e-5
    Nx, Ny, Nz = 20, 20, 20

    detNx, detNy = 512, 512
    pix_size = 55e-3
    det_dist = 54

    zmax = max(a * Nx, b * Ny, c * Nz) * np.pi * waist / wavelength
    lat_origs = [[0,0,z] for z in np.linspace(0, zmax, num=11, endpoint=True)]

    for counter, lat_orig in enumerate(lat_origs):
        logpath = os.path.join('logs', str(datetime.date.today()) + '.log')
        relpath = os.path.join('results/zseries', str(counter))
        setup_args = cbc.setup_args(handler=logging.FileHandler(logpath), relpath=relpath)
        lat_args = cbc.lat_args(a=a, b=b, c=c, Nx=Nx, Ny=Ny, Nz=Nz, lat_orig=lat_orig)
        kout_args= cbc.kout_args(det_dist=det_dist, detNx=detNx, detNy=detNy, pix_size=pix_size)
        diff = cbc.diff(setup_args=setup_args, lat_args=lat_args, kout_args=kout_args, waist=waist, wavelength=wavelength)
        diffres = diff.diff_pool()
        diffres.write()