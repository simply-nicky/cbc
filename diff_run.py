"""
diff_run.py - convergent beam crystallography example
"""
import logging
from timeit import default_timer as timer
import numpy as np
import cbc

WL = 1.14e-7
TH_DIV = 0.01
REC_BASIS = np.array([[0.00551908483885947, -0.00294352907953398, 0.0109864094612009],
                      [-0.0112435046699143, 0.000431835526544485, 0.00576393741858660],
                      [-0.00357471961041716, -0.0255767535096894, -0.00505686021507011]]) * 1e7
LAT_NA, LAT_NB, LAT_NC, LAT_R = 100, 100, 100, 5e-4
DET_NX, DET_NY = 2000, 2000
PIX_SIZE = 88.6e-3
DET_DIST = 250

def main(rec_basis=REC_BASIS,
         lat_r=LAT_R,
         th_div=TH_DIV,
         det_dist=DET_DIST,
         det_nx=DET_NX,
         det_ny=DET_NY,
         pix_size=PIX_SIZE,
         wavelength=WL):
    # waist = wavelength / np.pi / th_div
    focus = 1.
    aperture = 2 * focus * np.tan(th_div)
    delta_z = lat_r / th_div / 2
    basis = cbc.rec_basis(rec_basis)
    logpath = cbc.utils.get_logpath()
    detector = cbc.Detector(det_dist=det_dist, det_nx=det_nx, det_ny=det_ny, pix_size=pix_size)
    diff = cbc.DiffYar(beam=cbc.RectLens(focus, aperture, wavelength),
                       handler=logging.FileHandler(logpath),
                       detector=detector,
                       lattice=cbc.BallLattice(basis_a=basis[0],
                                               basis_b=basis[1],
                                               basis_c=basis[2],
                                               lat_r=lat_r))
    diff.move_lat([0, 0, delta_z])
    start = timer()
    diffres = diff.calculate().pool()
    diffres.write()
    print('Estimated time: %fs' % (timer() - start))

if __name__ == "__main__":
    main()
