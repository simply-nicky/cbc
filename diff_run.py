"""
diff_run.py - convergent beam crystallography example
"""
import logging
from timeit import default_timer as timer
import numpy as np
import cbc

WL = 1.14e-7
TH_DIV = 0.01
A_REC = np.array([0.00551908483885947, -0.00294352907953398, 0.0109864094612009]) * 1e7
B_REC = np.array([-0.0112435046699143, 0.000431835526544485, 0.00576393741858660]) * 1e7
C_REC = np.array([-0.00357471961041716, -0.0255767535096894, -0.00505686021507011]) * 1e7
LAT_NA, LAT_NB, LAT_NC, LAT_R = 100, 100, 100, 5e-4
DET_NX, DET_NY = 2000, 2000
PIX_SIZE = 88.6e-3
DET_DIST = 250

def main(a_rec=A_REC,
         b_rec=B_REC,
         c_rec=C_REC,
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
    delta_z = lat_r / th_div
    a_vec, b_vec, c_vec = cbc.rec_basis(a_rec, b_rec, c_rec)
    logpath = cbc.utils.get_logpath()
    detector = cbc.Detector(det_dist=det_dist, det_nx=det_nx, det_ny=det_ny, pix_size=pix_size)
    diff = cbc.DiffYar(beam=cbc.RectLens(focus, aperture, wavelength),
                       handler=logging.FileHandler(logpath),
                       detector=detector,
                       lattice=cbc.BallLattice(basis_a=a_vec, basis_b=b_vec, basis_c=c_vec, lat_r=lat_r))
    diff.move_lat([0, 0, delta_z])
    start = timer()
    diffres = diff.calculate().pool()
    diffres.write()
    print('Estimated time: %fs' % (timer() - start))

if __name__ == "__main__":
    main()
